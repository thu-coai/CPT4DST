import os
import torch
from torch.nn import CrossEntropyLoss
from random import sample
import pytorch_lightning as pl
from transformers import (AdamW, GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, BartTokenizer,
                          BartForConditionalGeneration, T5ForConditionalGeneration)
from model.adapterGPT2 import GPT2Adapter
from model.promptGPT2 import GPT2Prompt
from transformers import T5TokenizerFast
from model.adapter_t5 import AdapterT5
from utils.dataloader import get_data_loaders, get_current_task_data, make_loader
from collections import defaultdict


class Seq2SeqToD(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        if "t5" in args.model_checkpoint:
            if args.CL == 'ADAPTER':
                model = AdapterT5.from_pretrained(args.model_checkpoint)
                model.add_adapters(bottleneck_size=args.bottleneck_size, adapter_num=args.number_of_adpt)
            else:
                model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint)
            tokenizer = T5TokenizerFast.from_pretrained(args.model_checkpoint)
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))
            self.model_type = 't5'
        elif "bart" in args.model_checkpoint:
            model = BartForConditionalGeneration.from_pretrained(args.model_checkpoint)
            tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint, bos_token="[bos]", eos_token="[eos]",
                                                      sep_token="[sep]")
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))
            self.model_type = 'bart'
        elif "gpt2" in args.model_checkpoint:
            if (args.CL == "ADAPTER"):
                model = GPT2Adapter.from_pretrained(args.model_checkpoint)
                model.add_adapters(bottleneck_size=args.bottleneck_size, adapter_num=args.number_of_adpt)
            elif args.CL == "PROMPT":
                model = GPT2Prompt.from_pretrained(args.model_checkpoint)
            else:
                model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
            tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint, bos_token="[bos]", eos_token="[eos]",
                                                      sos_token="[SOS]", sep_token="[sep]", pad_token='[PAD]')
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))
            self.model_type = 'gpt2'

        self.save_hyperparameters(args)
        self.model = model
        self.tokenizer = tokenizer
        self.lr = args.lr
        self.current_task = 0
        self.fisher = defaultdict(list)
        self.optpar = defaultdict(list)
        self.episodic_mem = defaultdict(list)
        self.CL = args.CL
        self.reg = args.reg
        self.first_task = True
        self.model_name = args.model_checkpoint
        self.reply_memory = []
        self.task_list_seen = []
        self.clinit = args.clinit

    def reload_model(self):
        print('reloading t5 model')
        assert self.model_type == 't5' and self.CL == 'VANILLA_BASELINE'
        model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        self.model = model

    def set_number_of_tasks(self, n_tasks):
        self.n_tasks = n_tasks

    def set_up_gem(self):
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        dev = next(self.model.parameters()).device
        self.grads = torch.Tensor(sum(self.grad_dims), self.n_tasks).to(dev)

    def compute_PPL(self, batch, task_id=-1, device='cuda'):
        assert self.model_type == 'gpt2'
        with torch.no_grad():
            lm_logits, *_ = self.model(
                input_ids=batch["input_id_PPL"].to(device),
                attention_mask=None,
                labels=None,
                task_id=task_id,
                return_dict=False
            )
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = batch["output_id_PPL"].to(device)[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = torch.reshape(loss, shift_labels.size())
        return (loss.sum(1) / (loss != 0).sum(1)).tolist()

    def training_step(self, batch, batch_idx):
        if self.CL == "GEM" and not self.first_task:
            dev = next(self.model.parameters()).device
            for id_task, (_, task_memory) in enumerate(self.episodic_mem.items()):
                batch_mem = sample(task_memory, 1)[0]  # ==> we sample one batch from episodic memory
                self.model.zero_grad()
                (loss), *_ = self.model(input_ids=batch_mem["encoder_input"].to(dev),
                                        attention_mask=batch_mem["attention_mask"].to(
                                            dev) if "gpt2" not in self.model_name else None,
                                        labels=batch_mem["decoder_output"].to(dev),
                                        return_dict=False
                                        )
                loss.backward()
                store_grad(self.model.parameters, self.grads, self.grad_dims, id_task)
            self.model.zero_grad()

        elif (self.CL == "AGEM" and not self.first_task):
            dev = next(self.model.parameters()).device
            batch_mem = sample(self.episodic_mem["all"], 1)[0]  # ==> we sample one batch from episodic memory
            self.model.zero_grad()
            (loss), *_ = self.model(input_ids=batch_mem["encoder_input"].to(dev),
                                    attention_mask=batch_mem["attention_mask"].to(
                                        dev) if "gpt2" not in self.model_name else None,
                                    labels=batch_mem["decoder_output"].to(dev),
                                    return_dict=False
                                    )
            loss.backward()
            grad_ref = []
            for p in self.model.parameters():
                if p.requires_grad:
                    grad_ref.append(p.grad.view(-1))
            grad_ref = torch.cat(grad_ref)  ## from eq. 10 of AGEM Paper

            self.model.zero_grad()

        # print(batch["encoder_input"].size())
        ## LOSS ON CURRENT DATA
        if self.CL == "ADAPTER":

            loss = self.model(
                input_ids=batch["encoder_input"],
                attention_mask=batch["attention_mask"],
                labels=batch["decoder_output"],
                task_id=self.task_list_seen.index(batch["task_id"][0]),
                return_dict=False
            )[0]
        else:
            loss = self.model(input_ids=batch["encoder_input"],
                              attention_mask=batch["attention_mask"],
                              labels=batch["decoder_output"],
                              return_dict=False)[0]

        if (self.CL == "AGEM" and not self.first_task):
            ## Code from https://github.com/GMvandeVen/continual-learning/blob/master/encoder.py#L244
            loss.backward()
            grad_cur = []
            for p in self.model.parameters():
                if p.requires_grad:
                    grad_cur.append(p.grad.view(-1))
            grad_cur = torch.cat(grad_cur)
            # -check inequality constrain
            angle = (grad_cur * grad_ref).sum()
            if angle < 0:
                # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                length_rep = (grad_ref * grad_ref).sum()
                grad_proj = grad_cur - (angle / length_rep) * grad_ref
                # -...and replace all the gradients within the model with this projected gradient
                index = 0
                for p in self.model.parameters():
                    if p.requires_grad:
                        n_param = p.numel()  # number of parameters in [p]
                        p.grad.copy_(grad_proj[index:index + n_param].view_as(p))
                        index += n_param
        elif self.CL == "GEM" and not self.first_task:
            loss.backward()
            store_grad(self.model.parameters, self.grads, self.grad_dims, id_task + 1)
            indx = torch.LongTensor([j for j in range(id_task + 1)])
            dotp = torch.mm(self.grads.to(dev)[:, id_task].unsqueeze(0),
                            self.grads.to(dev).index_select(1, indx.to(dev)))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads.to(dev)[:, id_task].unsqueeze(1),
                              self.grads.to(dev).index_select(1, indx.to(dev)), self.reg)
                # copy gradients back
                overwrite_grad(self.model.parameters, self.grads.to(dev)[:, id_task], self.grad_dims)

        elif self.CL == "L2" and not self.first_task:
            dev = next(self.model.parameters()).device
            l2_reg = 0

            for n, p in self.model.named_parameters():
                l = self.reg * (p - self.optpar[n].to(dev)).pow(2)
                l2_reg += l.sum()
            self.log('l2_reg', l2_reg, on_epoch=True)
            loss = loss + l2_reg
        elif self.CL == "EWC" and not self.first_task:
            dev = next(self.model.parameters()).device
            ewc_loss = 0
            for n, p in self.model.named_parameters():
                ## Eq (3) of https://arxiv.org/pdf/1612.00796.pdf
                l = self.reg * self.fisher[n].to(dev) * (p - self.optpar[n].to(dev)).pow(2)
                ewc_loss += l.sum()
            self.log('EWC_reg', ewc_loss, on_epoch=True)
            loss = loss + ewc_loss

        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if (self.CL == "ADAPTER"):
            with torch.no_grad():
                loss = self.model(input_ids=batch["encoder_input"],
                                  attention_mask=batch["attention_mask"],
                                  labels=batch["decoder_output"],
                                  task_id=self.task_list_seen.index(batch["task_id"][0]),
                                  return_dict=False
                                  )[0]
        else:
            # print(batch["encoder_input"].size())
            with torch.no_grad():
                loss = self.model(input_ids=batch["encoder_input"],
                                  attention_mask=batch["attention_mask"],
                                  labels=batch["decoder_output"],
                                  return_dict=False
                                  )[0]
        self.log('val_loss', loss)

        # if batch_idx == 0:
        #     outputs = self.model.generate(
        #         input_ids=batch["encoder_input"],
        #         attention_mask=batch["attention_mask"],
        #         use_cache=False,
        #         return_dict_in_generate=True,
        #         max_length=100,
        #     )
        #     dst_predictions = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False,
        #                                                   clean_up_tokenization_spaces=True)
        #     input_seq = self.tokenizer.batch_decode(batch["encoder_input"])
        # for inp, out in zip(input_seq, dst_predictions):
        #     print('='*50)
        #     inp = inp.replace('<pad>', '')
        #     print(inp)
        #     print('-'*50)
        #     print(out)
        return loss

    def configure_optimizers(self):
        if self.CL == "ADAPTER":
            if self.model_type == 'gpt2':
                parameters_to_update = [p for n, p in self.named_parameters() if "adapter" in str(n)]
                return AdamW(parameters_to_update, lr=self.lr, correct_bias=True)
            elif self.model_type == 't5':
                parameters_to_update = [p for n, p in self.named_parameters() if "adapter" in str(n)]
                return AdamW(parameters_to_update, lr=self.lr, correct_bias=True)
        elif self.CL == "PROMPT":
            parameters_to_update = [p for n, p in self.named_parameters() if "prompt" in str(n)]
            return AdamW(parameters_to_update, lr=self.lr, correct_bias=True)
        else:
            return AdamW(self.parameters(), lr=self.lr, correct_bias=True)

    def backward(self, loss, optimizer, optimizer_idx):
        if (self.CL == "GEM" or self.CL == "AGEM") and not self.first_task:
            pass
        else:
            loss.backward()
