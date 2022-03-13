import json
import random
import re

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from prompt_files.p_tuning.todcl.todcl_domain_finetune import PromptDSTModule
import os
import torch
import pytorch_lightning as pl
from prompt_files.prompt_dataset import T5PromptGenFullStateDataset
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup, AdamW,
)
from typing import Dict
from prompt_files.lightning_utils.lightning_base import BaseTransformer, add_generic_args
from prompt_files.p_tuning.modeling_prompt_t5 import T5ForPromptEncDecDST
from prompt_files.prompt_test import predict_t5
from prompt_files.t5_model.tokenization_t5 import T5Tokenizer
from prompt_files.prompts_config import PROMPT_TOKENS, UNUSED_TOKENS, META_PROMPT_TOKENS
from prompt_files.prompt_dataset import MemT5PromptDSTDataset
from prompt_files.prompt_test import predict_t5
from pathlib import Path
from prompt_files.lightning_utils.utils import (
    label_smoothed_nll_loss,
    pickle_save,
    save_json,
)
from collections import defaultdict
import re
from prompt_files.transformer_utils import logging

logger = logging.get_logger(__name__)
# logger.setLevel(logging.INFO)

DEVICE = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'


def prepare_prompt_parser(parser):
    parser = pl.Trainer.add_argparse_args(parser)
    parser = PromptDSTModule.add_model_specific_args(parser, os.getcwd())
    return parser


class MemT5PromptDSTModule(BaseTransformer):
    loss_names = ["loss"]
    val_metric = 'loss'

    def __init__(self, hparams, **kwargs):
        self.fake_examples = []
        self.hparams = hparams

        assert self.hparams.model_type == 't5', 'not support model type {} for MemT5PromptDSTModule'.format(
            self.hparams.model_type)
        model = T5ForPromptEncDecDST.from_pretrained(self.hparams.model_name_or_path)
        tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_name_or_path)
        tokenizer.add_tokens(UNUSED_TOKENS)
        assert len(tokenizer) == model.config.vocab_size
        tokenizer.add_tokens(PROMPT_TOKENS)
        tokenizer.add_tokens(META_PROMPT_TOKENS)

        super().__init__(self.hparams, num_labels=None, model=model, tokenizer=tokenizer, **kwargs)

        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)

        self.metrics = defaultdict(list)

        self.dataset_class = MemT5PromptDSTDataset
        self.dataset = defaultdict(self.dataset_class)

        self.my_train_logger = SummaryWriter(os.path.join(self.hparams.output_dir, 'training_logs/train'))
        self.my_val_logger = SummaryWriter(os.path.join(self.hparams.output_dir, 'training_logs/val'))
        self.my_meta_train_logger = SummaryWriter(os.path.join(self.hparams.output_dir, 'training_logs/meta_train'))
        self.my_meta_val_logger = SummaryWriter(os.path.join(self.hparams.output_dir, 'training_logs/meta_val'))
        self.num_step = 0

        self.save_hyperparameters()

        self.task_list_seen = []
        self.cur_domain = None
        self.training_prompt_name = 'prompt'
        self.forward_domains = []
        self.backward_domain = None

        if hparams.same_pos_emb_for_prompts:
            self.model.set_same_prompt_pos_emb()

        self.initialize_prompt()

        self.random_generator = random.Random(42)

    def set_dataset(self, train_dataset, val_dataset):
        self.dataset = {'train': train_dataset, 'dev': val_dataset}

    def initialize_prompt(self):
        print('re-initializing prompt embedder!', self.hparams.embedding_initialization)
        self.model.initialize_prompt_embedder(self.hparams.embedding_initialization)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        print('optimize {}!'.format(self.training_prompt_name))
        if self.training_prompt_name == 'prompt':
            optimizer_grouped_parameters = [
                {
                    "params": model.prompt_embedder.parameters(),
                    "weight_decay": self.hparams.weight_decay,
                    "lr": self.hparams.learning_rate
                },
                {
                    "params": model.prompt_bias.parameters(),
                    "weight_decay": 0.0,
                    "lr": self.hparams.learning_rate
                },
            ]
        elif self.training_prompt_name == 'meta_prompt':
            optimizer_grouped_parameters = [
                {
                    "params": model.meta_prompt_embedder.parameters(),
                    "weight_decay": self.hparams.weight_decay,
                    "lr": self.hparams.meta_lr
                },
                {
                    "params": model.meta_prompt_bias.parameters(),
                    "weight_decay": 0.0,
                    "lr": self.hparams.meta_lr
                },
            ]
        elif self.training_prompt_name == 'first_prompt':
            optimizer_grouped_parameters = [
                {
                    "params": model.prompt_embedder.parameters(),
                    "weight_decay": self.hparams.weight_decay,
                    "lr": self.hparams.first_lr
                },
                {
                    "params": model.prompt_bias.parameters(),
                    "weight_decay": 0.0,
                    "lr": self.hparams.first_lr
                },
            ]
        else:
            raise

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
        )
        return optimizer

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def display_sample(self, batch):
        for i in range(len(batch)):
            input_ids_womask = batch['input_ids_womask'][i]
            decoder_input_ids_womask = batch['decoder_input_ids_womask'][i]
            target_ids_womask = batch['target_ids_womask'][i]
            logger.info("===========input===========")
            logger.info(self.tokenizer.decode(input_ids_womask).replace('<pad>', ''))
            logger.info("===========decoder input===========")
            logger.info(self.tokenizer.decode(decoder_input_ids_womask))
            logger.info("===========target output===========")
            logger.info(self.tokenizer.decode(target_ids_womask))
            logger.info("===========end===========")

    def _step(self, batch: dict) -> Dict[str, any]:
        # TODO: add backward transfer, "batch" could be a dict {"a": batch_a, "b": batch_b}
        pad_token_id = self.tokenizer.pad_token_id

        # supervised loss
        input_ids_womask = batch['input_ids_womask']
        decoder_input_ids_womask = batch.get('decoder_input_ids_womask', None)
        attention_mask_womask = batch['attention_mask_womask']
        labels = batch['target_ids_womask']
        labels[labels >= self.model.vocab_size] = pad_token_id

        outputs = self.model(
            input_ids=input_ids_womask,
            attention_mask=attention_mask_womask,
            decoder_input_ids=decoder_input_ids_womask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )

        if self.hparams.label_smoothing == 0:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
            lm_logits = outputs.logits
            assert lm_logits.shape[-1] == self.model.config.vocab_size
            loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.hparams.label_smoothing, ignore_index=pad_token_id
            )

        return {"loss": loss}

    def _generative_step(self, batch: dict) -> dict:
        assert not self.model.training
        dst_input_ids = batch['input_ids_womask'].to(DEVICE)
        dst_input_mask = batch['attention_mask_womask'].to(DEVICE)
        dst_decoder_input_ids = batch.get('decoder_inputs_womask', None)
        target_seqs_womask = batch['target_seqs_womask']
        max_length = 100

        if dst_decoder_input_ids is not None:
            outputs = self.model.generate(
                input_ids=dst_input_ids,
                attention_mask=dst_input_mask,
                decoder_input_ids=dst_decoder_input_ids.to(DEVICE),
                use_cache=False,
                return_dict_in_generate=True,
                max_length=max_length,
            )
        else:
            outputs = self.model.generate(
                input_ids=dst_input_ids,
                attention_mask=dst_input_mask,
                use_cache=False,
                return_dict_in_generate=True,
                max_length=max_length,
            )
        dst_predictions = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False,
                                                      clean_up_tokenization_spaces=True)
        dst_predictions = [_.replace('<pad>', '').replace('</s>', '').strip() for _ in dst_predictions]
        dst_predictions = [re.sub('\<(extra_id_)(.*?)\>', '|', s) for s in dst_predictions]
        target_values = [re.sub('\<(extra_id_)(.*?)\>', '|', s) for s in target_seqs_womask]
        ret = {'acc': 0, 'total': len(dst_predictions)}

        for i, (pred_str, target_str) in enumerate(zip(dst_predictions, target_values)):
            pred_list = pred_str.split('|')
            pred_list = [_.strip() for _ in pred_list]
            target_list = target_str.split('|')
            target_list = [_.strip() for _ in target_list]
            if all([a == b for a, b in zip(pred_list, target_list)]):
                ret['acc'] += 1
        return ret

    def training_step(self, batch, batch_idx) -> Dict:
        if self.backward_domain:
            self.memory_batch = batch['memory_dataloader']
            batch = batch['dataloader']
        loss_tensors = self._step(batch)
        if self.num_step == 0:
            self.display_sample(batch)
        if self.training_prompt_name in ['prompt', 'first_prompt']:
            tb_logger = self.my_train_logger
        elif self.training_prompt_name == 'meta_prompt':
            tb_logger = self.my_meta_train_logger
        else:
            raise
        for name, loss in loss_tensors.items():
            tb_logger.add_scalar('{}/{}'.format(name, self.cur_domain), loss, self.num_step)
        self.num_step += 1
        return loss_tensors

    def backward(self, loss, optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        if not self.backward_domain:
            loss.backward(*args, **kwargs)
        else:
            loss.backward(*args, **kwargs)
            g_ori = self.model.meta_prompt_embedder.weight.grad.data.clone()
            self.zero_grad()
            loss_tensors = self._step(self.memory_batch)
            loss_tensors['loss'].backward(*args, **kwargs)
            g_ref = self.model.meta_prompt_embedder.weight.grad.data.clone()
            g_ori_flat = g_ori.view(-1)
            g_ref_flat = g_ref.view(-1)
            grad_sim = torch.sum(g_ori_flat * g_ref_flat).item()
            if grad_sim > 0:
                self.model.meta_prompt_embedder.weight.grad = g_ori
                self.pos_grad += 1
            else:
                self.neg_grad += 1
                self.zero_grad()
                # g_ref_norm = torch.norm(g_ref_flat).item()
                # g_proj = g_ori_flat - grad_sim / g_ref_norm * g_ref_flat
                # self.model.meta_prompt_embedder.weight.grad = g_proj.view(g_ori.size())

            # print(loss_tensors['loss'], grad_sim)

    def validation_step(self, batch, batch_idx, only_loss=False):
        base_metrics = self._step(batch)
        if not only_loss:
            base_metrics.update(self._generative_step(batch))
        return base_metrics

    def validation_epoch_end(self, validation_step_outputs, prefix="val"):
        losses = {k: torch.stack([x[k] for x in validation_step_outputs]).mean() for k in validation_step_outputs[0] if
                  isinstance(validation_step_outputs[0][k], torch.Tensor)}
        if self.backward_domain:
            print('pos grad batch: {}, neg grad batch: {}'.format(self.pos_grad, self.neg_grad))
            self.pos_grad, self.neg_grad = 0, 0
        if self.training_prompt_name in ['prompt', 'first_prompt']:
            tb_logger = self.my_val_logger
        elif self.training_prompt_name == 'meta_prompt':
            tb_logger = self.my_meta_val_logger
        else:
            raise
        for name, loss in losses.items():
            tb_logger.add_scalar('{}/{}'.format(name, self.cur_domain), loss, self.num_step)
        metrics = losses.copy()
        if 'acc' in validation_step_outputs[0]:
            for dst_key in ['acc', 'total']:
                metrics[dst_key] = sum([x[dst_key] for x in validation_step_outputs])
            metrics['jga'] = metrics['acc'] / metrics['total']
            tb_logger.add_scalar('jga/{}'.format(self.cur_domain), metrics['jga'], self.num_step)
        return metrics

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False):
        """
        need to set below variables outside:
            self.cur_domain
            self.training_prompt_name
            self.forward_domains
            self.backward_domain: if is not None, return additional dataloader for memory samples
        """
        logger.info('load {} loader for {} on {} domain'.format(type_path, self.training_prompt_name, self.cur_domain))
        assert type_path in ['train', 'dev']
        if self.hparams.multi:
            assert self.training_prompt_name == 'meta_prompt'
            dataloader = DataLoader(
                self.dataset[type_path],
                num_workers=self.hparams.num_workers,
                collate_fn=self.dataset[type_path].collate_fn,
                batch_sampler=self.dataset[type_path].make_full_data_sampler(batch_size=batch_size))
        else:
            # TODO
            # ['RAND_PROMPT', 'FW_PROMPT', 'FWBW_PROMPT']
            if self.hparams.CL == 'RAND_PROMPT':
                assert len(self.forward_domains) == 0
            else:
                if self.hparams.CL == 'FW_PROMPT':
                    if self.training_prompt_name == 'meta_prompt':
                        do_real_augment = (type_path=='train') and self.hparams.aug_train_metaprompt
                        if type_path == 'train' and self.current_epoch == 0:
                            do_generate_fake = (type_path=='train') and self.hparams.generate_fake_example
                            if do_generate_fake:
                                self.fake_examples = self.generate_fake_examples(self.forward_domains)
                        self.dataset[type_path].resample_dataset(self.cur_domain, self.training_prompt_name,
                                                                 self.forward_domains, do_augment=do_real_augment,
                                                                 fake_examples=self.fake_examples)
                    else:
                        self.dataset[type_path].resample_dataset(self.cur_domain, self.training_prompt_name)
                elif self.hparams.CL == 'FWBW_PROMPT':
                    if self.training_prompt_name == 'meta_prompt':
                        do_real_augment = (type_path == 'train') and self.hparams.aug_train_metaprompt
                        if type_path == 'train' and self.current_epoch == 0:
                            do_generate_fake = (type_path=='train') and self.hparams.generate_fake_example
                            if do_generate_fake:
                                self.fake_examples = self.generate_fake_examples([self.backward_domain])
                        if self.backward_domain is None:
                            self.dataset[type_path].resample_dataset(self.cur_domain, self.training_prompt_name,
                                                                     self.forward_domains, do_augment=do_real_augment,
                                                                     fake_examples=self.fake_examples)
                        else:
                            self.dataset[type_path].resample_dataset(self.cur_domain, self.training_prompt_name,
                                                                     do_augment=do_real_augment,
                                                                     backward_domain=self.backward_domain,
                                                                     fake_examples=self.fake_examples)
                            self.pos_grad, self.neg_grad = 0, 0
                    else:
                        self.dataset[type_path].resample_dataset(self.cur_domain, self.training_prompt_name)

            if self.hparams.CL == 'FWBW_PROMPT' and self.backward_domain is not None and type_path == 'dev':
                assert self.training_prompt_name == 'meta_prompt'
                self.dataset['train'].resample_dataset(self.cur_domain, self.training_prompt_name,
                                                       backward_domain=self.backward_domain)
                memory_dataloader = DataLoader(
                    self.dataset['train'],
                    num_workers=self.hparams.num_workers,
                    collate_fn=self.dataset['train'].collate_fn,
                    batch_sampler=self.dataset['train'].make_domain_sampler(batch_size=batch_size,
                                                                            target_domain='memory'))
                logger.info('memory_dataloader len {}'.format(len(memory_dataloader)))
                return memory_dataloader

            dataloader = DataLoader(
                self.dataset[type_path],
                num_workers=self.hparams.num_workers,
                collate_fn=self.dataset[type_path].collate_fn,
                batch_sampler=self.dataset[type_path].make_domain_sampler(batch_size=batch_size,
                                                                          target_domain=self.cur_domain))

            if self.hparams.CL == 'FWBW_PROMPT' and self.backward_domain is not None and type_path == 'train':
                assert self.training_prompt_name == 'meta_prompt'
                memory_dataloader = DataLoader(
                    self.dataset[type_path],
                    num_workers=self.hparams.num_workers,
                    collate_fn=self.dataset[type_path].collate_fn,
                    batch_sampler=self.dataset[type_path].make_domain_sampler(batch_size=batch_size,
                                                                              target_domain='memory'))
                self.memory_dataloader = memory_dataloader
                logger.info('dataloader len {}'.format(len(dataloader)))
                logger.info('memory_dataloader len {}'.format(len(memory_dataloader)))
                return {'dataloader': dataloader, 'memory_dataloader': memory_dataloader}

        logger.info('dataloader len {}'.format(len(dataloader)))
        return dataloader

    def train_dataloader(self):
        # reload every epoch
        return self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    def prepare_val_dataloader(self):
        # prepare once before training
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def setup(self, mode):
        return

    def initialize_prompt_by_trained_prompt(self, prompt_init_dict):
        print('initializing prompt using trained prompts')
        print(list(prompt_init_dict.items())[:10])
        self.model.initialize_prompt_by_trained_prompt(prompt_init_dict)

    def initialize_prompt_by_trained_metaprompt(self, prompt_init_dict):
        print('initializing prompt using trained meta prompts')
        print(list(prompt_init_dict.items())[:10])
        self.model.initialize_prompt_by_trained_metaprompt(prompt_init_dict)

    def initialize_metaprompt_by_trained_prompt(self, prompt_init_dict):
        print('initializing meta prompt using trained prompts')
        print(list(prompt_init_dict.items())[:10])
        self.model.initialize_metaprompt_by_trained_prompt(prompt_init_dict)

    def generate_fake_examples(self, forward_domains):
        if len(forward_domains) == 0:
            return []
        dials = self.dataset['train'].dialogs[self.cur_domain]
        fake_dials = self.random_generator.choices(dials, k=min(300, len(dials)))
        print('num fake_dials: {}'.format(len(fake_dials)))
        fake_examples = []
        for dial in fake_dials:
            prompt_domain = self.random_generator.choice(forward_domains)
            soft_prompt = self.dataset['train'].domain2soft_prompt(prompt_domain)
            e = self.dataset['train'].convert_dial_to_example(dial, prompt_domain, soft_prompt)
            fake_examples.append(e)
        for s in tqdm(range(0, len(fake_examples), self.hparams.valid_batch_size), desc='preparing fake examples'):
            batch_dials = fake_examples[s: s + self.hparams.valid_batch_size]
            batch = self.dataset['train'].collate_fn(batch_dials)
            batch_ds = [_['ds'] for _ in batch_dials]
            predict_t5(self.model, self.tokenizer, batch, batch_ds, batch_dials)
            for gen_example, ds_str in zip(batch_dials, batch_ds):
                ds_list = ds_str.split()
                target_seq = ''
                for extra_id_num, slot in enumerate(ds_list):
                    value = gen_example['pred_state'].get(slot, 'none')
                    target_seq += '<extra_id_{}>{}'.format(extra_id_num, value)
                gen_example['dst_target_sequence'] = target_seq.lower()
                input_seq = gen_example['dst_input_sequence']
                input_seq = re.sub('\<(prompt_)(.*?)\>', '', input_seq)
                input_seq += ''.join(['<meta_prompt_{}>'.format(i) for i in range(self.hparams.num_domain_prompt)])
                gen_example['dst_input_sequence'] = input_seq
        logger.info('dst_input_sequence: {}'.format(fake_examples[0]['dst_input_sequence']))
        logger.info('dst_target_sequence: {}'.format(fake_examples[0]['dst_target_sequence']))
        return fake_examples

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        return parser


def prepare_MemT5PromptDST_parser(parser):
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MemT5PromptDSTModule.add_model_specific_args(parser, os.getcwd())
    return parser


class T5PromptGenFullStateModule(MemT5PromptDSTModule):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.dataset_class = T5PromptGenFullStateDataset
        self.dataset = defaultdict(self.dataset_class)

    def validation_step(self, batch, batch_idx):
        base_metrics = self._step(batch)
        return base_metrics

