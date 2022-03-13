#!/usr/bin/env python

import argparse
import json
import logging
import os
import sys
import re

sys.path.append(os.getcwd())
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from tensorboardX import SummaryWriter

# for tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import torch
from torch.utils.data import DataLoader

from prompt_files.lightning_utils.callbacks import Seq2SeqLoggingCallback, get_early_stopping_callback, get_checkpoint_callback

from prompt_files.lightning_utils.utils import (
    assert_all_frozen,
    freeze_embeds,
    freeze_params,
    label_smoothed_nll_loss,
    pickle_save,
    save_json,
)


# need the parent dir module
from prompt_files.lightning_utils.lightning_base import BaseTransformer, add_generic_args, generic_train, AdamW, Adafactor
from prompt_files.t5_model import T5Tokenizer
from prompt_files.t5_model.tokenization_t5_fast import T5TokenizerFast
from prompt_files.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from prompt_files.t5_model.tokenization_t5 import T5Tokenizer

from prompt_files.prompts_config import PROMPT_TOKENS, UNUSED_TOKENS, META_PROMPT_TOKENS
from prompt_files.p_tuning.modeling_prompt_t5 import T5ForPromptEncDecDST
# from prompt_files.p_tuning.modeling_prefix_t5 import T5ForPrefixDST
from prompt_files.p_tuning.todcl.todcl_domain_dataset import T5DomainPromptDSTDatasetTodcl, GPT2DomainPromptDSTDatasetTodcl
# from prompt_files.p_tuning.modeling_prompt_gpt import GPTForPromptDST



import random
import numpy.random as np_random
logger = logging.getLogger(__name__)

DEVICE='cuda' if torch.cuda.device_count() > 0 else 'cpu'

def set_random_seed(rand_seed):
    random.seed(rand_seed)
    np_random.seed(rand_seed)
    torch.manual_seed(rand_seed)


class PromptDSTModule(BaseTransformer):
    loss_names = ["loss"]
    val_metric = 'loss'

    def __init__(self, hparams, **kwargs):
        self.hparams = hparams
        if self.hparams.sortish_sampler and self.hparams.gpus > 1:
            self.hparams.replace_sampler_ddp = False
        elif self.hparams.max_tokens_per_batch is not None:
            if self.hparams.gpus > 1:
                raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
            if self.hparams.sortish_sampler:
                raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")

        if self.hparams.model_type == 't5':
            if self.hparams.prefix_tuning:
                model = T5ForPrefixDST.from_pretrained(self.hparams.model_name_or_path)
            else:
                model = T5ForPromptEncDecDST.from_pretrained(self.hparams.model_name_or_path)
            # tokenizer = T5TokenizerFast.from_pretrained(self.hparams.model_name_or_path)
            tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_name_or_path)
            tokenizer.add_tokens(UNUSED_TOKENS)
            assert len(tokenizer) == 32128
            self.dataset_class = T5DomainPromptDSTDatasetTodcl
        elif self.hparams.model_type == 'gpt2':
            model = GPTForPromptDST.from_pretrained(self.hparams.model_name_or_path)
            tokenizer = GPT2TokenizerFast.from_pretrained(self.hparams.model_name_or_path)
            self.dataset_class = GPT2DomainPromptDSTDatasetTodcl
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            model.config.pad_token_id = tokenizer.pad_token_id
            tokenizer.padding_side = 'left'
            model.resize_token_embeddings(len(tokenizer))
        else:
            raise
        assert len(tokenizer) == model.config.vocab_size
        tokenizer.add_tokens(PROMPT_TOKENS)
        tokenizer.add_tokens(META_PROMPT_TOKENS)
        # tokenizer.add_tokens(MLM_PROMPT_TOKENS)

        super().__init__(self.hparams, num_labels=None, model=model, tokenizer=tokenizer, **kwargs)

        # initialization for prompt embeds
        self.config.prompt_style = self.hparams.prompt_style

        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)

        self.step_count = 0
        self.metrics = defaultdict(list)
        self.model_type = self.config.model_type
        self.vocab_size = len(self.tokenizer)

        if self.hparams.freeze_embeds:
            freeze_embeds(self.model)
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        self.num_workers = self.hparams.num_workers
        self.decoder_start_token_id = None  # default to config

        assert self.hparams.dataset == 'multiwoz'

        self.dataset = {'train': None, 'val': None, 'test': None}

        self.already_saved_batch = False

        self.my_train_logger = SummaryWriter(os.path.join(self.hparams.output_dir, 'training_logs/train'))
        self.my_val_logger = SummaryWriter(os.path.join(self.hparams.output_dir, 'training_logs/val'))
        self.num_step = 0

        if not hasattr(self.hparams, 'embedding_initialization'):
            self.hparams.embedding_initialization = 'random'
        self.question_prompt_token_mappings = {}
        # self.initialize_prompt()

        self.save_hyperparameters()


    def assign_prompt_embeddings(self):
        # initialize with domain/slot embeddings
        initialize_prompt_to_token_ids = {}
        dataset = self.get_dataset('train')
        tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_name_or_path)
        prompt_style = self.hparams.prompt_style
        assert sum([_ == 'd' for _ in prompt_style]) == 1
        assert sum([_ == 's' for _ in prompt_style]) == 1
        assert 'd' in prompt_style.split('|')[0]
        assert 's' in prompt_style.split('|')[0]
        d_pos = prompt_style.index('d')
        domain_prompt_len = int(prompt_style[d_pos + 1])
        s_pos = prompt_style.index('s')
        assert d_pos < s_pos
        prefix_prompt_len = sum([int(_) for _ in prompt_style[:d_pos]])

        for ds in dataset.questions:
            print(ds)
            domain, slot = ds.split('_')[1].split('-')
            enc_p, dec_p = dataset.questions[ds]
            print(enc_p)
            prompt_ids = re.findall('\<prompt_(\d*)\>', enc_p)
            domain_prompt_ids = prompt_ids[prefix_prompt_len:]
            domain_prompt_ids = [int(_) for _ in domain_prompt_ids]
            domain_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(domain))
            initialize_prompt_to_token_ids.update({k: v for k, v in zip(domain_prompt_ids, domain_token_ids)})
            slot_prompt_ids = prompt_ids[(prefix_prompt_len + domain_prompt_len) :]
            slot_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(slot))
            initialize_prompt_to_token_ids.update({k: v for k, v in zip(slot_prompt_ids, slot_token_ids)})
            print(initialize_prompt_to_token_ids)
        self.model.assign_prompt_embedding(initialize_prompt_to_token_ids)


    def initialize_prompt(self):
        assert self.model_type in ['t5', 'gpt2']
        print('re-initializing prompt embedder!')
        self.model.initialize_prompt_embedder(self.hparams.embedding_initialization)

        # assign domain/slot embedding to prompt embedder
        if 'd' in self.hparams.prompt_style:
            self.assign_prompt_embeddings()
        torch.save(self.model, os.path.join(self.hparams.output_dir, 'init_model.ckpt'))


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.prompt_embedder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
        ]
        if self.hparams.prefix_tuning:
            optimizer_grouped_parameters.append({
                "params": [p for n, p in model.prefix_mlps.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            })
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
        )
        return optimizer


    def save_readable_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """A debugging utility"""
        readable_batch = {
            k: self.tokenizer.batch_decode(v.tolist()) if "mask" not in k else v.shape for k, v in batch.items()
        }
        save_json(readable_batch, Path(self.output_dir) / "text_batch.json")
        save_json({k: v.tolist() for k, v in batch.items()}, Path(self.output_dir) / "tok_batch.json")

        self.already_saved_batch = True
        return readable_batch

    def forward(self, **kwargs):
        return self.model(**kwargs)


    def display_batch(self, batch):
        if self.hparams.model_type in ['t5', 'bart']:
            input_ids_womask = batch['input_ids_womask']
            decoder_input_ids_womask = batch.get('decoder_input_ids_womask', None)
            target_ids_womask = batch['target_ids_womask']
            for i, (s, din, t) in enumerate(zip(input_ids_womask, decoder_input_ids_womask, target_ids_womask)):
                print('===========source_womask============')
                print(self.tokenizer.decode(s).replace('<pad>', ''))
                print('============decoder input womask=============')
                print(din)
                print(self.tokenizer.decode(din))
                print('===========target womask============')
                print(t)
                print(self.tokenizer.decode(t))
                break
        else:
            for i, (s, t) in enumerate(zip(batch['gpt_input_ids'][:8], batch['gpt_target_ids'][:8])):
                inp_tokens = [_ for _ in self.tokenizer.convert_ids_to_tokens(s)]
                tar_tokens = [_ for _ in self.tokenizer.convert_ids_to_tokens(t)]
                print('==========================')
                # show_list = [(it, tt) for it, tt in zip(inp_tokens, tar_tokens)]
                # same_token = [a == b for a, b in show_list]
                print('===========input ============')
                print(inp_tokens)
                print('===========target ============')
                print(tar_tokens)
                # print('===========same tokens============')
                # print(same_token)
                # input()


    def _step(self, batch: dict, show=False) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        # if self.num_step == 0:
        #     self.display_batch(batch)

        if self.hparams.model_type in ['t5', 'bart']:
            input_ids_womask = batch['input_ids_womask']
            decoder_input_ids_womask = batch.get('decoder_input_ids_womask', None)
            attention_mask_womask = batch['attention_mask_womask']
            labels = batch['target_ids_womask']
            labels[labels >= self.model.vocab_size] = pad_token_id

            outputs = self(
                input_ids=input_ids_womask,
                attention_mask=attention_mask_womask,
                decoder_input_ids=decoder_input_ids_womask,
                use_cache=False,
                output_hidden_states=True,
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

        elif self.hparams.model_type in ['gpt2']:
            input_ids = batch['gpt_input_ids']
            labels = batch['gpt_target_ids']

            outputs = self(
                input_ids=input_ids,
                labels=labels,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            loss = outputs.loss
        else:
            raise


        return (loss,)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        show = False
        if self.hparams.small_sample_run and batch_idx == 0:
            show = True
        loss_tensors = self._step(batch, show=show)
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        for name, loss in logs.items():
            self.my_train_logger.add_scalar('{}'.format(name), loss, self.num_step)
        self.num_step += 1
        return logs

    def validation_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        if batch_idx == 0:
            self._generative_step(batch, 'val')
        return base_metrics


    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        if prefix != 'val':
            return {}
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        for name, loss in losses.items():
            self.my_val_logger.add_scalar(name, loss, self.num_step)
        loss = losses["loss"]
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count
        metrics.update({f"{prefix}_loss": loss})
        return metrics

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)


    def _generative_step(self, batch: dict, type_path) -> dict:
        assert not self.model.training
        if self.model_type == 'gpt2':
            gen_input_ids = batch['gen_input_ids']
            max_length = gen_input_ids.shape[1] + 100
            outputs = self.model.generate(
                input_ids=gen_input_ids,
                use_cache=False,
                return_dict_in_generate=True,
                max_length=max_length,
            )
            outputs.sequences = outputs.sequences[:, gen_input_ids.shape[1]:]
            raise NotImplementedError
        elif self.model_type in ['bart', 't5']:
            dst_input_ids = batch['input_ids_womask'].to(DEVICE)
            dst_input_mask = batch['attention_mask_womask'].to(DEVICE)
            dst_decoder_input_ids = batch.get('decoder_inputs_womask', None)
            input_seqs_womask = self.tokenizer.batch_decode(batch['input_ids_womask'])
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
            dst_predictions = [_.replace('<pad>', '') for _ in dst_predictions]
            dst_predictions = [_.replace('</s>', '') for _ in dst_predictions]
            dst_predictions = [re.sub('\<(prompt_)(.*?)\>', '', s) for s in dst_predictions]
            dst_predictions = [_.strip() for _ in dst_predictions]
            dst_predictions = [re.sub('\<(extra_id_)(.*?)\>', '|', s) for s in dst_predictions]
            target_values = [re.sub('\<(extra_id_)(.*?)\>', '|', s) for s in target_seqs_womask]
            ret = {'acc': 0, 'total': len(dst_predictions)}
            # print(self.tokenizer.batch_decode(dst_input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True))
            # print(dst_predictions)
            for i, (pred_str, target_str) in enumerate(zip(dst_predictions, target_values)):
                pred_list = pred_str.split('|')
                pred_list = [_.strip() for _  in pred_list]
                target_list = target_str.split('|')
                target_list = [_.strip() for _  in target_list]
                if all([a == b for a, b in zip(pred_list, target_list)]):
                    ret['acc'] += 1
        else:
            raise

        # print('validation:')
        # for i in range(1):
        #     print('')
        #     print('xx'*50)
        #     print(batch['ds'][i])
        #     print('=========input womask===========')
        #     # print(input_seqs_womask[i].replace('<pad>', ''))
        #     print(input_seqs_womask[i])
        #     print('===========generation womask===============')
        #     print(dst_predictions[i])
        #     print('===========target womask===============')
        #     print(target_seqs_womask[i])
        return ret

    def test_step(self, batch, batch_idx):
        self._generative_step(batch, 'test')

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path):
        dataset = self.dataset_class(
            tokenizer=self.tokenizer,
            data_dir=self.hparams.data_dir,
            type_path=type_path,
            small_sample_run=self.hparams.small_sample_run,
            prompt_style=self.hparams.prompt_style,
        )

        return dataset


    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        if self.dataset[type_path] is None:
            self.dataset[type_path] = self.get_dataset(type_path)

        self.dataset[type_path].reset(self.step_count)
        print('{} dataset total length: {}'.format(type_path, len(self.dataset[type_path])))
        sampler = self.dataset[type_path].make_sortish_sampler(batch_size)
        print('{} sampler total length: {}'.format(type_path, len(sampler)))

        return DataLoader(
            self.dataset[type_path],
            collate_fn=self.dataset[type_path].collate_fn,
            num_workers=self.num_workers,
            batch_sampler=sampler,
        )

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)

        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--overwrite_output_dir", action="store_true", default=False)
        parser.add_argument("--max_tokens_per_batch", type=int, default=None)
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
        parser.add_argument(
            '--dataset',
            default='multiwoz'
        )
        parser.add_argument(
            '--data_dir',
        )
        parser.add_argument(
            '--small_sample_run',
            action='store_true'
        )
        parser.add_argument(
            '--prompt_style',
            nargs='+',
            help='single number means use n tokens as prompt.'
                 'character D/S means domain name / slot name'
                 'character E means <extra_id_0>'
                 'e.g. [1, D, 1, S, 1, E, 1] -> <p1> domain_name <p2> slot_name <p3> <extra_id_0> <p4>'
                 'if character d/s means prompt tokens that are initialized using domain/slot tokens'
                 'e.g. [d, 5, s, 5, e] -> [tokenizer(domain) (pad to 5 length)] + [tokenizer(slot) (pad to 5 length)] + [<extra_id_0>]'
        )
        parser.add_argument('--model_type')
        parser.add_argument('--prefix_tuning', action='store_true')
        parser.add_argument('--embedding_initialization', default='random')
        return parser



def prepare_parser():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = PromptDSTModule.add_model_specific_args(parser, os.getcwd())
    return parser


def cover_unused_args(args):
    args.early_stopping_patience = -1

    # if 'multiwoz21' in args.data_dir:
    #     args.data_version = 'multiwoz21'
    # elif 'multiwoz23' in args.data_dir:
    #     args.data_version = 'multiwoz23'
    # elif 'augment_multiwoz22' in args.data_dir:
    #     args.data_version = 'augment_multiwoz22'


def main(args) -> PromptDSTModule:
    Path(args.output_dir).mkdir(exist_ok=True)
    sample_log_dir = os.path.join(args.output_dir, 'sample_log')
    os.makedirs(sample_log_dir, exist_ok=True)
    args.sample_log_dir = sample_log_dir

    if len(os.listdir(args.output_dir)) > 3 and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    model: PromptDSTModule = PromptDSTModule(args)
    model.initialize_prompt()

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.output_dir, './logs'))

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False

    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(os.path.join(args.output_dir, 'checkpoints'), model.val_metric, save_top_k=100),
        early_stopping_callback=es_callback,
        logger=tb_logger,
        val_ratio=1.0,
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")

    if not args.do_predict:
        return model

    model.freeze()

    trainer.test(model, test_dataloaders=model.test_dataloader())
    return model


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()
    cover_unused_args(args)

    print(args)
    set_random_seed(42)

    # import time
    # while True:
    #     total0, used0 = check_mem(0)
    #     total1, used1 = check_mem(1)
    #     if int(used0) < 1000 and int(used1) < 1000:
    #         break
    #     time.sleep(5)
    #     print('waiting')

    main(args)
