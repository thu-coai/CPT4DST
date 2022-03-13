import json
import os
import random

import torch
from torch.utils.data import Sampler, SubsetRandomSampler, SequentialSampler, BatchSampler, Dataset, RandomSampler

# from prompt_files.p_tuning.dataset import T5PromptDSTDataset


def t5_shift_tokens_right(input_ids):
    decoder_start_token_id = 0
    pad_token_id = 0

    assert (
            decoder_start_token_id is not None
    ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

    return shifted_input_ids


class MyBatchSampler(BatchSampler):
    def __init__(self, sampler: Sampler, batch_size: int, mini_batch_size: int, drop_last: bool) -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.mini_batch_size = mini_batch_size

    def __iter__(self):
        batch = []
        for idxs in self.sampler:
            batch.extend(idxs)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // (self.batch_size // self.mini_batch_size)  # type: ignore
        else:
            return (len(self.sampler) + (self.batch_size // self.mini_batch_size) - 1) // (
                        self.batch_size // self.mini_batch_size)  # type: ignore


class T5DomainPromptDSTDatasetTodcl(Dataset):
    # one prompt for each domain
    def __init__(self,
                 tokenizer,
                 data_dir,  # must contain 'train_dials.json'
                 type_path,
                 small_sample_run,
                 prompt_style,
                 test_file_path=None,  # only for test, the test file with predicted cl_domain
                 ):
        # load data
        self.random_generator = random.Random(42)
        if test_file_path is None:
            self.src_file = os.path.join(data_dir, '{}_dials.json'.format(type_path))
        else:
            assert type_path == 'test'
            self.src_file = test_file_path
        print('loading dataset file from {}'.format(self.src_file))
        self.type_path = type_path
        self.dialogs = json.load(open(self.src_file))
        train_dialogs = json.load(open(os.path.join(data_dir, 'train_dials.json')))

        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.small_sample_run = small_sample_run

        # prepare prompt template
        # prompt_style: format of prompt for each slot
        self.prompt_style = prompt_style
        self.num_prompt_token_per_slot = sum([int(_) for _ in self.prompt_style if _.isdigit()])
        prompt_list = []
        assert 'E' in self.prompt_style
        if '|' not in self.prompt_style:
            self.prompt_style.append('|')
        for i, ch in enumerate(self.prompt_style):
            if ch.isdigit():
                prompt_list.extend(['<prompt_{}>' for _ in range(int(ch))])
            elif ch == 'D':
                prompt_list.append('<domain>')
            elif ch == 'S':
                prompt_list.append('<slot>')
            elif ch == 'E':
                prompt_list.append('<extra_id_0>')
            elif ch == '|':
                prompt_list.append('|')
        prompt = ''.join(prompt_list)
        if '<domain><slot>' in prompt:
            prompt = prompt.replace('<domain><slot>', '<domain> <slot>')
        self.prompt_template = prompt

        # prepare questions and ds and cl_domains
        self.ds = []
        self.cl_domain_to_ds = {}
        self.questions = {}
        for cl_domain_id, dials in train_dialogs.items():
            cl_domain = tuple(eval(cl_domain_id))
            if cl_domain not in self.cl_domain_to_ds:
                self.cl_domain_to_ds[cl_domain] = set()
            for dial in dials:
                state = dial['state']
                for ds in state:
                    self.cl_domain_to_ds[cl_domain].add(ds)
                    if ds not in self.ds:
                        self.ds.append(ds)
        self.ds = list(sorted(self.ds))
        for ds in self.ds:
            self.make_new_prompt(ds)
        self.cl_domains = list(sorted(list(self.cl_domain_to_ds.keys())))

        # prepare cl_domain to ds
        self.cl_domain_to_ds = {}
        for cl_domain_id, dials in train_dialogs.items():
            cl_domain = tuple(eval(cl_domain_id))
            if cl_domain not in self.cl_domain_to_ds:
                self.cl_domain_to_ds[cl_domain] = set()
            for dial in dials:
                for ds in dial['state']:
                    domain = ds.split('-')[0]
                    assert domain in cl_domain, print('domains in dial states must be in cl_domain')
                    self.cl_domain_to_ds[cl_domain].add(ds)
        self.cl_domain_to_ds = {k: list(sorted(list(v))) for k, v in self.cl_domain_to_ds.items()}

        # prepare examples
        self.cl_domain2examples = {k: [] for k in self.cl_domains}
        for cl_domain_id, dials in self.dialogs.items():
            cl_domain = tuple(eval(cl_domain_id))
            for dial in dials:
                self.cl_domain2examples[cl_domain].append(self.convert_dial_to_example(dial, cl_domain))
        if small_sample_run:
            self.cl_domain2examples = {k: v[:32] for k, v in self.cl_domain2examples.items()}

        self.cl_domain2numexamples = {k: len(self.cl_domain2examples[k]) for k in self.cl_domains}
        self.dataset_len = sum(self.cl_domain2numexamples.values())

    def make_new_prompt(self, ds):
        prompt_id_incr = len(self.questions) * self.num_prompt_token_per_slot
        prompt_ids = [prompt_id_incr + i for i in range(self.num_prompt_token_per_slot)]
        prompt = self.prompt_template.format(*(prompt_ids))
        self.questions[ds] = prompt.split('|')

    def reset(self, cur_epoch=-1):
        for cl_domain in self.cl_domain2examples:
            self.random_generator.shuffle(self.cl_domain2examples[cl_domain])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        example = None

        for cl_domain in self.cl_domains:
            domain_num = self.cl_domain2numexamples[cl_domain]
            if index < domain_num:
                example = self.cl_domain2examples[cl_domain][index]
                break
            index -= domain_num
        assert example is not None
        dst_input_seq, dst_target_seq = example['dst_input_sequence'], example['dst_target_sequence']

        return {
            'dst_input_sequence': dst_input_seq,
            'dst_target_sequence': dst_target_seq,
            'ds': example['ds'],
            'dst_generation_decoder_inputs': example['dst_generation_decoder_inputs'],
        }

    def make_sortish_sampler(self, batch_size, **kwargs):
        return SubsetRandomSampler(
            list(BatchSampler(SequentialSampler(list(range(self.dataset_len))), batch_size, drop_last=False)))
        # use mini-batch sampler
        # mini_batch_size = 8
        # assert self.dataset_len > mini_batch_size
        # return MyBatchSampler(
        #     SubsetRandomSampler(
        #         list(BatchSampler(SequentialSampler(list(range(self.dataset_len))), mini_batch_size, drop_last=True))),
        #     batch_size=batch_size, mini_batch_size=mini_batch_size, drop_last=False)

    def prepare_for_generation(self, dial, cl_domain):
        return self.convert_dial_to_example(dial, cl_domain)

    def convert_dial_to_example(self, dial, cl_domain):
        dss = self.cl_domain_to_ds[cl_domain]
        domain_enc_prompt = ''
        target_seq = ''
        ds_extra_id_num = 0
        dss_list = []
        for ds in self.ds:
            if ds in dss:
                enc_p, dec_p = self.questions[ds]
                domain_enc_prompt += enc_p.replace('<extra_id_0>', '<extra_id_{}>'.format(ds_extra_id_num))
                value = dial['state'].get(ds, 'none')
                target_seq += '<extra_id_{}>{}'.format(ds_extra_id_num, value)
                ds_extra_id_num += 1
                dss_list.append(ds)
        return {
            'dst_input_sequence': dial['history'] + domain_enc_prompt,
            'dst_target_sequence': target_seq,
            'ds': ' '.join(dss_list),
            'dial_id': dial['dataset'] + '_' + dial['dial_id'] + '_' + str(dial['turn_id']),
            'dst_generation_decoder_inputs': ''
        }

    def collate_fn(self, batch):
        dst_input_seqs = [x['dst_input_sequence'] for x in batch]
        # dst_ds = [self.ds.index(x['ds']) for x in batch]
        dst_ds = [x['ds'] for x in batch]

        dst_input_dict = self.tokenizer(dst_input_seqs,
                                        max_length=1024,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')
        dst_input_ids = dst_input_dict['input_ids']
        dst_input_mask = dst_input_dict['attention_mask']

        input_batch = {}
        if 'dst_target_sequence' in batch[0]:
            # training mode
            dst_target_seqs = [x['dst_target_sequence'] for x in batch]
            dst_target_dict = self.tokenizer(dst_target_seqs,
                                             max_length=1024,
                                             padding=True,
                                             truncation=True,
                                             return_tensors='pt')
            dst_target_ids = dst_target_dict['input_ids']

            input_batch = {
                "target_ids_womask": dst_target_ids,
                'target_seqs_womask': dst_target_seqs,
                'decoder_input_ids_womask': t5_shift_tokens_right(dst_target_ids),
            }

        if batch[0]['dst_generation_decoder_inputs'] != '':
            dst_decoder_input_seqs = [x['dst_generation_decoder_inputs'] for x in batch]
            dst_decoder_input_dict = self.tokenizer(dst_decoder_input_seqs,
                                                     max_length=1024,
                                                     padding=True,
                                                     truncation=True,
                                                     return_tensors='pt')
            dst_decoder_input_ids = dst_decoder_input_dict['input_ids']
            dst_decoder_input_ids = t5_shift_tokens_right(dst_decoder_input_ids)
            input_batch.update({
                "decoder_inputs_womask": dst_decoder_input_ids,
            })

        input_batch.update({
            "input_ids_womask": dst_input_ids,
            "attention_mask_womask": dst_input_mask,
            'input_seqs_womask': dst_input_seqs,
            'ds': dst_ds,
        })
        return input_batch



class GPT2DomainPromptDSTDatasetTodcl(Dataset):
    # one prompt for each domain
    def __init__(self,
                 tokenizer,
                 data_dir,  # must contain 'train_dials.json'
                 type_path,
                 small_sample_run,
                 prompt_style,
                 test_file_path=None,  # only for test, the test file with predicted cl_domain
                 ):
        # load data
        self.random_generator = random.Random(42)
        if test_file_path is None:
            self.src_file = os.path.join(data_dir, '{}_dials.json'.format(type_path))
        else:
            assert type_path == 'test'
            self.src_file = test_file_path
        print('loading dataset file from {}'.format(self.src_file))
        self.type_path = type_path
        self.dialogs = json.load(open(self.src_file))
        train_dialogs = json.load(open(os.path.join(data_dir, 'train_dials.json')))

        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.small_sample_run = small_sample_run

        # prepare prompt template
        # prompt_style: format of prompt for each slot
        self.prompt_style = prompt_style
        self.num_prompt_token_per_slot = sum([int(_) for _ in self.prompt_style if _.isdigit()])
        prompt_list = []
        assert 'E' not in self.prompt_style
        assert '|' not in self.prompt_style

        for i, ch in enumerate(self.prompt_style):
            if ch.isdigit():
                prompt_list.extend(['<prompt_{}>' for _ in range(int(ch))])
            elif ch == 'D':
                prompt_list.append('<domain>')
            elif ch == 'S':
                prompt_list.append('<slot>')

        prompt = ''.join(prompt_list)
        if '<domain><slot>' in prompt:
            prompt = prompt.replace('<domain><slot>', '<domain> <slot>')
        self.prompt_template = prompt

        # prepare questions and ds and cl_domains
        self.ds = []
        self.cl_domain_to_ds = {}
        self.questions = {}
        for cl_domain_id, dials in train_dialogs.items():
            cl_domain = tuple(eval(cl_domain_id))
            if cl_domain not in self.cl_domain_to_ds:
                self.cl_domain_to_ds[cl_domain] = set()
            for dial in dials:
                for ds in dial['state']:
                    self.cl_domain_to_ds[cl_domain].add(ds)
                    if ds not in self.ds:
                        self.ds.append(ds)
        self.ds = list(sorted(self.ds))
        for ds in self.ds:
            self.make_new_prompt(ds)
        self.cl_domains = list(sorted(list(self.cl_domain_to_ds.keys())))
        self.cl_domain_to_ds = {k: list(sorted(list(v))) for k, v in self.cl_domain_to_ds.items()}

        # prepare examples
        self.cl_domain2examples = {k: [] for k in self.cl_domains}
        for cl_domain_id, dials in self.dialogs.items():
            cl_domain = tuple(eval(cl_domain_id))
            for dial in dials:
                self.cl_domain2examples[cl_domain].append(self.convert_dial_to_example(dial, cl_domain))
        if small_sample_run:
            self.cl_domain2examples = {k: v[:32] for k, v in self.cl_domain2examples.items()}

        self.cl_domain2numexamples = {k: len(self.cl_domain2examples[k]) for k in self.cl_domains}
        self.dataset_len = sum(self.cl_domain2numexamples.values())

    def make_new_prompt(self, ds):
        prompt_id_incr = len(self.questions) * self.num_prompt_token_per_slot
        prompt_ids = [prompt_id_incr + i for i in range(self.num_prompt_token_per_slot)]
        prompt = self.prompt_template.format(*(prompt_ids))
        self.questions[ds] = prompt

    def reset(self, cur_epoch=-1):
        return

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        example = None
        for cl_domain in self.cl_domains:
            domain_num = self.cl_domain2numexamples[cl_domain]
            if index < domain_num:
                example = self.cl_domain2examples[cl_domain][index]
                break
            index -= domain_num
        assert example is not None
        return example


    def make_sortish_sampler(self, batch_size, **kwargs):
        return SubsetRandomSampler(
            list(BatchSampler(SequentialSampler(list(range(self.dataset_len))), batch_size, drop_last=False)))


    def prepare_for_generation(self, dial, cl_domain):
        return self.convert_dial_to_example(dial, cl_domain)


    def convert_dial_to_example(self, dial, cl_domain):
        dss = self.cl_domain_to_ds[cl_domain]
        domain_enc_prompt = ''
        target_seq = ''
        ds_extra_id_num = 0
        dss_list = []
        for ds in dss:
            enc_p = self.questions[ds]
            domain_enc_prompt += enc_p
            value = dial['state'].get(ds, 'none')
            target_seq += ' {},'.format(value)
            ds_extra_id_num += 1
            dss_list.append(ds)
        return {
            'dst_input_sequence': dial['history'] + domain_enc_prompt,
            'dst_target_sequence': target_seq + self.tokenizer.eos_token,
            'ds': ' '.join(dss_list),
            'dial_id': dial['dataset'] + '_' + dial['dial_id'] + '_' + str(dial['turn_id']),
            'dst_generation_decoder_inputs': dial['history'] + domain_enc_prompt,
        }


    def collate_fn(self, batch):
        gpt_input_seqs = [x['dst_input_sequence'] + x['dst_target_sequence'] for x in batch]
        domains = [x['ds'] for x in batch]

        gpt_input_dict = self.tokenizer(gpt_input_seqs,
                                        max_length=1024,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')
        gpt_input_ids = gpt_input_dict['input_ids']

        target_seqs = [x['dst_target_sequence'] for x in batch]
        target_dict = self.tokenizer(target_seqs,
                                     max_length=1024,
                                     padding=True,
                                     truncation=True,
                                     return_tensors='pt')
        target_ids = target_dict['input_ids']
        target_id_len = target_ids.shape[1]
        noloss_mask = torch.ones_like(gpt_input_ids).bool()
        noloss_mask[:, -target_id_len:] = target_ids != gpt_input_ids[:, -target_id_len:]
        gpt_target_ids = gpt_input_ids.clone()
        gpt_target_ids[noloss_mask] = self.tokenizer.pad_token_id

        gen_input_seqs = [x['dst_input_sequence'] for x in batch]
        gen_dict = self.tokenizer(gen_input_seqs,
                                  max_length=1024,
                                  padding=True,
                                  truncation=True,
                                  return_tensors='pt')
        gen_input_ids = gen_dict['input_ids']

        input_batch = {
            'target_seqs': target_seqs,
            'gpt_target_ids': gpt_target_ids,
            'gpt_input_ids': gpt_input_ids,
            'gpt_input_seqs': gpt_input_seqs,
            'gen_input_ids': gen_input_ids,
            'gen_input_seqs': gen_input_seqs,
            'ds': domains,
        }

        return input_batch



class DialoGPTDomainPromptDSTDatasetTodcl(GPT2DomainPromptDSTDatasetTodcl):
    def convert_dial_to_example(self, dial, cl_domain, test_mode=False):
        # test_mode: calculate loss on full dialog
        ret = []
        domain_enc_prompt = self.questions[cl_domain]
        dialog_words = dial['history'].split()
        dialog_utts = []
        cur_utt_list = []
        for i, word in enumerate(dialog_words):
            if word in ['USER:', 'SYSTEM:']:
                cur_utt_list.append(self.tokenizer.eos_token)
                dialog_utts.append(' '.join(cur_utt_list))
                cur_utt_list = []
            else:
                cur_utt_list.append(word)
        cur_utt_list.append(self.tokenizer.eos_token)
        dialog_utts.append(' '.join(cur_utt_list))
        dialog_utts = dialog_utts[1:]

        context_seq, target_seq = '', ' '.join(dialog_utts)
        target_seq = ' ' + target_seq
        ret.append({
            'dst_input_sequence': domain_enc_prompt + context_seq,
            'dst_target_sequence': target_seq,
            'ds': cl_domain,
            'dst_generation_decoder_inputs': domain_enc_prompt + context_seq,
        })
        return ret
