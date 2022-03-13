import re
from typing import List

from prompt_files.p_tuning.todcl.todcl_domain_dataset import *
from torch.utils.data import RandomSampler, Dataset
import torch
import numpy as np
from prompt_files.transformer_utils import logging
logger = logging.get_logger(__name__)
# logger.setLevel(logging.INFO)


class RangeIndexSampler(Sampler):
    def __init__(self, start, end):
        self.indexes = list(range(start, end))
        np.random.shuffle(self.indexes)

    def __iter__(self):
        yield from self.indexes

    def __len__(self):
        return len(self.indexes)


class RangeIndexSeqSampler(Sampler):
    def __init__(self, start, end, max_sample=None):
        if max_sample is None:
            self.indexes = list(range(start, end))
        else:
            self.indexes = list(range(start, end, max(1, (end-start)//max_sample)))

    def __iter__(self):
        yield from self.indexes

    def __len__(self):
        return len(self.indexes)


class MemT5PromptDSTDataset(Dataset):
    """
    A unified dataset which support:
    - Multi-task:
        set training_prompt_name='meta' and resample for training,
        set prepare_for_generation(domain='meta') for testing.
        make_full_data_sampler
    - RandInit:
        directly use
        make_domain_sampler
    - CLInit & SelectInit:
        set training_prompt_name='meta_prompt'/'prompt', resample dataset for training.
        copy prompt, resample every epoch if perm_desc
        make_domain_sampler
    - Memory replay:
        set training_prompt_name='meta_prompt'/'prompt', resample dataset for training.
        prepare forward_domains, sample replay_memory
        make_domain_sampler
    - Backward transfer:
        set training_prompt_name='meta_prompt'/'prompt', resample dataset for training.
        prepare backward_domain, sample replay_memory
        make_domain_sampler for current domain data + make_domain_sampler(domain='memory') for backward domain memory
    """
    def __init__(self,
                 tokenizer,
                 type_path,
                 dialogs,
                 domain2slot,
                 num_domain_prompt=100,
                 small_sample_run=False,
                 permute_desc=False,
                 multitask=False
                 ):
        self.permute_desc = permute_desc
        self.dialogs = dialogs
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.small_sample_run = small_sample_run
        self.type_path = type_path
        self.multitask = multitask

        # prepare prompt template
        self.num_domain_prompt = num_domain_prompt

        self.slot2description = {}
        schema = json.load(
            open('data/dstc8-schema-guided-dialogue/train/schema.json')) + json.load(
            open('data/dstc8-schema-guided-dialogue/dev/schema.json')) + json.load(
            open('data/dstc8-schema-guided-dialogue/test/schema.json'))
        for service in schema:
            service_name = service['service_name'].lower()
            slots = service['slots']
            for slot in slots:
                slot_name = 'sgd_{}-{}'.format(service_name, slot['name'])
                desc = slot['description'] + ': <extra_id_0> . '
                self.slot2description[slot_name] = desc

        self.domain2slot = domain2slot
        self.domains = list(sorted(list(self.domain2slot.keys())))

        self.replay_memory = {d: [] for d in self.domains}  # dials
        # prepare examples
        if type_path in ['train', 'val']:
            self.domain2samples = {k: [] for k in self.domains}
            self.domain2samples['memory'] = []
            for domain, dials in self.dialogs.items():
                assert domain in self.domains
                soft_prompt = self.domain2soft_prompt(domain)
                for dial in dials:
                    self.domain2samples[domain].append(self.convert_dial_to_example(dial, domain, soft_prompt))
            if small_sample_run:
                self.domain2samples = {k: v[:10] for k, v in self.domain2samples.items()}

            self.domain2numsamples = {k: len(self.domain2samples[k]) for k in self.domain2samples.keys()}
            self.dataset_len = sum(self.domain2numsamples.values())
            print('domain2numsamples', self.domain2numsamples)
            print('total', self.dataset_len)

        self.aug_metatrain_data = {d: [] for d in self.domains}  # dials
        self.random_generator = random.Random(52)

    def convert_dial_to_example(self, dial, domain, soft_prompt, do_augment=False, extra_aug_slots=[]):
        # extra_aug_domains: use slots from these domains to query dial, slot values should be 'none'
        slots = self.domain2slot[domain]
        domain_enc_prompt = ' </s> '
        target_seq = ''
        extra_id_num = 0
        if do_augment:
            assert self.type_path == 'train'
            # num_slots = self.random_generator.randint(len(slots)//2, len(slots))
            num_slots = self.random_generator.randint(1, len(slots))
            slots = self.random_generator.sample(slots, num_slots)
            if len(extra_aug_slots) > 0:
                # num_extra_slots = self.random_generator.randint(1, min(len(slots)//2, len(extra_aug_slots)))
                num_extra_slots = self.random_generator.randint(1, min(len(slots), len(extra_aug_slots)))
                extra_slots = self.random_generator.sample(extra_aug_slots, num_extra_slots)
                slots += extra_slots
        if self.type_path == 'train' and do_augment:
            slots = np.random.permutation(slots)

        for i, slot in enumerate(slots):
            enc_p = self.slot2description[slot]
            domain_enc_prompt += enc_p.replace('<extra_id_0>', '<extra_id_{}>'.format(extra_id_num))
            value = dial['state'].get(slot, 'none')
            target_seq += '<extra_id_{}>{}'.format(extra_id_num, value)
            extra_id_num += 1
        input_dict = {
            'dst_input_sequence': (dial['history'] + domain_enc_prompt + soft_prompt).lower(),
            'dst_target_sequence': target_seq.lower(),
            'ds': ' '.join(slots),
            'dial_id': dial['dataset'] + '_' + dial['dial_id'] + '_' + str(dial['turn_id']),
            'dst_generation_decoder_inputs': ''
        }
        return input_dict

    def resample_dataset(self, cur_domain, training_prompt_name, forward_domains:List=None, backward_domain=None,
                         do_augment=False, fake_examples=[]):
        logger.info('{} data: resample {} domain for {}. fw_domains: {}, backward_domain: {}, aug: {}'.format(
            self.type_path, cur_domain, training_prompt_name, forward_domains, backward_domain, do_augment))
        all_extra_slots = []
        if forward_domains is not None:
            for d in forward_domains:
                all_extra_slots += self.domain2slot[d]
        if backward_domain is not None:
            all_extra_slots += self.domain2slot[backward_domain]
        for domain, dials in self.dialogs.items():
            if domain != cur_domain:
                continue
            self.domain2samples[cur_domain] = []
            if training_prompt_name == 'meta_prompt':
                soft_prompt = self.domain2soft_prompt('meta')
            else:
                soft_prompt = self.domain2soft_prompt(cur_domain)
            for dial in dials:
                self.domain2samples[cur_domain].append(
                    self.convert_dial_to_example(dial, cur_domain, soft_prompt, do_augment, all_extra_slots))
        if self.type_path == 'train' and training_prompt_name == 'meta_prompt':
            # at training, use memory from forward_domains
            soft_prompt = self.domain2soft_prompt('meta')
            if forward_domains:
                logger.info('{} domain samples: {}'.format(cur_domain, len(self.domain2samples[cur_domain])))
                for domain in forward_domains:
                    dials = self.replay_memory[domain]
                    extra_slots_for_mem = [_ for _ in self.domain2slot[cur_domain]]
                    for d in forward_domains:
                        if d != domain:
                            extra_slots_for_mem += self.domain2slot[d]
                    for dial in dials:
                        self.domain2samples[cur_domain].append(
                            self.convert_dial_to_example(dial, domain, soft_prompt, do_augment, extra_slots_for_mem))
                logger.info('{} domain samples+memory: {}'.format(cur_domain, len(self.domain2samples[cur_domain])))

                if len(fake_examples) > 0:
                    self.domain2samples[cur_domain] += fake_examples
                    logger.info('{} domain samples+memory+generated_exampples: {}'.format(cur_domain, len(self.domain2samples[cur_domain])))

            if backward_domain:
                self.domain2samples['memory'] = []
                dials = self.replay_memory[backward_domain]
                for dial in dials:
                    self.domain2samples['memory'].append(self.convert_dial_to_example(dial, backward_domain, soft_prompt))
                logger.info('backward {} domain memory: {}'.format(backward_domain, len(self.domain2samples['memory'])))
                if len(fake_examples) > 0:
                    self.domain2samples[cur_domain] += fake_examples
                    logger.info(
                        'backward {} domain generate fake examples: {}'.format(backward_domain,
                                                                               len(self.domain2samples[cur_domain])))
        if self.small_sample_run:
            self.domain2samples = {k: v[:10] for k, v in self.domain2samples.items()}
        self.domain2numsamples = {k: len(self.domain2samples[k]) for k in self.domain2samples.keys()}
        self.dataset_len = sum(self.domain2numsamples.values())

    def collate_fn(self, batch):
        dst_input_seqs = [x['dst_input_sequence'] for x in batch]
        dst_ds = [x['ds'] for x in batch]

        dst_input_dict = self.tokenizer(dst_input_seqs,
                                        max_length=1024,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')
        dst_input_ids = dst_input_dict['input_ids']
        dst_input_mask = dst_input_dict['attention_mask']

        input_batch = {
            "input_ids_womask": dst_input_ids,
            "attention_mask_womask": dst_input_mask,
            'input_seqs_womask': dst_input_seqs,
            'ds': dst_ds,
        }
        if 'dst_target_sequence' in batch[0]:
            # training mode
            dst_target_seqs = [x['dst_target_sequence'] for x in batch]
            dst_target_dict = self.tokenizer(dst_target_seqs,
                                             max_length=1024,
                                             padding=True,
                                             truncation=True,
                                             return_tensors='pt')
            dst_target_ids = dst_target_dict['input_ids']

            input_batch.update({
                'target_ids_womask': dst_target_ids,
                'target_seqs_womask': dst_target_seqs,
                'decoder_input_ids_womask': t5_shift_tokens_right(dst_target_ids),
            })

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

        return input_batch

    def get_prompt_init_dict(self, from_cl_domain=None, to_cl_domain=None):
        assert from_cl_domain is not None or to_cl_domain is not None
        if from_cl_domain is None:
            from_prompt_idxs = list(range(self.num_domain_prompt))
        else:
            from_idx = self.domains.index(from_cl_domain)
            from_prompt_idxs = list(range(from_idx * self.num_domain_prompt, (from_idx + 1) * self.num_domain_prompt))
        if to_cl_domain is None:
            to_prompt_idxs = list(range(self.num_domain_prompt))
        else:
            to_idx = self.domains.index(to_cl_domain)
            to_prompt_idxs = list(range(to_idx * self.num_domain_prompt, (to_idx + 1) * self.num_domain_prompt))
        ret_dict = {b: a for a, b in zip(from_prompt_idxs, to_prompt_idxs)}
        return ret_dict

    def domain2soft_prompt(self, domain):
        if domain == 'meta' or self.multitask:
            soft_prompt = ''.join(['<meta_prompt_{}>'.format(i) for i in range(self.num_domain_prompt)])
        else:
            domain_idx = self.domains.index(domain)
            soft_prompt = ''.join(['<prompt_{}>'.format(i + domain_idx * self.num_domain_prompt) for i in
                                   range(self.num_domain_prompt)])
        return soft_prompt

    def prepare_for_generation(self, dial, domain):
        return self.convert_dial_to_example(dial, domain, self.domain2soft_prompt(domain))

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        example = None
        for domain in ['memory'] + self.domains:
            domain_num = self.domain2numsamples[domain]
            if index < domain_num:
                example = self.domain2samples[domain][index]
                break
            index -= domain_num
        assert example is not None
        return example

    def make_domain_sampler(self, batch_size, target_domain):
        # target domain could be 'memory'
        assert self.type_path in ['train', 'val']
        start_sample_idx = 0
        end_sample_idx = -1
        for domain in ['memory'] + self.domains:
            domain_num = self.domain2numsamples[domain]
            if target_domain == domain:
                end_sample_idx = start_sample_idx + domain_num
                break
            else:
                start_sample_idx += domain_num
        assert end_sample_idx > -1
        return BatchSampler(RangeIndexSampler(start_sample_idx, end_sample_idx), batch_size, drop_last=False)

    def make_full_data_sampler(self, batch_size):
        # For multi-task learning
        return BatchSampler(RangeIndexSampler(0, self.dataset_len), batch_size, drop_last=False)

    def make_sequential_domain_sampler(self, batch_size, target_domain):
        # target domain could be 'memory'
        assert self.type_path in ['train', 'val']
        start_sample_idx = 0
        end_sample_idx = -1
        for domain in ['memory'] + self.domains:
            domain_num = self.domain2numsamples[domain]
            if target_domain == domain:
                end_sample_idx = start_sample_idx + domain_num
                break
            else:
                start_sample_idx += domain_num
        assert end_sample_idx > -1
        return BatchSampler(RangeIndexSeqSampler(start_sample_idx, end_sample_idx), batch_size, drop_last=False)


class T5PromptGenFullStateDataset(MemT5PromptDSTDataset):
    def resample_dataset(self, cur_domain, training_prompt_name, forward_domains:List=None, backward_domain=None,
                         do_augment=False, fake_examples=[]):
        logger.info('resample {} domain for {}. fw_domains: {}, backward_domain: {}'.format(
            cur_domain, training_prompt_name, forward_domains, backward_domain))

        for domain, dials in self.dialogs.items():
            if domain != cur_domain:
                continue
            self.domain2samples[cur_domain] = []
            if training_prompt_name == 'meta_prompt':
                soft_prompt = self.domain2soft_prompt('meta')
            else:
                soft_prompt = self.domain2soft_prompt(cur_domain)
            for dial in dials:
                self.domain2samples[cur_domain].append(
                    self.convert_dial_to_example(dial, cur_domain, soft_prompt))

        if self.small_sample_run:
            self.domain2samples = {k: v[:10] for k, v in self.domain2samples.items()}
        self.domain2numsamples = {k: len(self.domain2samples[k]) for k in self.domain2samples.keys()}
        self.dataset_len = sum(self.domain2numsamples.values())


    def convert_dial_to_example(self, dial, domain, soft_prompt, do_augment=False, extra_aug_slots=[]):
        input_dict = {
            'dst_input_sequence': (dial['history'] + soft_prompt).lower(),
            'dst_target_sequence': dial['reply'].replace('</s>', '').lower(),
            'dial_id': dial['dataset'] + '_' + dial['dial_id'] + '_' + str(dial['turn_id']),
        }
        return input_dict


    def collate_fn(self, batch):
        dst_input_seqs = [x['dst_input_sequence'] for x in batch]

        dst_input_dict = self.tokenizer(dst_input_seqs,
                                        max_length=1024,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')
        dst_input_ids = dst_input_dict['input_ids']
        dst_input_mask = dst_input_dict['attention_mask']

        input_batch = {
            "input_ids_womask": dst_input_ids,
            "attention_mask_womask": dst_input_mask,
            'input_seqs_womask': dst_input_seqs,
        }
        if 'dst_target_sequence' in batch[0]:
            # training mode
            dst_target_seqs = [x['dst_target_sequence'] for x in batch]
            dst_target_dict = self.tokenizer(dst_target_seqs,
                                             max_length=1024,
                                             padding=True,
                                             truncation=True,
                                             return_tensors='pt')
            dst_target_ids = dst_target_dict['input_ids']

            input_batch.update({
                'target_ids_womask': dst_target_ids,
                'target_seqs_womask': dst_target_seqs,
                'decoder_input_ids_womask': t5_shift_tokens_right(dst_target_ids),
            })
        return input_batch
