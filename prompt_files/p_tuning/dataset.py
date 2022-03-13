# import copy
# from pathlib import Path
# import json
# from typing import List, Dict
# from collections import Counter
#
# import numpy as np
# import random
#
# import torch
# from torch.utils.data import SubsetRandomSampler, BatchSampler, RandomSampler, SequentialSampler
#
# from data_utils.multiwoz.config import SLOTIDX2DS, DS2SLOTIDX, ALL_SLOTS
# from torch.utils.data import Dataset
#
#
# def bart_shift_tokens_right(input_ids: torch.Tensor):
#     """
#     Shift input ids one token to the right.
#     """
#     decoder_start_token_id = 2
#     pad_token_id = 1
#     shifted_input_ids = input_ids.new_zeros(input_ids.shape)
#     shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
#     shifted_input_ids[:, 0] = decoder_start_token_id
#
#     assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
#     # replace possible -100 values in labels by `pad_token_id`
#     shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
#
#     return shifted_input_ids
#
#
# def t5_shift_tokens_right(input_ids):
#     decoder_start_token_id = 0
#     pad_token_id = 0
#
#     assert (
#             decoder_start_token_id is not None
#     ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"
#
#     # shift inputs to the right
#     shifted_input_ids = input_ids.new_zeros(input_ids.shape)
#     shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
#     shifted_input_ids[..., 0] = decoder_start_token_id
#
#     assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
#     # replace possible -100 values in labels by `pad_token_id`
#     shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
#
#     assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"
#
#     return shifted_input_ids
#
#
# class PromptDSTDataset(Dataset):
#     def __init__(self,
#                  tokenizer,
#                  data_dir,
#                  type_path,
#                  small_sample_run,
#                  few_shot_rate=None,
#                  domains=ALL_SLOTS.keys(),
#                  prompt_style=None,
#                  value_none_sample_ratio=1.0,
#                  **unused,
#                  ):
#         # load files
#         if few_shot_rate is None:
#             self.src_file = Path(data_dir).joinpath(type_path + "_dials.json")
#         else:
#             if type_path == 'train':
#                 self.src_file = Path(data_dir).joinpath('fewshot_train_dataset', '{}_dials_{}.json'.format(type_path, few_shot_rate))
#             else:
#                 self.src_file = Path(data_dir).joinpath(type_path + "_dials.json")
#         self.question_file = Path(data_dir).joinpath('slot_questions.json')
#         self.type_path = type_path
#         print('loading dataset file from {}'.format(self.src_file))
#         print('loading questions file from {}'.format(self.question_file))
#         self.domains = domains
#         self.random_generator = random.Random(42)
#         self.dialogs = json.load(open(self.src_file))
#         self.dialog_ids = list(self.dialogs.keys())
#         self.questions = json.load(open(self.question_file))
#
#         if small_sample_run:
#             self.dialog_ids = self.dialog_ids[:50]
#
#         self.tokenizer = tokenizer
#         self.pad_token_id = self.tokenizer.pad_token_id
#         self.small_sample_run = small_sample_run
#
#         # input formats
#         self.prompt_style = prompt_style
#         self.num_history = 2
#         self.num_context = -1
#         self.turn_selection_rate = 0.5
#         self.num_examples = -1
#         self.value_none_sample_ratio = value_none_sample_ratio
#
#         # prompt initialization
#         if self.type_path == 'test':
#             self.turn_selection_rate = 1.0
#
#         self.ds = SLOTIDX2DS  # only include data from these domain-slots
#         self.ds = [ds for ds in SLOTIDX2DS if ds.split('-')[0] in self.domains]
#
#
#     def filter_none_state(self, _state):
#         # remove empty values
#         ret_dict = {}
#         for _service, _service_state in _state.items():
#             if _service not in ret_dict:
#                 ret_dict[_service] = {}
#             for _slot, _value in _service_state.items():
#                 if _slot in ['parking', 'internet'] and _value == 'free':
#                     ret_dict[_service][_slot] = 'yes'
#                     continue
#
#                 if _value in ['not mentioned', '', 'none']:
#                     continue
#                 else:
#                     ret_dict[_service][_slot] = _value
#         ret_dict = {k: ret_dict[k] for k in ret_dict if len(ret_dict[k]) != 0}
#         return ret_dict
#
#
#     def get_state_update(self, prev_state, state):
#         # WARN! PREV_STATE MUST BE STATE[START_TURN-1] !!! or None
#         state_update = {}
#         if prev_state is None:
#             state_update = copy.deepcopy(state)
#         else:
#             for domain in state:
#                 if domain not in prev_state:
#                     state_update.update({domain: copy.deepcopy(state[domain])})
#                     continue
#                 else:
#                     for slot in state[domain]:
#                         if slot not in prev_state[domain] or state[domain][slot] != prev_state[domain][slot]:
#                             if domain not in state_update:
#                                 state_update[domain] = {}
#                             state_update[domain].update({slot: state[domain][slot]})
#         state_update = self.filter_none_state(state_update)
#         return state_update
#
#
#     def get_update_turns(self, belief_states, start_turn, sample_turn):
#         prev_state = {} if start_turn == 0 else belief_states[start_turn - 1]
#         state_update = self.get_state_update(prev_state, belief_states[sample_turn])
#         diff_states = [self.get_state_update(s, belief_states[sample_turn]) for s in belief_states]
#         diff_states = [belief_states[sample_turn]] + diff_states
#         for domain in state_update:
#             for slot in state_update[domain]:
#                 value = state_update[domain][slot]
#                 for turn_id in range(sample_turn, max(-1, start_turn - 1), -1):
#                     if domain in diff_states[turn_id] and slot in diff_states[turn_id][domain] and \
#                             diff_states[turn_id][domain][slot] == value:
#                         state_update[domain][slot] = (state_update[domain][slot], turn_id + 1)
#                         break
#         for domain in state_update:
#             for slot in state_update[domain]:
#                 value = state_update[domain][slot]
#                 if not isinstance(value, tuple):
#                     state_update[domain][slot] = (state_update[domain][slot], start_turn)
#         return state_update
#
#
#
#     def _format_all_slots_dict(self, default_slot_value):
#         ret = {}
#         for ds in self.ds:
#             ret[ds] = copy.deepcopy(default_slot_value)
#         return ret
#
#
#     def read_dialog(self, did):
#         def _add_end_punct(utt):
#             utt = utt.strip()
#             from string import punctuation
#             if utt[-1] not in punctuation:
#                 utt += '.'
#             return utt
#
#         dialog = self.dialogs[did]
#         sys_utts: List = [_add_end_punct(_) for _ in dialog['lexical_sys']]
#         usr_utts: List = [_add_end_punct(_) for _ in dialog['lexical_usr']]
#         belief_states: List = dialog['bstate']
#         service_names: List = dialog['active_domains']
#         return sys_utts, usr_utts, belief_states, service_names
#
#
#     def build_dialog_sequence_with_history(self, user_utts, sys_utts, history_start_turn, context_start_turn, end_turn):
#         # start_turn and end_turn: inclusive
#         assert history_start_turn <= context_start_turn and context_start_turn <= end_turn
#         dialog_list = []
#
#         dialog_list.append('history:')
#         if history_start_turn < context_start_turn:
#             if history_start_turn > 0:
#                 dialog_list.append('system:')
#                 dialog_list.append(sys_utts[history_start_turn - 1])
#             for _turn in range(history_start_turn, context_start_turn - 1):
#                 dialog_list.append('user:')
#                 dialog_list.append(user_utts[_turn])
#                 dialog_list.append('system:')
#                 dialog_list.append(sys_utts[_turn])
#             dialog_list.append('user:')
#             dialog_list.append(user_utts[context_start_turn - 1])
#
#         dialog_list.append('context:')
#         if context_start_turn > 0:
#             dialog_list.append('system:')
#             dialog_list.append(sys_utts[context_start_turn - 1])
#         for _turn_id in range(context_start_turn, end_turn):
#             dialog_list.append('user:')
#             dialog_list.append(user_utts[_turn_id])
#             dialog_list.append('system:')
#             dialog_list.append(sys_utts[_turn_id])
#         dialog_list.append('user:')
#         dialog_list.append(user_utts[end_turn])
#         dialog_list = [_ for _ in dialog_list if _ != '']
#         return ' '.join(dialog_list)
#
#
#     def process_dialog_for_generation(self, did, end_turn, n_context):
#         if n_context < 0:
#             context_start_turn_id = 0
#             history_start_turn_id = 0
#         else:
#             context_start_turn_id = max(end_turn - n_context + 1, 0)
#             history_start_turn_id = max(context_start_turn_id-self.num_history+1, 0)
#         sys_utts, user_utts, _, _ = self.read_dialog(did)
#         dialog_input = self.build_dialog_sequence_with_history(user_utts, sys_utts, history_start_turn_id,
#                                                                context_start_turn_id, end_turn)
#         dialog_inputs = []
#         for ds in self.ds:
#             domain, slot = ds.split('-')
#             dialog_inputs.append(dialog_input + ' question: {} ?'.format(self.questions[domain][slot]))
#         return dialog_inputs
#
#
#     def build_dst_input_target_sample(self, example):
#         did, sampled_history_start_turn_id, sampled_context_start_turn_id, end_turn, value = example
#         sys_utts, user_utts, _, _ = self.read_dialog(did)
#         dialog_input = self.build_dialog_sequence_with_history(user_utts, sys_utts, sampled_history_start_turn_id,
#                                                                sampled_context_start_turn_id, end_turn)
#         dialog_target = value.split()
#         dialog_target = ' '.join(dialog_target)
#         return dialog_input, dialog_target
#
#
#     def __len__(self):
#         return self.num_examples
#
#     def __getitem__(self, index) -> Dict:
#         example = None
#         example_ds = None
#         for ds_idx, ds in enumerate(self.ds):
#             ds_num = self.ds2numexamples[ds_idx]
#             if index < ds_num:
#                 example = self.sampled_slot2dialogturns[ds][index]
#                 example_ds = ds
#                 break
#             index -= ds_num
#
#         dst_input_seq, dst_target_seq = self.build_dst_input_target_sample(example)
#         domain, slot = example_ds.split('-')
#         dst_input_seq += ' question: {} ?'.format(self.questions[domain][slot])
#
#         return {
#             'dst_input_sequence': dst_input_seq,
#             'dst_target_sequence': dst_target_seq,
#             'ds': example_ds,
#         }
#
#
#     def make_sortish_sampler(self, batch_size, **kwargs):
#         return SubsetRandomSampler(list(BatchSampler(RandomSampler(list(range(self.num_examples))), batch_size, drop_last=False)))
#
#
#     def dropadd_sample_examples(self, sampled_ds_examples, expected_num_examples):
#         num_sampled_ds_examples = sum([len(_) for _ in sampled_ds_examples.values()])
#         n_total_drop_or_add = abs(num_sampled_ds_examples - expected_num_examples)
#         ds2cut_nums = np.random.choice(list(self.sampled_slot2dialogturns.keys()), n_total_drop_or_add)
#         ds2cut_nums = Counter(ds2cut_nums)
#         for ds, n in ds2cut_nums.items():
#             if num_sampled_ds_examples > expected_num_examples:
#                 sampled_ds_examples[ds] = sampled_ds_examples[ds][:-n]
#             else:
#                 sampled_ds_examples[ds] += sampled_ds_examples[ds][-n:]
#
#
#     def reset(self, cur_epoch=-1):
#         raise NotImplementedError
#
#
# class T5PromptDSTDataset(PromptDSTDataset):
#     def __init__(self, **args):
#         super().__init__(**args)
#         prompt_token_mapping = args['prompt_token_mapping']
#         num_prompt_token_per_slot = sum([int(_) for _ in self.prompt_style if _.isdigit()])
#         prompt_list = []
#         next_prompt_token_id = 0
#         assert 'E' in self.prompt_style
#         if '|' not in self.prompt_style:
#             self.prompt_style.append('|')
#
#         for i, ch in enumerate(self.prompt_style):
#             if ch.isdigit():
#                 prompt_list.extend(['<prompt_{}>' for _ in range(int(ch))])
#                 next_prompt_token_id += int(ch)
#             elif ch == 'D':
#                 prompt_list.append('<domain>')
#             elif ch == 'S':
#                 prompt_list.append('<slot>')
#             elif ch == 'E':
#                 prompt_list.append('<extra_id_0>')
#             elif ch == '|':
#                 prompt_list.append('|')
#
#         prompt = ''.join(prompt_list)
#
#         if '<domain><slot>' in prompt:
#             prompt = prompt.replace('<domain><slot>', '<domain> <slot>')
#
#         for i, ds in enumerate(SLOTIDX2DS):
#             domain, slot = ds.split('-')
#             prompt_id_incr = i * num_prompt_token_per_slot
#             prompt_ids = [prompt_id_incr+i for i in range(num_prompt_token_per_slot)]
#             prompt_ids = [prompt_token_mapping.get(_, _) for _ in prompt_ids]
#             slot_prompt = prompt.replace('<domain>', domain)
#             slot_prompt = slot_prompt.replace('<slot>', slot)
#             enc_p, dec_p = slot_prompt.format(*prompt_ids).split('|')
#             self.questions[domain][slot] = [enc_p, dec_p]
#
#     def build_dst_input_target_sample(self, example):
#         did, sampled_history_start_turn_id, sampled_context_start_turn_id, end_turn, value = example
#         sys_utts, user_utts, _, _ = self.read_dialog(did)
#         dialog_input = self.build_dialog_sequence_with_history(user_utts, sys_utts, sampled_history_start_turn_id,
#                                                                sampled_context_start_turn_id, end_turn)
#         if value == 'dontcare':
#             value = 'don\'t care'
#         dialog_target = value.split()
#         dialog_target = ' '.join(['<extra_id_0>'] + dialog_target)
#         return dialog_input, dialog_target
#
#
#     def __getitem__(self, index) -> Dict:
#         example = None
#         example_ds = None
#         assert index < sum(self.ds2numexamples), print('='*50, index, self.ds2numexamples, sum(self.ds2numexamples), '='*50)
#
#         for ds_idx, ds in enumerate(self.ds):
#             ds_num = self.ds2numexamples[ds_idx]
#             if index < ds_num:
#                 example = self.sampled_slot2dialogturns[ds][index]
#                 example_ds = ds
#                 break
#             index -= ds_num
#         assert example is not None, print(self.ds2numexamples, index)
#         dst_input_seq, dst_target_seq = self.build_dst_input_target_sample(example)
#         domain, slot = example_ds.split('-')
#         dst_input_seq += '{}'.format(self.questions[domain][slot][0])
#         dst_target_seq = self.questions[domain][slot][1] + dst_target_seq
#
#         return {
#             'dst_input_sequence': dst_input_seq,
#             'dst_target_sequence': dst_target_seq,
#             'ds': example_ds,
#             'dst_generation_decoder_inputs': self.questions[domain][slot][1],
#         }
#
#
#     def process_dialog_for_generation(self, did, end_turn, n_context):
#         if n_context < 0:
#             context_start_turn_id = 0
#             history_start_turn_id = 0
#         else:
#             context_start_turn_id = max(end_turn - n_context + 1, 0)
#             history_start_turn_id = max(context_start_turn_id-self.num_history, 0)
#         sys_utts, user_utts, _, _ = self.read_dialog(did)
#         dialog_input = self.build_dialog_sequence_with_history(user_utts, sys_utts, history_start_turn_id,
#                                                                context_start_turn_id, end_turn)
#         dialog_inputs = []
#         for ds in self.ds:
#             domain, slot = ds.split('-')
#             dialog_inputs.append({'domain': domain,
#                                   'slot': slot,
#                                   'ds': ds,
#                                   'dst_input_sequence': dialog_input + '{}'.format(self.questions[domain][slot][0]),
#                                   'dst_generation_decoder_inputs': self.questions[domain][slot][1],
#                                   })
#         return dialog_inputs
#
#
#     def make_sortish_sampler(self, batch_size, **kwargs):
#         if self.type_path != 'test':
#             return SubsetRandomSampler(list(BatchSampler(RandomSampler(list(range(self.num_examples))), batch_size, drop_last=False)))
#         else:
#             return BatchSampler(SequentialSampler(list(range(self.num_examples))), batch_size, drop_last=False)
#
#
#     def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
#         dst_input_seqs = [x['dst_input_sequence'] for x in batch]
#         # dst_ds = [self.ds.index(x['ds']) for x in batch]
#         dst_ds = [x['ds'] for x in batch]
#
#         dst_input_dict = self.tokenizer(dst_input_seqs,
#                                         max_length=1024,
#                                         padding=True,
#                                         truncation=True,
#                                         return_tensors='pt')
#         dst_input_ids = dst_input_dict['input_ids']
#         dst_input_mask = dst_input_dict['attention_mask']
#
#         input_batch = {}
#         if 'dst_target_sequence' in batch[0]:
#             # training mode
#             dst_target_seqs = [x['dst_target_sequence'] for x in batch]
#             dst_target_dict = self.tokenizer(dst_target_seqs,
#                                              max_length=1024,
#                                              padding=True,
#                                              truncation=True,
#                                              return_tensors='pt')
#             dst_target_ids = dst_target_dict['input_ids']
#
#             input_batch = {
#                 "target_ids_womask": dst_target_ids,
#                 'target_seqs_womask': dst_target_seqs,
#                 'decoder_input_ids_womask': t5_shift_tokens_right(dst_target_ids),
#             }
#
#         if batch[0]['dst_generation_decoder_inputs'] != '':
#             dst_decoder_input_seqs = [x['dst_generation_decoder_inputs'] for x in batch]
#             dst_decoder_input_dict = self.tokenizer(dst_decoder_input_seqs,
#                                                      max_length=1024,
#                                                      padding=True,
#                                                      truncation=True,
#                                                      return_tensors='pt')
#             dst_decoder_input_ids = dst_decoder_input_dict['input_ids']
#             dst_decoder_input_ids = t5_shift_tokens_right(dst_decoder_input_ids)
#             input_batch.update({
#                 "decoder_inputs_womask": dst_decoder_input_ids,
#             })
#
#         input_batch.update({
#             "input_ids_womask": dst_input_ids,
#             "attention_mask_womask": dst_input_mask,
#             'input_seqs_womask': dst_input_seqs,
#             'ds': dst_ds,
#         })
#         return input_batch
#
#
#
# class BartPromptDSTDataset(PromptDSTDataset):
#     def __init__(self, **args):
#         super().__init__(**args)
#         assert len(self.prompt_style) == 1 and self.prompt_style[0].isdigit()
#         prompt_style = int(self.prompt_style[0])
#         # prompt initialization
#         for i, ds in enumerate(SLOTIDX2DS):
#             domain, slot = ds.split('-')
#             if self.prompt_style[0] < 100:
#                 self.questions[domain][slot] = ' '.join(
#                     ['<prompt_{}>'.format(_) for _ in range(i * prompt_style, (i + 1) * prompt_style)])
#             elif prompt_style == 333:
#                 prompt_list = ['<prompt_{}>'.format(_) for _ in range(i * 9, (i + 1) * 9)]
#                 prompt_list = prompt_list[:3] + [domain] + prompt_list[3:6] + [slot] + prompt_list[6:]
#                 self.questions[domain][slot] = ' '.join(prompt_list)
#
#
#     def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
#         dst_input_seqs = [x['dst_input_sequence'] for x in batch]
#         dst_ds = [self.ds.index(x['ds']) for x in batch]
#
#         dst_input_dict = self.tokenizer(dst_input_seqs,
#                                         max_length=1024,
#                                         padding=True,
#                                         truncation=True,
#                                         return_tensors='pt')
#
#         dst_input_ids = dst_input_dict['input_ids']
#         dst_input_mask = dst_input_dict['attention_mask']
#
#         input_batch = {}
#         if 'dst_target_sequence' in batch[0]:
#             dst_target_seqs = [x['dst_target_sequence'] for x in batch]
#             dst_target_dict = self.tokenizer(dst_target_seqs,
#                                              max_length=1024,
#                                              padding=True,
#                                              truncation=True,
#                                              return_tensors='pt')
#             dst_target_ids = dst_target_dict['input_ids']
#             input_batch = {
#                 "target_ids_womask": dst_target_ids,
#                 'target_seqs_womask': dst_target_seqs,
#                 'decoder_input_ids_womask': bart_shift_tokens_right(dst_target_ids),
#             }
#
#         input_batch.update({
#             "input_ids_womask": dst_input_ids,
#             "attention_mask_womask": dst_input_mask,
#             'input_seqs_womask': dst_input_seqs,
#             'ds': torch.tensor(dst_ds).long(),
#         })
#         return input_batch
#
#     def process_dialog_for_generation(self, did, end_turn, n_context):
#         if n_context < 0:
#             context_start_turn_id = 0
#             history_start_turn_id = 0
#         else:
#             context_start_turn_id = max(end_turn - n_context + 1, 0)
#             history_start_turn_id = max(context_start_turn_id-self.num_history+1, 0)
#         sys_utts, user_utts, _, _ = self.read_dialog(did)
#         dialog_input = self.build_dialog_sequence_with_history(user_utts, sys_utts, history_start_turn_id,
#                                                                context_start_turn_id, end_turn)
#         dialog_inputs = []
#         for ds in self.ds:
#             domain, slot = ds.split('-')
#             dialog_inputs.append({'domain': domain,
#                                   'slot': slot,
#                                   'ds': ds,
#                                   'dst_input_sequence': dialog_input + ' question: {} ?'.format(
#                                       self.questions[domain][slot])
#                                   })
#
#         return dialog_inputs
#
#
# class GPT2PromptDSTDataset(PromptDSTDataset):
#     def __init__(self, **args):
#         super().__init__(**args)
#         num_prompt_token_per_slot = sum([int(_) for _ in self.prompt_style if _.isdigit()])
#         prompt_list = []
#         next_prompt_token_id = 0
#         assert 'E' not in self.prompt_style
#         assert '|' not in self.prompt_style
#
#         for i, ch in enumerate(self.prompt_style):
#             if ch.isdigit():
#                 prompt_list.extend(['<prompt_{}>' for _ in range(int(ch))])
#                 next_prompt_token_id += int(ch)
#             elif ch == 'D':
#                 prompt_list.append('<domain>')
#             elif ch == 'S':
#                 prompt_list.append('<slot>')
#
#         prompt = ''.join(prompt_list)
#
#         if '<domain><slot>' in prompt:
#             prompt = prompt.replace('<domain><slot>', '<domain> <slot>')
#
#         for i, ds in enumerate(SLOTIDX2DS):
#             domain, slot = ds.split('-')
#             prompt_id_incr = i * num_prompt_token_per_slot
#             prompt_ids = [prompt_id_incr+i for i in range(num_prompt_token_per_slot)]
#             slot_prompt = prompt.replace('<domain>', domain)
#             slot_prompt = slot_prompt.replace('<slot>', slot)
#             slot_prompt = slot_prompt.format(*prompt_ids)
#             self.questions[domain][slot] = slot_prompt
#
#
#     def __getitem__(self, index) -> Dict:
#         example = None
#         example_ds = None
#         for ds_idx, ds in enumerate(self.ds):
#             ds_num = self.ds2numexamples[ds_idx]
#             if index < ds_num:
#                 example = self.sampled_slot2dialogturns[ds][index]
#                 example_ds = ds
#                 break
#             index -= ds_num
#
#         dialog_input, value = self.build_dst_input_target_sample(example)
#         domain, slot = example_ds.split('-')
#         dialog_input += '{}'.format(self.questions[domain][slot])
#         target = dialog_input + '{}{}'.format(value, self.tokenizer.eos_token)
#
#         return {
#             'dst_input_sequence': dialog_input,
#             'dst_target_sequence': target,
#             'ds': example_ds,
#             'value': value,
#         }
#
#     def process_dialog_for_generation(self, did, end_turn, n_context):
#         if n_context < 0:
#             context_start_turn_id = 0
#             history_start_turn_id = 0
#         else:
#             context_start_turn_id = max(end_turn - n_context + 1, 0)
#             history_start_turn_id = max(context_start_turn_id-self.num_history, 0)
#         sys_utts, user_utts, _, _ = self.read_dialog(did)
#         dialog_input = self.build_dialog_sequence_with_history(user_utts, sys_utts, history_start_turn_id,
#                                                                context_start_turn_id, end_turn)
#         dialog_inputs = []
#         for ds in self.ds:
#             domain, slot = ds.split('-')
#             dialog_inputs.append(dialog_input + '{}'.format(self.questions[domain][slot]))
#         return dialog_inputs
#
#
#     def build_dst_input_target_sample(self, example):
#         did, sampled_history_start_turn_id, sampled_context_start_turn_id, end_turn, value = example
#         sys_utts, user_utts, _, _ = self.read_dialog(did)
#         dialog_input = self.build_dialog_sequence_with_history(user_utts, sys_utts, sampled_history_start_turn_id,
#                                                                sampled_context_start_turn_id, end_turn)
#         return dialog_input, value
#
#
#     def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
#         gen_input_seqs = [x['dst_input_sequence'] for x in batch]
#         dst_ds = [self.ds.index(x['ds']) for x in batch]
#
#         gen_input_dict = self.tokenizer(gen_input_seqs,
#                                         max_length=1024,
#                                         padding=True,
#                                         truncation=True,
#                                         return_tensors='pt')
#
#         gen_input_ids = gen_input_dict['input_ids']
#
#         input_batch = {}
#         if 'dst_target_sequence' in batch[0]:
#             dst_target_seqs = [x['dst_target_sequence'] for x in batch]
#             dst_target_dict = self.tokenizer(dst_target_seqs,
#                                              max_length=1024,
#                                              padding=True,
#                                              truncation=True,
#                                              return_tensors='pt')
#             dst_input_ids = dst_target_dict['input_ids']
#             # print(dst_target_ids.shape)
#             noloss_mask = torch.zeros_like(dst_input_ids).bool()
#             input_ids_len = gen_input_ids.shape[1]
#             # print(dst_input_ids.shape)
#             # print(dst_input_ids)
#             # print(dst_target_ids)
#             noloss_mask[:, :input_ids_len] = dst_input_ids[:, :input_ids_len] == gen_input_ids
#             # print(noloss_mask)
#             # raise
#             dst_target_ids = dst_input_ids.clone()
#             dst_target_ids[noloss_mask] = self.tokenizer.pad_token_id
#
#             input_batch = {
#                 'train_target_ids': dst_target_ids,
#                 'train_input_ids': dst_input_ids,
#                 'train_target_seqs': dst_target_seqs,
#             }
#
#         input_batch.update({
#             "gen_input_ids": gen_input_ids,
#             'gen_input_seqs': gen_input_seqs,
#             'ds': torch.tensor(dst_ds).long(),
#             'value': [_['value'] for _ in batch]
#         })
#
#         return input_batch
#
#
#     def make_sortish_sampler(self, batch_size, **kwargs):
#         return SubsetRandomSampler(list(BatchSampler(SequentialSampler(list(range(self.num_examples))), batch_size, drop_last=False)))
