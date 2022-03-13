import json

import torch
import numpy as np
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from functools import partial
from utils.preprocess import get_datasets
from collections import defaultdict, OrderedDict
import pprint
import random
from tabulate import tabulate
from dictdiffer import diff
import re

pp = pprint.PrettyPrinter(indent=4)

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]


def parse_utt2state(text):
    API = OrderedDict()
    # assume only one intent
    intent, svs = re.match('(.*?)\((.*)\)', text).groups()
    API[intent] = OrderedDict()
    for _, s, v in re.findall('(^|,)(.*?)="(.*?)"', svs):
        v = '"{}"'.format(v)
        API[intent][s] = v
    if len(API[intent]) == 0:
        API[intent]["none"] = "none"
    return API


def cal_state_update(prev_state_utt, cur_state_utt):
    if not prev_state_utt or not cur_state_utt:
        return cur_state_utt
    prev_state = parse_utt2state(prev_state_utt.strip())
    cur_state = parse_utt2state(cur_state_utt.strip())
    prev_intent = list(prev_state.keys())[0]
    cur_intent = list(cur_state.keys())[0]
    prev_state = {s: v for s, v in prev_state[prev_intent].items() if s != "none"}
    cur_state = {s: v for s, v in cur_state[cur_intent].items() if s != "none"}
    diff_state = {s: v for s, v in cur_state.items() if s not in prev_state or prev_state[s] != v}
    # print(prev_state)
    # print(cur_state)
    # print(diff_state)
    # turn diff state into utt again
    diff_state_utt = cur_intent + '('
    for s, v in diff_state.items():
        diff_state_utt += '{}={},'.format(s, v)
    if diff_state_utt[-1] == ",":
        diff_state_utt = diff_state_utt[:-1]
    diff_state_utt += ")"
    return diff_state_utt


class DatasetTrain(Dataset):
    """Custom data.Dataset compatible with DataLoader."""

    def __init__(self, data, domains=None):
        self.data = data
        self.dataset_len = len(self.data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = self.data[index]
        return item

    def __len__(self):
        return self.dataset_len


def get_intent_from_dial(args, data, task_id, tokenizer):
    dialogues = []
    utt_len = []
    hist_len = []
    for dial in data:
        plain_history = []
        for idx_t, t in enumerate(dial['dialogue']):
            ## DUPLICATE DIALOGUE
            if f'{t["id"]}' == "dlg-ff2b8de2-467d-4917-be13-1529765752e9":
                continue
            if (t['spk'] == "USER"):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif (t['spk'] == "API-OUT"):
                pass
            elif ((t['spk'] == "SYSTEM") and idx_t != 0 and t["utt"] != ""):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif ((t['spk'] == "API") and idx_t != 0 and t["utt"] != ""):
                intent = " ".join(t["service"])
                dialogues.append({"history": " ".join(plain_history[-args.max_history:]),
                                  "reply": f'{intent} {tokenizer.eos_token}',
                                  "history_reply": " ".join(
                                      plain_history[-args.max_history:]) + f'[SOS]{intent} {tokenizer.eos_token}',
                                  "spk": t["spk"],
                                  "dataset": t["dataset"],
                                  "dial_id": t["id"],
                                  "turn_id": t["turn_id"],
                                  "task_id": task_id,
                                  })
    if args.verbose:
        for d in random.sample(dialogues, len(dialogues)):
            pp.pprint(d)
            break
        print()
        input()
    return dialogues


def get_DST_from_dial(args, data, task_id, tokenizer, max_train_dials_per_domain=None, split=None):
    dialogues = []
    if split == 'train' and  max_train_dials_per_domain is not None:
        data = data[:max_train_dials_per_domain]
    if split == 'dev' and max_train_dials_per_domain is not None:
        data = data[:(max_train_dials_per_domain//3)]

    for dial in data:
        plain_history = []
        if args.CL in ['VANILLA', 'EWC', 'REPLAY']:
            plain_history.append('domain: {}'.format(task_id))
        all_slots = []
        for idx_t, t in enumerate(dial['dialogue']):
            ## DUPLICATE DIALOGUE
            if f'{t["id"]}' == "dlg-ff2b8de2-467d-4917-be13-1529765752e9":
                continue
            if args.samples_per_domain == len(dialogues):
                break
            if (t['spk'] == "USER"):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif (t['spk'] == "API-OUT"):
                pass
            elif ((t['spk'] == "SYSTEM") and idx_t != 0 and t["utt"] != ""):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif ((t['spk'] == "API") and idx_t != 0):
                slots = t["utt"].strip()
                dialogues.append({"history": " ".join(plain_history[-args.max_history:]),
                                  "reply": f'{slots} {tokenizer.eos_token}',
                                  "history_reply": " ".join(
                                      plain_history[-args.max_history:]) + f'[SOS]{slots} {tokenizer.eos_token}',
                                  "spk": t["spk"],
                                  "dataset": t["dataset"],
                                  "dial_id": t["id"],
                                  "turn_id": t["turn_id"],
                                  "task_id": task_id,
                                  'state': t['state']
                                  })

    if args.verbose:
        for d in random.sample(dialogues, len(dialogues)):
            pp.pprint(d)
            break
        print()
        input()
    return dialogues


def get_NLG_from_dial(args, data, task_id, tokenizer):
    dialogues = []
    utt_len = []
    hist_len = []
    for dial in data:
        plain_history = []
        latest_API_OUT = "API-OUT: "
        for idx_t, t in enumerate(dial['dialogue']):
            ## DUPLICATE DIALOGUE
            if f'{t["id"]}' == "dlg-ff2b8de2-467d-4917-be13-1529765752e9":
                continue
            if (t['spk'] == "USER"):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif (t['spk'] == "API-OUT"):
                latest_API_OUT = f"{t['utt'].strip()}"
            elif ((t['spk'] == "SYSTEM") and idx_t != 0 and t["utt"] != ""):
                if (latest_API_OUT != ""):
                    dialogues.append({"history": latest_API_OUT,
                                      "reply": f'{t["utt"].strip()} {tokenizer.eos_token}',
                                      "history_reply": latest_API_OUT + f'[SOS]{t["utt"].strip()} {tokenizer.eos_token}',
                                      "spk": t["spk"],
                                      "dataset": t["dataset"],
                                      "dial_id": t["id"],
                                      "turn_id": t["turn_id"],
                                      "task_id": task_id})
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
                latest_API_OUT = ""

    if args.verbose:
        for d in random.sample(dialogues, len(dialogues)):
            pp.pprint(d)
            break
        print()
        input()
    return dialogues


def get_e2e_from_dial(args, data, task_id, tokenizer):
    dialogues = []
    utt_len = []
    hist_len = []
    for dial in data:
        plain_history = []
        latest_API_OUT = "API-OUT: "
        for idx_t, t in enumerate(dial['dialogue']):
            ## DUPLICATE DIALOGUE
            if f'{t["id"]}' == "dlg-ff2b8de2-467d-4917-be13-1529765752e9" and f'{t["id"]}' == "dlg-fdd242eb-56be-48c0-a56e-5478472500d0":
                continue
            if (t['spk'] == "USER"):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif (t['spk'] == "API-OUT"):
                latest_API_OUT = f"{t['spk']}: {t['utt'].strip()}"
            elif ((t['spk'] == "SYSTEM") and idx_t != 0 and t["utt"] != ""):
                dialogues.append({"history": " ".join(plain_history[-args.max_history:] + [latest_API_OUT]),
                                  "reply": f'{t["utt"].strip()} {tokenizer.eos_token}',
                                  "history_reply": " ".join(plain_history[-args.max_history:] + [
                                      latest_API_OUT]) + f'[SOS]{t["utt"].strip()} {tokenizer.eos_token}',
                                  "spk": t["spk"],
                                  "dataset": t["dataset"],
                                  "dial_id": t["id"],
                                  "turn_id": t["turn_id"],
                                  "task_id": task_id})
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
                latest_API_OUT = "API-OUT: "
            elif ((t['spk'] == "API") and idx_t != 0 and t["utt"] != ""):
                dialogues.append({"history": " ".join(plain_history[-args.max_history:]),
                                  "reply": f'{t["utt"].strip()} {tokenizer.eos_token}',
                                  "history_reply": " ".join(plain_history[
                                                            -args.max_history:]) + f'[SOS]{t["utt"].strip()} {tokenizer.eos_token}',
                                  "spk": t["spk"],
                                  "dataset": t["dataset"],
                                  "dial_id": t["id"],
                                  "turn_id": str(t["turn_id"]) + "API",
                                  "task_id": task_id})
    if args.verbose:
        for d in random.sample(dialogues, len(dialogues)):
            pp.pprint(d)
            break
        print()
        input()
    return dialogues


def get_current_task_data(args, dataset_dic, task_id, number_of_sample):
    temp_aug = random.sample(dataset_dic, min(number_of_sample, len(dataset_dic)))
    aug_data = []
    cnt_API = 0
    for d in temp_aug:
        ## add a first token for the generation
        if (args.task_type == "E2E"):
            if (d["spk"] == "API"):
                cnt_API += 1
                d["history_reply"] = f"[{str(eval(task_id)[0])}-API]" + d["history_reply"]
            else:
                d["history_reply"] = f"[{str(eval(task_id)[0])}]" + d["history_reply"]
        else:
            d["history_reply"] = f"[{str(eval(task_id)[0])}]" + d["history_reply"]
        aug_data.append(d)
    return aug_data


def collate_fn(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["history"], padding=True, return_tensors="pt", truncation=False,
                            add_special_tokens=False)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(batch_data["reply"], padding=True, return_tensors="pt", truncation=False,
                             add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids'] == tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']
    return batch_data


def collate_fn_GPT2(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["history_reply"], padding=True, return_tensors="pt", truncation=False,
                            add_special_tokens=False, return_attention_mask=False)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = None
    output_batch = tokenizer(batch_data["history_reply"], padding=True, return_tensors="pt", truncation=False,
                             add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids'] == tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']

    #### DATA FOR COMPUTING PERPLEXITY OF DIALOGUE HISTORY ==> FOR ADAPTER
    batched_history = tokenizer(batch_data["history"], padding=True, return_tensors="pt", truncation=False,
                                add_special_tokens=False, return_attention_mask=False)
    batch_data["input_id_PPL"] = batched_history['input_ids']
    batched_history_out = tokenizer(batch_data["history"], padding=True, return_tensors="pt", truncation=False,
                                    add_special_tokens=False, return_attention_mask=False)
    batched_history_out['input_ids'].masked_fill_(batched_history_out['input_ids'] == tokenizer.pad_token_id, -100)
    batch_data["output_id_PPL"] = batched_history_out['input_ids']  ### basically just remove pad from ppl calculation
    return batch_data


def collate_fn_DialoGPT(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        if key in ["history_reply", "history"]:
            batch_data[key] = [d[key].replace("USER: ", "", 1).replace("USER:", tokenizer.eos_token).replace("SYSTEM:", tokenizer.eos_token) for d in data]
        else:
            batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["history_reply"], padding=True, return_tensors="pt", truncation=False,
                            add_special_tokens=False, return_attention_mask=False)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = None
    output_batch = tokenizer(batch_data["history_reply"], padding=True, return_tensors="pt", truncation=False,
                             add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids'] == tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']

    #### DATA FOR COMPUTING PERPLEXITY OF DIALOGUE HISTORY ==> FOR ADAPTER
    batched_history = tokenizer(batch_data["history"], padding=True, return_tensors="pt", truncation=False,
                                add_special_tokens=False, return_attention_mask=False)
    batch_data["input_id_PPL"] = batched_history['input_ids']
    batched_history_out = tokenizer(batch_data["history"], padding=True, return_tensors="pt", truncation=False,
                                    add_special_tokens=False, return_attention_mask=False)
    batched_history_out['input_ids'].masked_fill_(batched_history_out['input_ids'] == tokenizer.pad_token_id, -100)
    batch_data["output_id_PPL"] = batched_history_out['input_ids']  ### basically just remove pad from ppl calculation
    return batch_data


def collate_fn_t5(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_dict = tokenizer(batch_data['history'],
                           max_length=1024,
                           padding=True,
                           truncation=True,
                           return_tensors='pt')
    batch_data["encoder_input"] = input_dict['input_ids']
    batch_data["attention_mask"] = input_dict['attention_mask']

    output_dict = tokenizer(batch_data['reply'],
                           max_length=1024,
                           padding=True,
                           truncation=True,
                           return_tensors='pt')
    batch_data["decoder_output"] = output_dict['input_ids']
    return batch_data


def make_loader(args, list_sample, tokenizer):
    # collate_fn_ = collate_fn_DialoGPT if ("gpt2" in args.model_checkpoint) else collate_fn
    if "dialogpt" in args.model_checkpoint:
        collate_fn_ = collate_fn_DialoGPT
    elif "gpt2" in args.model_checkpoint:
        collate_fn_ = collate_fn_GPT2
    elif 't5' in args.model_checkpoint:
        collate_fn_ = collate_fn_t5
    else:
        collate_fn_ = collate_fn
    return DataLoader(DatasetTrain(list_sample), batch_size=args.train_batch_size, shuffle=True,
                      collate_fn=partial(collate_fn_, tokenizer=tokenizer))


def make_test_loader(args, list_sample, tokenizer):
    if 'gpt2' in args.model_checkpoint:
        collate_fn_ = collate_fn_GPT2
    elif 't5' in args.model_checkpoint:
        collate_fn_ = collate_fn_t5
    else:
        raise
    return DataLoader(DatasetTrain(list_sample), batch_size=args.valid_batch_size, shuffle=False,
                      collate_fn=partial(collate_fn_, tokenizer=tokenizer))


def get_data_loaders(args, tokenizer, test=False, inclusive_domains=None, max_train_dials_per_domain=None):
    """ Prepare the dataset for training and evaluation """
    aggregate = get_datasets(dataset_list=args.dataset_list.split(','), setting=args.setting, verbose=args.verbose,
                             develop=args.debug, inclusive_domains=inclusive_domains)

    if args.CL == 'RAND_PROMPT' or args.CL == 'META_PROMPT' or args.CL == 'MLM_PROMPT' or args.CL == 'MULTI_PROMPT' or args.CL == 'CL_INIT_PROMPT' \
            or args.CL == 'MEM_META_PROMPT' or args.CL == 'MULTITASK_INIT_PROMPT' or 'PROMPT' in args.CL:
        # set collate_fn_ here to run the code but dataloaders are not used
        collate_fn_ = collate_fn_GPT2
    else:
        if "dialogpt" in args.model_checkpoint:
            collate_fn_ = collate_fn_DialoGPT
        elif "gpt2" in args.model_checkpoint:
            collate_fn_ = collate_fn_GPT2
        elif 't5' in args.model_checkpoint:
            collate_fn_ = collate_fn_t5
        else:
            collate_fn_ = collate_fn


    if (test):
        datasets = {"test": {}}
    else:
        datasets = {"train": {}, "dev": {}, "test": {}}

    for split in datasets.keys():
        for tasks_id, task in aggregate["BYDOMAIN"][split].items():
            if (args.task_type == "E2E"):
                datasets[split][tasks_id] = get_e2e_from_dial(args, task, tasks_id, tokenizer)
            elif (args.task_type == "INTENT"):
                datasets[split][tasks_id] = get_intent_from_dial(args, task, tasks_id, tokenizer)
            elif (args.task_type == "DST"):
                datasets[split][tasks_id] = get_DST_from_dial(args, task, tasks_id, tokenizer, max_train_dials_per_domain, split)
            elif (args.task_type == "NLG"):
                datasets[split][tasks_id] = get_NLG_from_dial(args, task, tasks_id, tokenizer)

    if 'PROMPT' not in args.CL:
        for split in datasets.keys():
            for tasks_id, dials in datasets[split].items():
                for dial in dials:
                    dial.pop('state')

    task_id_train = set(task_id for task_id, dataset_task in datasets["train"].items())
    task_id_dev = set(task_id for task_id, dataset_task in datasets["dev"].items())
    task_id_test = set(task_id for task_id, dataset_task in datasets["test"].items())
    common_task_id = list(task_id_train & task_id_dev & task_id_test)

    # if max_train_dials_per_domain is not None:
    #     datasets['train'] = {k: v[:max_train_dials_per_domain] for k, v in datasets['train'].items()}
    #     datasets['dev'] = {k: v[:max_train_dials_per_domain//3] for k, v in datasets['dev'].items()}

    # # some dialogs are excluded in todcl original process
    # task = defaultdict(lambda: defaultdict(str))
    # for split in ["train", "dev", "test"]:
    #     for task_id, dataset_task in datasets[split].items():
    #         task[task_id][split] = len(dataset_task)
    # for task_id in task:
    #     if any([_ == 0 for _ in task[task_id].values()]):
    #         for split in ["train", "dev", "test"]:
    #             datasets[split].pop(task_id)

    ### LOGGING SOME INFORMATION ABOUT THE TASKS
    print(f"Tasks: {common_task_id}")
    print(f"Num Example of Tasks {len(common_task_id)}")
    task = defaultdict(lambda: defaultdict(str))
    for split in ["train", "dev", "test"]:
        for task_id, dataset_task in datasets[split].items():
            task[task_id][split] = len(dataset_task)

    table = []
    for dom_name, split_len in task.items():
        table.append({"dom": dom_name,
                      "train dials": len(aggregate["BYDOMAIN"]["train"][dom_name]),
                      "dev dials": len(aggregate["BYDOMAIN"]["dev"][dom_name]),
                      "test dials": len(aggregate["BYDOMAIN"]["test"][dom_name]),
                      "train samples": split_len["train"],
                      "dev samples": split_len["dev"],
                      "test samples": split_len["test"]})
    print(tabulate(table, headers="keys"))
    text_file = open("data_stat.csv", "w")
    text_file.write(tabulate(table, headers="keys", tablefmt="tsv").replace('\t', ','))
    text_file.close()


    train_loaders = {}
    valid_loaders = {}
    train_datasets = {}
    val_datasets = {}
    test_loaders = {}

    if (args.CL != "MULTI"):
        if (not test):
            for task_id, dataset_task in datasets["train"].items():
                if (task_id in common_task_id):
                    train_loaders[task_id] = DataLoader(DatasetTrain(dataset_task), batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        collate_fn=partial(collate_fn_, tokenizer=tokenizer))
                    train_datasets[task_id] = dataset_task
            for task_id, dataset_task in datasets["dev"].items():
                if (task_id in common_task_id):
                    valid_loaders[task_id] = DataLoader(DatasetTrain(dataset_task), batch_size=args.valid_batch_size,
                                                        shuffle=False,
                                                        collate_fn=partial(collate_fn_, tokenizer=tokenizer))
                    val_datasets[task_id] = dataset_task

            for task_id, dataset_task in datasets["test"].items():
                if (task_id in common_task_id):
                    test_loaders[task_id] = DataLoader(DatasetTrain(dataset_task), batch_size=args.valid_batch_size,
                                                        shuffle=False,
                                                        collate_fn=partial(collate_fn_, tokenizer=tokenizer))

    elif (args.CL == "MULTI"):
        if (not test):
            dataset_train = []
            for task_id, dataset_task in datasets["train"].items():
                if (task_id in common_task_id):
                    dataset_train += dataset_task
            train_loaders = DataLoader(DatasetTrain(dataset_train), batch_size=args.train_batch_size, shuffle=True,
                                       collate_fn=partial(collate_fn_, tokenizer=tokenizer))

            dataset_dev = []
            for task_id, dataset_task in datasets["dev"].items():
                if (task_id in common_task_id):
                    dataset_dev += dataset_task
            valid_loaders = DataLoader(DatasetTrain(dataset_dev), batch_size=args.valid_batch_size, shuffle=False,
                                       collate_fn=partial(collate_fn_, tokenizer=tokenizer))

    temp_list = []
    for task_id, dataset_task in datasets["test"].items():
        if (task_id in common_task_id):
            temp_list.append(dataset_task)
    test_datasets = sum(temp_list, [])
    # all_data_test_loaders = DataLoader(DatasetTrain(sum(temp_list, [])), batch_size=args.test_batch_size, shuffle=False,
    #                           collate_fn=partial(collate_fn_, tokenizer=tokenizer))

    ### THIS IS JUST FOR CHECKING DUPLICATE DIALOGUES
    # testing_dict = defaultdict(list)
    # for idx_b, batch in tqdm(enumerate(test_loaders), total=len(test_loaders)):
    #     for (d_id, t_id, ta_id) in zip(batch["dial_id"], batch["turn_id"], batch["task_id"]):
    #         if (f'{d_id}_{t_id}_{ta_id}' not in testing_dict):
    #             testing_dict[f'{d_id}_{t_id}_{ta_id}'].append(1)
    #         else:
    #             print(f'{d_id}_{t_id}_{ta_id}')

    return train_loaders, valid_loaders, test_loaders, (train_datasets, val_datasets, test_datasets)
