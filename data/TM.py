import glob
import json
import csv
import re
import numpy as np
from tqdm import tqdm
from termcolor import colored
from collections import defaultdict
from tabulate import tabulate
from copy import deepcopy

DOMAINS =["uber", "movie", "restaurant", "coffee", "pizza", "auto", "sport", "flight", "food-ordering", "hotel", "music"]

def remove_numbers_from_string_bak(s):
    numbers = re.findall(r'\d+', s)
    for n in numbers:
        s = s.replace(n,"")
    return s


def remove_numbers_from_string(s):
    return s.lower()


def create_API_str(API_struct):
    str_API = ""
    for k,v in API_struct.items():
        str_API += f"{k}("
        for arg, val in v.items():
            val = val.replace('"',"'")
            str_API += f'{arg}="{val}",'

        str_API = str_API[:-1]
        str_API += ") "
    return str_API

def prepocess_API_bak(frame):
    if(frame==""):
        return "",None,[]
    else:
        API_struct = defaultdict(lambda: defaultdict(str))
        services = set()
        for s in frame:
            parsed = remove_numbers_from_string(s["annotations"][-1]["name"]).split(".")
            API_struct[parsed[0]]["_".join(parsed[1:])] = s["text"]
            services.add("_".join(parsed[1:]))
        services = [s.strip() for s in services]
        return create_API_str(API_struct), API_struct, list(services)



def fix_turn_bak(turns):
    for i_t, t in enumerate(turns):
        if(t['spk']=="API-OUT" or t['spk']=="API"):
            t['utt'], t['n_struct'], t['service'] = prepocess_API(t["struct"])
    if len(turns)>0 and turns[0]['spk'] == "API-OUT":
        turns = turns[1:]

    if len(turns)>0 and turns[-1]['spk'] == "API":
        turns = turns[:-1]
    new_turns = []
    for i_t, t in enumerate(turns):
        if(t['spk']=="API-OUT" and t['utt']!=""):
            if turns[i_t-1]['utt']=="":
                new_turns[-1]["utt"] = list(turns[i_t]['n_struct'].keys())[0]+"()"
                new_turns[-1]["service"] = [list(turns[i_t]['n_struct'].keys())[0]]

        new_turns.append(turns[i_t])
    return new_turns


def prepocess_API(frame, domain_name):
    # 修改：tm数据集中的domain name和标注中的domain name不一致
    # 比如 domain name = restaurant-table-2， 标注=restaurant_reservation.location.restaurant
    if(frame==""):
        return "",None,[]
    else:
        API_struct = defaultdict(lambda: defaultdict(str))
        services = set()
        for s in frame:
            parsed = remove_numbers_from_string(s["annotations"][-1]["name"]).split(".")
            # API_struct[parsed[0]]["_".join(parsed[1:])] = s["text"]
            API_struct[domain_name]["_".join(parsed[1:])] = s["text"]
            services.add("_".join(parsed[1:]))
        services = [s.strip() for s in services]
        return create_API_str(API_struct), API_struct, list(services)


def fix_turn(turns, domain_name):
    for i_t, t in enumerate(turns):
        if(t['spk']=="API-OUT" or t['spk']=="API"):
            t['utt'], t['n_struct'], t['service'] = prepocess_API(t["struct"], domain_name)
    if len(turns)>0 and turns[0]['spk'] == "API-OUT":
        turns = turns[1:]

    if len(turns)>0 and turns[-1]['spk'] == "API":
        turns = turns[:-1]
    new_turns = []
    for i_t, t in enumerate(turns):
        if(t['spk']=="API-OUT" and t['utt']!=""):
            if turns[i_t-1]['utt']=="":
                new_turns[-1]["utt"] = list(turns[i_t]['n_struct'].keys())[0]+"()"
                new_turns[-1]["service"] = [list(turns[i_t]['n_struct'].keys())[0]]

        new_turns.append(turns[i_t])
    return new_turns

def get_data(dialogue,year,develop=False):
    data = []
    for i_d,d in tqdm(enumerate(dialogue),total=len(dialogue)):

        #### GET THE DOMAIN OF THE DIALOGUE
        flag = True
        serv = ""
        for dom in DOMAINS:
            if(dom in d["instruction_id"]):
                serv = f"{dom}"
                flag = False
            elif("hungry" in d["instruction_id"] or
                "dinner" in d["instruction_id"] or
                "lunch" in d["instruction_id"] or
                "dessert" in d["instruction_id"]):
                serv = f"restaurant"
                flag = False
            elif("nba" in d["instruction_id"] or
                 "mlb" in d["instruction_id"] or
                 "epl" in  d["instruction_id"] or
                 "mls" in d["instruction_id"] or
                 "nfl" in d["instruction_id"] ):
                serv = f"sport"
                flag = False
        if(flag): print(d["instruction_id"])
        ####
        # serv = d["instruction_id"].replace('-', '_')

        serv = serv.replace('-', '_')
        dial = {"id":d["conversation_id"].strip(), "services": [f"TM{year}_"+serv], "dataset":f"TM{year}"}
        turns =[]
        state = []
        for t_idx, t in enumerate(d["utterances"]):
            if(t["speaker"]=="USER"):
                cur_da = t["segments"] if "segments" in t else []
                state += cur_da
                if(len(turns)!=0 and turns[-1]['spk']=="API"):
                    turns[-2]["utt"] += " "+t["text"]
                    turns[-1]["struct"] = deepcopy(state)
                    # if "segments" in t and type(turns[-1]["struct"])==list:
                    #     turns[-1]["struct"] += t["segments"]
                    # elif("segments" in t and type(turns[-1]["struct"])==str):
                    #     turns[-1]["struct"] = t["segments"]
                    # else:
                    #     turns[-1]["struct"] = ""
                else:
                    turns.append({"dataset":f"TM{year}","id":d["conversation_id"].strip(),"turn_id":t_idx,"spk":t["speaker"],"utt":t["text"]})
                    turns.append({"dataset":f"TM{year}","id":d["conversation_id"].strip(),"turn_id":t_idx,
                                  "spk":"API","utt":"","struct":deepcopy(state),"service":[]})
            else:
                if(len(turns)!=0 and turns[-1]['spk']=="SYSTEM"):
                    turns[-1]["utt"] += " "+t["text"]
                    if "segments" in t and type(turns[-2]["struct"])==list:
                        turns[-2]["struct"] += t["segments"]
                    elif("segments" in t and type(turns[-2]["struct"])==str):
                        turns[-2]["struct"] = t["segments"]
                    else:
                        turns[-2]["struct"] = ""
                else:
                    turns.append({"dataset":f"TM{year}","id":d["conversation_id"].strip(),"turn_id":t_idx,"spk":"API-OUT","utt":"","struct":t["segments"] if "segments" in t else "","service":[]})
                    turns.append({"dataset":f"TM{year}","id":d["conversation_id"].strip(),"turn_id":t_idx,"spk":"SYSTEM","utt":t["text"]})

        # dial["dialogue"] = fix_turn(turns)
        dial["dialogue"] = fix_turn(turns, serv)

        # 有些标注有问题。tm应该是全部是single domain，但有些对话中标注了多个domain
        flag = True
        for turn_no, turn in enumerate(dial['dialogue']):
            if turn['spk'] == 'API':
                # assert len(turn['n_struct']) <= 1, print(turn['n_struct'].keys(), json.dumps(d, indent=4))
                if len(turn['n_struct']) > 1:
                    flag = False
                    print('TM: wrong annotation')
                    break
                turn['state'] = {}
                for k, v in turn['n_struct'][serv].items():
                    # reject 和 accept 应该表示system预定是否成功，在user里不应该出现
                    k = k.replace('_reject', '')
                    k = k.replace('_accept', '')
                    turn['state']['{}-{}'.format(dial['services'][0].lower(), k.lower())] = v.lower()
        if flag:
            data.append(dial)
        if(develop and i_d==10): break
    return data

def preprocessTM2019(develop=False):
    dialogue = json.load(open("data/Taskmaster/TM-1-2019/woz-dialogs.json"))
    data = get_data(dialogue,"A",develop)

    # print(json.dumps(data[0], indent=4))
    # input('preprocesstm2019')

    data_by_domain = defaultdict(list)
    for dial in data:
        if(len(dial["services"])==1):
            data_by_domain[str(sorted(dial["services"]))].append(dial)

    data_by_domain_new = defaultdict(list)
    for dom, data in data_by_domain.items():
        train_data, dev_data, test_data = np.split(data, [int(len(data)*0.8), int(len(data)*0.9)])
        data_by_domain_new[str(sorted([remove_numbers_from_string(s) for s in eval(dom)]))].append([train_data, dev_data, test_data])

    train_data, valid_data, test_data = [], [], []
    table = []
    for dom, list_of_split in data_by_domain_new.items():
        train, valid, test = [], [], []
        for [tr,va,te] in list_of_split:
            train += rename_service_dialogue(tr,dom)
            valid += rename_service_dialogue(va,dom)
            test += rename_service_dialogue(te,dom)
        table.append({"dom":dom, "train":len(train), "valid":len(valid), "test":len(test)})
        train_data += train
        valid_data += valid
        test_data += test
    print(tabulate(table, headers="keys"))

    return train_data,valid_data,test_data


def rename_service_dialogue(dial_split,name):
    new_dial = []
    for dial in dial_split:
        dial["services"] = eval(name)
        new_dial.append(dial)
    return new_dial

def preprocessTM2020(develop=False):
    data = []
    for f in glob.glob(f"data/Taskmaster/TM-2-2020/data/*.json"):
        dialogue = json.load(open(f))
        data += get_data(dialogue,"B",develop)

    data_by_domain = defaultdict(list)
    for dial in data:
        if(len(dial["services"])==1):
            data_by_domain[str(sorted(dial["services"]))].append(dial)

    data_by_domain_new = defaultdict(list)
    for dom, data in data_by_domain.items():
        train_data, dev_data, test_data = np.split(data, [int(len(data)*0.8), int(len(data)*0.9)])
        data_by_domain_new[str(sorted([remove_numbers_from_string(s) for s in eval(dom)]))].append([train_data, dev_data, test_data])

    train_data, valid_data, test_data = [], [], []
    table = []
    for dom, list_of_split in data_by_domain_new.items():
        train, valid, test = [], [], []
        for [tr,va,te] in list_of_split:
            train += rename_service_dialogue(tr,dom)
            valid += rename_service_dialogue(va,dom)
            test += rename_service_dialogue(te,dom)
        table.append({"dom":dom, "train":len(train), "valid":len(valid), "test":len(test)})
        train_data += train
        valid_data += valid
        test_data += test
    print(tabulate(table, headers="keys"))
    return train_data,valid_data,test_data
