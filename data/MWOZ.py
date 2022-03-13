from termcolor import colored
import glob
import json
from collections import defaultdict
from tqdm import tqdm

def get_value_dst(DST):
    active_dst = defaultdict(list)
    for k,v in DST.items():
        for k_s, v_s in v['semi'].items():
            if(len(v_s)!=0):
                active_dst[k].append([k_s, v_s])
        for k_s, v_s in v['book'].items():
            if(len(v_s)!=0 and k_s != "booked"):
                active_dst[k].append([k_s, v_s])
    return active_dst



def get_domains(dial):
    dial_domain = set()
    for t in dial['log']:
        if 'metadata' in t:
            active_dst = get_value_dst(t['metadata'])
            services_in_state = set(active_dst.keys())
            dial_domain |= services_in_state
    ret = list(dial_domain)
    ret = ["MWOZ_"+_ for _ in ret]
    return ret


def loadCSV(split):
    split_id = []
    with open(f"data/multiwoz/data/MultiWOZ_2.1/{split}ListFile.txt") as f:
        for l in f:
            split_id.append(l.replace("\n",""))
    return split_id

def preprocessMWOZ(develop=False):
    # todo preprocess for other datasets
    # 1. save states
    # 2. add dataset name prefixes to state slot names
    data = []
    dialogue = json.load(open("data/multiwoz/data/MultiWOZ_2.2/data.json"))
    for i_d, (d_idx, d) in tqdm(enumerate(dialogue.items()),total=len(dialogue.items())):
        dial = {"id":d_idx, "services": get_domains(d), "dataset":"MWOZ"}
        if "MWOZ_police" in dial["services"] or "MWOZ_hospital" in dial["services"] or "MWOZ_bus" in dial["services"]: continue
        turns =[]
        dst_prev = {}
        for t_idx, t in enumerate(d['log']):
            if(t_idx % 2 ==0):
                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"USER","utt":t["text"]})
                # print("USER",t["text"])
                str_API_ACT = ""
                if "dialog_act" in t:
                    intents_act = set()
                    for k,slt in t["dialog_act"].items():
                        # print(k,slt)
                        if "Inform" in k or "Request" in k:
                            str_API_ACT += f"{k.lower().replace('-','_')}("
                            for (s,v) in slt:
                                if s != "none" and v != "none":
                                    v = v.replace('"',"'")
                                    str_API_ACT += f'{s.lower()}="{v}",'
                                    # str_API_ACT += f'{k.lower().replace('-','_')}.{s.lower()} = "{v}" '
                                    intents_act.add(k.lower().replace('-','_'))
                            if(str_API_ACT[-1]==","):
                                str_API_ACT = str_API_ACT[:-1]
                            str_API_ACT += ") "
                # print("API", str_API)
            else:
                dst_api = get_value_dst(t["metadata"])
                # check goal problems
                # services = {_.replace('MWOZ_', '') for _ in dial['services']}
                # services_in_state = set(dst_api.keys())
                # if len(services) > 1 and (not services_in_state.issubset(services)):
                #     print(d_idx, services_in_state, services)
                #     input()

                str_API = ""
                intents = set()
                state = {}
                for k,slt in dst_api.items():
                    str_API += f"{k.lower().replace('-','_')}("
                    for (s,v) in slt:
                        ds = f'MWOZ_{k}-{s}'
                        if len(v)!= 0:
                            v = v[0].replace('"',"'")
                            str_API += f'{s.lower()}="{v}",'
                            intents.add(k.lower().replace('-','_'))
                            state[ds] = v
                        else:
                            state[ds] = 'none'
                    if(len(str_API)>0 and str_API[-1]==","):
                        str_API = str_API[:-1]
                    str_API += ") "
                if(str_API==""):
                    turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API","utt":str_API_ACT,"service":list(intents_act), "state":{}})
                    # print("API",str_API_ACT)
                else:
                    turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API","utt":str_API,"service":list(intents), "state":state})
                    # print("API", str_API)

                ## API RETURN
                str_ACT = ""
                if "dialog_act" in t:
                    for k,slt in t["dialog_act"].items():
                        # print(k,slt)
                        if "Inform" in k or "Recommend" in k or "Booking-Book" in k or "-Select" in k:
                            str_ACT += f"{k.lower().replace('-','_')}("
                            for (s,v) in slt:
                                if s != "none" and v != "none":
                                    v = v.replace('"',"'")
                                    str_ACT += f'{s.lower()}="{v}",'
                                    # str_ACT += f'{k.lower().replace("-",".")}.{s.lower()} = "{v}" '
                            if(str_ACT[-1]==","):
                                str_ACT = str_ACT[:-1]
                            str_ACT += ") "
                        if "Booking-NoBook" in k:
                            # str_ACT += f'{k.lower().replace("-",".")} '
                            str_ACT += f"{k.lower().replace('-','_')}() "

                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API-OUT","utt":str_ACT,"service":None})
                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"SYSTEM","utt":t["text"]})
        dial["dialogue"] = turns
        data.append(dial)
        if(develop and i_d==1000): break

    split_id_dev, split_id_test = loadCSV("val"), loadCSV("test")

    train_data, dev_data, test_data = [], [], []

    for dial in data:
        if dial["id"] in split_id_dev:
           dev_data.append(dial)
        elif dial["id"] in split_id_test:
           test_data.append(dial)
        else:
           train_data.append(dial)
    return train_data, dev_data, test_data
