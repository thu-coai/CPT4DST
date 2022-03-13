import itertools
from functools import partial
from pprint import pprint

import torch
import json
import os
import os.path
import math
import glob
import re
import shutil
import time
from random import sample
import pytorch_lightning as pl
import random
from pytorch_lightning import Trainer, seed_everything
from tqdm import tqdm

from utils.dataloader import get_data_loaders, get_current_task_data, make_loader, DatasetTrain, make_test_loader, \
    collate_fn_DialoGPT, collate_fn_GPT2, collate_fn_t5


from test import test_model_seq2seq, generate_sample_prev_task, test_model_seq2seq_ADAPTER, test_t5_adapter
from scorer import score_folder
from collections import defaultdict
from CL_learner import Seq2SeqToD
from argparse import ArgumentParser

from torch.utils.data import DataLoader
# from prompt_files.prompt_module import CLPromptDSTModule, MetaPromptDSTModule, MemMetaPromptDSTModule, \
#     MultiTaskInitModule, prepare_prompt_parser
# from prompt_files.prompt_dataset import CLT5PromptDSTDataset, \
#     T5MLMPromptDataset, MultiPromptDSTDataset
from prompt_files.prompt_test import predict
from copy import deepcopy
import torch.multiprocessing

DEVICE = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'

torch.multiprocessing.set_sharing_strategy('file_system')


def get_checkpoint(log_dir, index_to_load):
    file = glob.glob(f"{log_dir}/*")
    for f in file:
        f_noprefix = f.replace(f"{log_dir}", "")
        num = [int(s) for s in re.findall(r'\d+', f_noprefix)]
        if index_to_load in num:
            version = os.listdir(f + "/lightning_logs")[0]
            check_name = os.listdir(f + "/lightning_logs/" + version + "/checkpoints/")[0]
            # checkpoint_name = f.replace("[","\[").replace("]","\]").replace("\'","\\'")+"/lightning_logs/"+ version+"/checkpoints/"+check_name
            checkpoint_name = f + "/lightning_logs/" + version + "/checkpoints/" + check_name
    return checkpoint_name


def main(hparams):
    seed_everything(hparams.seed)
    if not hparams.saving_dir:
        if (hparams.CL == "ADAPTER"):
            hparams.saving_dir = "runs_{}/{}/{}_EPC_{}_LR_{}_BZ_{}_NADA_{}_BOTL_{}_PERM_{}".format(
                hparams.task_type, hparams.pardir, hparams.CL, hparams.n_epochs, hparams.lr, hparams.train_batch_size,
                hparams.number_of_adpt, hparams.bottleneck_size, hparams.seed
            )
        else:
            hparams.saving_dir = "runs_{}/{}/{}_EPC_{}_LR_{}_BZ_{}_EM_{}_LAMOL_{}_REG_{}_PERM_{}".format(
                hparams.task_type, hparams.pardir, hparams.CL, hparams.n_epochs, hparams.lr, hparams.train_batch_size,
                hparams.episodic_mem_size, hparams.percentage_LAM0L, hparams.reg, hparams.seed
            )
    if (hparams.CL == "MULTI"):
        hparams.multi = True
        hparams.continual = False
    else:
        hparams.multi = False
        hparams.continual = True

    model = Seq2SeqToD(hparams)
    data_dir = 'data/{}_{}_history_{}_sample_{}_bz_{}_{}_{}/'.format(
        hparams.CL if hparams.CL in ["MULTI", 'MULTI_PROMPT'] else "CL",
        hparams.dataset_list, hparams.max_history, hparams.samples_per_domain,
        hparams.train_batch_size, hparams.valid_batch_size, hparams.test_batch_size,
    )
    os.makedirs(data_dir, exist_ok=True)
    # if not hparams.use_cache:
    train_loader, val_loader, test_loader, (train_datasets, val_datasets, test_datasets) = get_data_loaders(hparams,
                                                                                                            model.tokenizer,
                                                                                                            inclusive_domains=hparams.inclusive_domains,
                                                                                                            max_train_dials_per_domain=hparams.max_train_dials_per_domain)
    json.dump(test_datasets, open('test_dials.json', 'w'), indent=4)

    seed_everything(hparams.seed)
    keys = list(train_loader.keys())
    random.Random(42).shuffle(keys)

    if hparams.dataset_order == 1:
        dataset_order = ["['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                         "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                         "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
    elif hparams.dataset_order == 2:
        dataset_order = ["['sgd_hotels_4']", "['sgd_flights_3']", "['sgd_rentalcars_2']", "['sgd_rentalcars_3']",
                         "['sgd_media_2']", "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_trains_1']",
                         "['sgd_services_3']", "['sgd_homes_2']", "['sgd_hotels_3']", "['sgd_flights_1']",
                         "['sgd_services_4']", "['sgd_homes_1']", "['sgd_hotels_1']"]
    elif hparams.dataset_order == 3:
        dataset_order = ["['sgd_services_4']", "['sgd_hotels_3']", "['sgd_music_1']", "['sgd_flights_1']",
                         "['sgd_hotels_1']", "['sgd_hotels_4']", "['sgd_media_2']", "['sgd_flights_3']",
                         "['sgd_trains_1']", "['sgd_homes_1']", "['sgd_restaurants_1']", "['sgd_rentalcars_2']",
                         "['sgd_services_3']", "['sgd_homes_2']", "['sgd_rentalcars_3']"]
    elif hparams.dataset_order == 4:
        dataset_order = ["['sgd_hotels_1']", "['sgd_media_2']", "['sgd_homes_1']", "['sgd_music_1']",
                         "['sgd_services_4']", "['sgd_restaurants_1']", "['sgd_flights_1']", "['sgd_hotels_4']",
                         "['sgd_services_3']", "['sgd_homes_2']", "['sgd_hotels_3']", "['sgd_trains_1']",
                         "['sgd_flights_3']", "['sgd_rentalcars_2']", "['sgd_rentalcars_3']"]
    elif hparams.dataset_order == 5:
        dataset_order = ["['sgd_services_4']", "['sgd_flights_3']", "['sgd_homes_1']", "['sgd_flights_1']",
                         "['sgd_music_1']", "['sgd_services_3']", "['sgd_rentalcars_3']", "['sgd_media_2']",
                         "['sgd_restaurants_1']", "['sgd_hotels_1']", "['sgd_rentalcars_2']", "['sgd_hotels_4']",
                         "['sgd_hotels_3']", "['sgd_homes_2']", "['sgd_trains_1']"]
    elif hparams.dataset_order == 6:
        dataset_order = ["['sgd_restaurants_1']", "['sgd_services_3']", "['sgd_flights_1']", "['sgd_trains_1']",
                         "['sgd_hotels_1']", "['sgd_services_4']", "['sgd_hotels_3']", "['sgd_rentalcars_2']",
                         "['sgd_flights_3']", "['sgd_hotels_4']", "['sgd_homes_2']", "['sgd_homes_1']",
                         "['sgd_rentalcars_3']", "['sgd_media_2']", "['sgd_music_1']"]

    elif hparams.dataset_order == 99:
        # debug
        dataset_order = ["['sgd_hotels_4']", "['sgd_trains_1']"]

    else:
        if hparams.dataset_order >= 100 and hparams.dataset_order <= 114:
            dataset_order = ["['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                             "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                             "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                             "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
            dataset_order = [dataset_order[hparams.dataset_order-100]]
        else:
            raise
    print('dataset order')
    print(dataset_order)
    print(f"RUNNING WITH SEED {hparams.seed}")

    if hparams.CL in ['VANILLA', 'REPLAY', 'EWC', 'VANILLA_BASELINE']:
        if "dialogpt" in hparams.model_checkpoint:
            collate_fn_ = collate_fn_DialoGPT
        elif "gpt2" in hparams.model_checkpoint:
            collate_fn_ = collate_fn_GPT2
        elif 't5' in hparams.model_checkpoint:
            collate_fn_ = collate_fn_t5
        else:
            raise

        temp_list = []
        test_dataset = {k: [] for k in dataset_order}
        for dial in test_datasets:
            if dial['task_id'] in dataset_order:
                test_dataset[dial['task_id']].append(dial)
        for task_id, dataset_task in test_dataset.items():
            if (task_id in dataset_order):
                temp_list.append(dataset_task)
        test_datasets = sum(temp_list, [])
        all_data_test_loaders = DataLoader(DatasetTrain(sum(temp_list, [])), batch_size=hparams.test_batch_size,
                                           shuffle=False,
                                           collate_fn=partial(collate_fn_, tokenizer=model.tokenizer))

    if (hparams.CL not in ['MULTI']):
        train_loader = {key: train_loader[key] for key in dataset_order}


    if hparams.do_train:
        task_seen_so_far = []
        if hparams.CL not in ["MULTI"]:
            model.set_number_of_tasks(len(list(train_loader.keys())))
        if hparams.CL == "GEM":
            model.set_up_gem()



        if hparams.CL == 'MULTI':
            start = time.time()
            trainer = Trainer(
                default_root_dir=hparams.saving_dir,
                accumulate_grad_batches=hparams.gradient_accumulation_steps,
                gradient_clip_val=hparams.max_norm,
                max_epochs=hparams.n_epochs,
                callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=False,
                                                      mode='min')],
                gpus=[0],
            )
            trainer.fit(model, train_loader, val_loader)
            end = time.time()
            print("Time elapsed:", end - start)
            model.model.save_pretrained(f'{hparams.saving_dir}')
            model.tokenizer.save_pretrained(f'{hparams.saving_dir}')
            test_model_seq2seq(hparams, model.model, model.tokenizer, test_loader, time=f"FINAL")
            score_folder(hparams.saving_dir, hparams)



        elif hparams.continual:
            memsize_for_task = [len(train_datasets[task_id]) for task_id in dataset_order]
            if hparams.no_memory_uniform_dist:
                total_dials = sum(memsize_for_task)
                memsize_for_task = [int(_ / total_dials * len(dataset_order) * hparams.episodic_mem_size) for _ in
                                    memsize_for_task]
            else:
                memsize_for_task = [min(_, hparams.episodic_mem_size) for _ in memsize_for_task]

            # for task_num, (task_id, task_loader) in enumerate(train_loader.items()):
            for task_num, task_id in enumerate(dataset_order):
                task_loader = train_loader[task_id]
                model.task_list_seen.append(task_id)

                if hparams.CL == 'VANILLA_BASELINE':
                    model.reload_model()

                if hparams.CL == 'ADAPTER' and hparams.clinit:
                    now_task_no = model.task_list_seen.index(task_id)
                    if now_task_no > 0:
                        print('initialize {} using {} adapter'.format(model.task_list_seen[now_task_no],
                                                                      model.task_list_seen[now_task_no - 1]))
                        model.model.clinit_adapter(model.task_list_seen.index(task_id))

                if (hparams.CL == "REPLAY"):
                    print(f"Memory Size {len(model.reply_memory)}")
                    task_loader = make_loader(hparams, train_datasets[task_id] + model.reply_memory, model.tokenizer)

                print()
                print(f"TASK:{task_id}")
                max_epochs = hparams.n_epochs
                if max_epochs > 0:
                    seed_everything(hparams.seed)
                    start = time.time()
                    trainer = Trainer(
                        default_root_dir=f'{hparams.saving_dir}/{task_num}_{task_id}',
                        accumulate_grad_batches=hparams.gradient_accumulation_steps,
                        gradient_clip_val=hparams.max_norm,
                        max_epochs=max_epochs,
                        callbacks=[
                            pl.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=True,
                                                       mode='min')] if hparams.CL not in ['PROMPT', 'META_PROMPT',
                                                                                          'MLM_PROMPT',
                                                                                          'CL_INIT_PROMPT',
                                                                                          'MEM_META_PROMPT',
                                                                                          'MULTITASK_INIT_PROMPT',
                                                                                          ] else
                        [pl.callbacks.EarlyStopping(monitor='jga', patience=5, verbose=True, mode='max'),
                         pl.callbacks.ModelCheckpoint(filename='{jga:.4f}-{epoch}', monitor='jga', mode='max',
                                                      save_top_k=1)],
                        reload_dataloaders_every_epoch=True if hparams.CL in ['PROMPT', 'CL_INIT_PROMPT',
                                                                              'META_PROMPT',
                                                                              'MEM_META_PROMPT'] else False,
                        # for shuffle descriptions
                        gpus=[0],
                        # limit_train_batches=10,  # for fast debug
                        precision=16,
                        num_sanity_val_steps=-1
                    )

                    trainer.fit(model, task_loader, val_loader[task_id])

                    end = time.time()
                    print("Time elapsed:", end - start)
                    best_model_path = trainer.checkpoint_callback.best_model_path
                    trainer.optimizers = None  # IMPORTANT! release GPU memory
                    del trainer

                    # load best model
                    # this model are better if the are runned to they epoch number
                    # if (hparams.CL != "LAMOL" and hparams.CL != "EWC"):
                    if (hparams.CL != "LAMOL"):
                        checkpoint = torch.load(best_model_path)
                        print("load from:", best_model_path)
                        checkpoint['state_dict'] = {k.replace('model.', ''): v for k, v in
                                                    checkpoint['state_dict'].items()}
                        model.model.load_state_dict(checkpoint['state_dict'])
                        del checkpoint

                # testing the model by generating the answers
                if (hparams.test_every_step):
                    if (hparams.CL == "ADAPTER"):
                        print('test on current domain validation set')
                        if hparams.model_type == 'gpt2':
                            test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, {task_id: val_loader[task_id]},
                                                       test_datasets,
                                                       time=f"{task_num}_{task_id}")
                            score_folder(f'{hparams.saving_dir}', hparams, time=f"{task_num}_{task_id}")
                        elif hparams.model_type == 't5':
                            test_t5_adapter(model, {task_id: val_loader[task_id]}, hparams,
                                            time=f"{task_num}_{task_id}")
                            score_folder(f'{hparams.saving_dir}/{task_num}_{task_id}', hparams, time='.')

                    elif hparams.CL == 'VANILLA_BASELINE':
                        print('test vanilla baseline on domain data')
                        pred_test_loader = test_loader[task_id]
                        test_model_seq2seq(hparams, model.model, model.tokenizer, pred_test_loader, time=f"{task_num}_{task_id}")
                        score_folder(f'{hparams.saving_dir}', hparams, time=f"{task_num}_{task_id}")
                    else:
                        print('test on all domain test data')
                        test_model_seq2seq(hparams, model.model, model.tokenizer, all_data_test_loaders,
                                           time=f"{task_num}_{task_id}")
                        score_folder(f'{hparams.saving_dir}', hparams, time=f"{task_num}_{task_id}")



                model.first_task = False
                ## save some training data into the episodic mem
                if hparams.CL == "AGEM":
                    for idx_b, b in enumerate(task_loader):
                        model.episodic_mem["all"].append(b)
                        if idx_b == hparams.episodic_mem_size: break
                elif hparams.CL == "REPLAY":
                    # in percentage
                    model.reply_memory += sample(train_datasets[task_id], memsize_for_task[task_num])  # sample(train_datasets[task_id],min(len(train_datasets[task_id]),int(hparams.episodic_mem_size*len(train_datasets[task_id])))


                else:  ## save example per task
                    for idx_b, b in enumerate(task_loader):
                        model.episodic_mem[task_id].append(b)
                        if idx_b == hparams.episodic_mem_size: break

                ##### Compute Fisher info Matrix for EWC
                if hparams.CL == "EWC" or hparams.CL == "L2":
                    model.model.cpu()
                    for n, p in model.model.named_parameters():
                        model.optpar[n] = torch.Tensor(p.cpu().data)
                        model.fisher[n] = torch.zeros(p.size())  # torch.Tensor(p.cpu().data).zero_()

                    if hparams.CL == "EWC":
                        for _, batch in enumerate(model.episodic_mem[task_id]):
                            model.model.zero_grad()
                            ret_tuple = model.model(input_ids=batch["encoder_input"],
                                                    attention_mask=batch["attention_mask"],
                                                    labels=batch["decoder_output"])
                            loss = ret_tuple[0]
                            loss.backward()
                            for n, p in model.model.named_parameters():
                                if p.grad is not None:
                                    model.fisher[n].data += p.grad.data ** 2

                        for name_f, _ in model.fisher.items():
                            model.fisher[name_f] /= len(model.episodic_mem[task_id])  # *hparams.train_batch_size
                        model.model.zero_grad()


                model.num_step = 0
                task_seen_so_far.append(task_id)



            if not hparams.CL == 'VANILLA_BASELINE':
                model.model.save_pretrained(f'{hparams.saving_dir}')
                model.tokenizer.save_pretrained(f'{hparams.saving_dir}')

                print('final test on all domains')
                if (hparams.CL == "ADAPTER"):
                    if hparams.model_type == 'gpt2':
                        model.task_list_seen = dataset_order
                        pred_test_loader = {k: v for k, v in test_loader.items() if k in dataset_order}
                        test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, pred_test_loader, test_datasets,
                                                   time=f"FINAL")
                        score_folder(hparams.saving_dir, hparams)
                    elif hparams.model_type == 't5':
                        pred_test_dataset = {k: [] for k in dataset_order}
                        for dial in test_datasets:
                            if dial['task_id'] in pred_test_dataset:
                                pred_test_dataset[dial['task_id']].append(dial)
                        test_loaders = {}
                        for task_id, dataset_task in pred_test_dataset.items():
                            task_id = str(list(eval(task_id)))
                            test_loaders[task_id] = make_test_loader(hparams, dataset_task, model.tokenizer)
                        test_loaders = {k: v for k, v in test_loaders.items() if k in dataset_order}
                        test_t5_adapter(model, test_loaders, hparams, time="FINAL")
                        score_folder(hparams.saving_dir, hparams)


                else:
                    test_model_seq2seq(hparams, model.model, model.tokenizer, all_data_test_loaders, time=f"FINAL")
                    score_folder(hparams.saving_dir, hparams)

    if hparams.do_eval:
        # hparams.saving_dir = os.path.join(hparams.saving_dir, 'best_tfmr')
        model.model.load_state_dict(torch.load(os.path.join(hparams.saving_dir, 'pytorch_model.bin')))
        if (hparams.CL == "ADAPTER"):
            if hparams.model_type == 'gpt2':
                model.task_list_seen = dataset_order
                pred_test_loader = {k: v for k, v in test_loader.items() if k in dataset_order}
                test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, pred_test_loader, test_datasets,
                                           time=f"FINAL")
                score_folder(hparams.saving_dir, hparams)
            elif hparams.model_type == 't5':
                model.task_list_seen = dataset_order
                pred_test_dataset = {k: [] for k in dataset_order}
                for dial in test_datasets:
                    if dial['task_id'] in pred_test_dataset:
                        pred_test_dataset[dial['task_id']].append(dial)
                test_loaders = {}
                for task_id, dataset_task in pred_test_dataset.items():
                    task_id = str(list(eval(task_id)))
                    test_loaders[task_id] = make_test_loader(hparams, dataset_task, model.tokenizer)
                test_loaders = {k: v for k, v in test_loaders.items() if k in dataset_order}
                test_t5_adapter(model, test_loaders, hparams, time="FINAL")
                score_folder(hparams.saving_dir, hparams)

        else:
            test_model_seq2seq(hparams, model.model, model.tokenizer, all_data_test_loaders, time=f"FINAL")
            score_folder(hparams.saving_dir, hparams)


def prepare_todcl_params():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, default="gpt2")
    parser.add_argument('--model_checkpoint', type=str, default="gpt2")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--dataset_list", type=str, default="SGD,TM19,TM20,MWOZ", help="Path for saving")
    parser.add_argument("--max_history", type=int, default=5, help="max number of turns in the dialogue")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--setting", type=str, default="single", help="Path for saving")
    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument("--test_every_step", action='store_true', help="continual baseline")
    parser.add_argument("--debug", action='store_true', help="continual baseline")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")

    parser.add_argument("--bottleneck_size", type=int, default=100)
    parser.add_argument("--number_of_adpt", type=int, default=40, help="number of adapterss")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--percentage_LAM0L", type=float, default=0.2, help="LAMOL percentage of augmented data used")
    parser.add_argument("--reg", type=float, default=0.01, help="CL regularization term")
    parser.add_argument("--episodic_mem_size", type=int, default=50,
                        help="number of batch/sample put in the episodic memory")
    #  options=["E2E","DST","NLG","INTENT"]
    parser.add_argument('--task_type', type=str, default="NLG")
    #  options=["VANILLA"]
    parser.add_argument('--CL', type=str, default="MULTI")
    # options=[1,2,3,4,5]
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--samples_per_domain', default=-1, type=int, help="restrict samples per domain for fast run")
    parser.add_argument('--pardir', type=str, default="test")
    parser.add_argument('--saving_dir', type=str)
    parser.add_argument("--do_train", action='store_true', help="train and evaluation")
    parser.add_argument("--do_eval", action='store_true', help="evaluation")
    parser.add_argument("--use_cache", action='store_true', help="use cached dataloader")
    parser.add_argument('--test_file_path', default='test_dials.json')

    # for combatibility
    parser.add_argument('--inclusive_domains', default=None)
    parser.add_argument('--max_train_dials_per_domain', type=int, help='limit dials for each domain', default=None)
    parser.add_argument('--choose_teacher_meta_prompt', action='store_true')

    parser.add_argument('--clinit', action='store_true')

    parser.add_argument('--dataset_order', type=int)
    parser.add_argument('--no_memory_uniform_dist', action='store_true')


    args = parser.parse_args()
    if 't5' in args.model_checkpoint:
        args.model_type = 't5'
    return args





if __name__ == '__main__':
    parser = ArgumentParser()
    # hyperparams = prepare_prompt_params(parser)
    hyperparams = prepare_todcl_params()
    print(hyperparams)
    main(hyperparams)
