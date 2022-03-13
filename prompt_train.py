import random

import torch
import json
import os
import os.path
import shutil
import time
from random import sample
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

from prompt_files.prompt_dataset import MemT5PromptDSTDataset
from utils.dataloader import get_data_loaders

from collections import defaultdict, OrderedDict
from argparse import ArgumentParser
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
from pprint import pprint

from prompt_files.prompt_module import MemT5PromptDSTModule, prepare_MemT5PromptDST_parser
from prompt_files.prompt_test import prompt5_predict, prompt5_predict_fwt, prompt5_predict_select_fwt
import torch.multiprocessing

DEVICE = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'

torch.multiprocessing.set_sharing_strategy('file_system')


def main(hparams):
    seed_everything(hparams.seed)
    assert hparams.CL in ['MULTI_PROMPT', 'RAND_PROMPT', 'FW_PROMPT', 'FWBW_PROMPT']
    hparams.multi = hparams.CL == 'MULTI_PROMPT'
    hparams.continual = not hparams.multi
    model = MemT5PromptDSTModule(hparams)
    model.to(DEVICE)
    train_loader, val_loader, test_loader, (train_datasets, val_datasets, test_datasets) = get_data_loaders(hparams,
                                                                                                            model.tokenizer,
                                                                                                            inclusive_domains=hparams.inclusive_domains,
                                                                                                            max_train_dials_per_domain=hparams.max_train_dials_per_domain)
    # split test dataset to domain
    test_dataset = defaultdict(list)
    for dial in test_datasets:
        test_dataset[dial['task_id']].append(dial)
    test_datasets = test_dataset


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

    elif hparams.dataset_order == 30:
        dataset_order = ["['sgd_events_3']", "['sgd_banks_2']", "['sgd_banks_1']", "['sgd_calendar_1']",
                         "['sgd_movies_3']", "['sgd_music_2']", "['sgd_services_2']", "['sgd_payment_1']",
                         "['sgd_media_1']", "['sgd_weather_1']", "['sgd_events_1']", "['sgd_flights_4']",
                         "['sgd_travel_1']", "['sgd_buses_2']", "['sgd_events_2']", "['sgd_alarm_1']",
                         "['sgd_buses_3']", "['sgd_services_1']", "['sgd_buses_1']", "['sgd_restaurants_2']",
                         "['sgd_hotels_2']", "['sgd_ridesharing_2']", "['sgd_rentalcars_1']", "['sgd_movies_1']",
                         "['sgd_ridesharing_1']", "['sgd_media_3']", "['sgd_music_3']", "['sgd_movies_2']",
                         "['sgd_flights_2']", "['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                         "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                         "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
        dataset_order = dataset_order[-5:]
    elif hparams.dataset_order == 31:
        dataset_order = ["['sgd_events_3']", "['sgd_banks_2']", "['sgd_banks_1']", "['sgd_calendar_1']",
                         "['sgd_movies_3']", "['sgd_music_2']", "['sgd_services_2']", "['sgd_payment_1']",
                         "['sgd_media_1']", "['sgd_weather_1']", "['sgd_events_1']", "['sgd_flights_4']",
                         "['sgd_travel_1']", "['sgd_buses_2']", "['sgd_events_2']", "['sgd_alarm_1']",
                         "['sgd_buses_3']", "['sgd_services_1']", "['sgd_buses_1']", "['sgd_restaurants_2']",
                         "['sgd_hotels_2']", "['sgd_ridesharing_2']", "['sgd_rentalcars_1']", "['sgd_movies_1']",
                         "['sgd_ridesharing_1']", "['sgd_media_3']", "['sgd_music_3']", "['sgd_movies_2']",
                         "['sgd_flights_2']", "['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                         "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                         "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
        dataset_order = dataset_order[-30:]
    elif hparams.dataset_order == 32:
        dataset_order = ["['sgd_events_3']", "['sgd_banks_2']", "['sgd_banks_1']", "['sgd_calendar_1']",
                         "['sgd_movies_3']", "['sgd_music_2']", "['sgd_services_2']", "['sgd_payment_1']",
                         "['sgd_media_1']", "['sgd_weather_1']", "['sgd_events_1']", "['sgd_flights_4']",
                         "['sgd_travel_1']", "['sgd_buses_2']", "['sgd_events_2']", "['sgd_alarm_1']",
                         "['sgd_buses_3']", "['sgd_services_1']", "['sgd_buses_1']", "['sgd_restaurants_2']",
                         "['sgd_hotels_2']", "['sgd_ridesharing_2']", "['sgd_rentalcars_1']", "['sgd_movies_1']",
                         "['sgd_ridesharing_1']", "['sgd_media_3']", "['sgd_music_3']", "['sgd_movies_2']",
                         "['sgd_flights_2']", "['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                         "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                         "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
        dataset_order = dataset_order
    else:
        raise

    # prepare domain2slot schema and data for selected domain in dataset_order
    domain2slot = defaultdict(set)
    for datasets in [train_datasets, val_datasets, test_datasets]:
        for domain, dials in datasets.items():
            for dial in dials:
                for slot in dial['state']:
                    domain2slot[domain].add(slot)
    domain2slot = {k: list(sorted(list(v))) for k, v in domain2slot.items()}
    train_datasets = {d: v for d, v in train_datasets.items() if d in dataset_order}
    val_datasets = {d: v for d, v in val_datasets.items() if d in dataset_order}
    test_datasets = {d: v for d, v in test_datasets.items() if d in dataset_order}

    train_dataset = model.dataset_class(tokenizer=model.tokenizer,
                                        type_path='train',
                                        dialogs=train_datasets,
                                        domain2slot=domain2slot,
                                        num_domain_prompt=hparams.num_domain_prompt,
                                        small_sample_run=hparams.small_sample_run,
                                        permute_desc=hparams.permute_desc,
                                        multitask=hparams.multi)
    val_dataset = model.dataset_class(tokenizer=model.tokenizer,
                                      type_path='val',
                                      dialogs=val_datasets,
                                      domain2slot=domain2slot,
                                      num_domain_prompt=hparams.num_domain_prompt,
                                      small_sample_run=hparams.small_sample_run,
                                      permute_desc=hparams.permute_desc,
                                      multitask=hparams.multi)
    model.set_dataset(train_dataset, val_dataset)

    # save datasets here
    json.dump(test_datasets, open('test_dials.json', 'w'), indent=4)

    # get some statistics
    # statistics_csv_list = []
    # first_line = ['Services', '# Slots', '# Dialogs', '', '',
    #              '# Samples', '', '',
    #              'Avg. Sample Length', 'Avg. Desc Length', 'Avg. Input Length']
    # statistics_csv_list.append(first_line)
    #
    # sta_items = ['', '', 'Train', 'Dev', 'Test',
    #              'Train', 'Dev', 'Test',
    #              '', '', '']
    # statistics_csv_list.append(sta_items)
    # for domain in dataset_order:
    #     domain_str = eval(domain)[0]
    #     domain_str = domain_str.replace('sgd_', '')
    #     domain_str = domain_str.replace('_', '$_{') + '}$'
    #     desc_str = ''
    #     slots = domain2slot[domain]
    #     for i, slot in enumerate(slots):
    #         enc_p = train_dataset.slot2description[slot]
    #         desc_str += enc_p.replace('<extra_id_0>', '<extra_id_{}>'.format(i))
    #     encoded = train_dataset.tokenizer.tokenize(desc_str)
    #     len_desc = len(encoded)
    #
    #     dial_lens = []
    #     dial_num_list = []
    #     sample_num_list = []
    #
    #     for split in ['train', 'val', 'test']:
    #         dial_ids = set()
    #         samples = eval('{}_datasets'.format(split))[domain]
    #         for sample in samples:
    #             encoded = train_dataset.tokenizer.tokenize(sample['history'])
    #             dial_length = len(encoded)
    #             dial_lens.append(dial_length)
    #             dial_ids.add(sample['dial_id'])
    #         dial_num_list.append(len(dial_ids))
    #         sample_num_list.append(len(samples))
    #
    #     dial_num_list = [str(_) for _ in dial_num_list]
    #
    #     sample_num_list = [str(_) for _ in sample_num_list]
    #     sta_for_domain_split = [domain_str, len(slots), *dial_num_list, *sample_num_list,
    #                             sum(dial_lens)//len(dial_lens), len_desc,
    #                             sum(dial_lens)//len(dial_lens)+len_desc+101]
    #     statistics_csv_list.append(sta_for_domain_split)
    # import csv
    # with open('data_statistics.csv', 'w') as csvfile:
    #     csv_w = csv.writer(csvfile)
    #     for line_w in statistics_csv_list:
    #         csv_w.writerow(line_w)
    # exit()

    seed_everything(hparams.seed)
    if hparams.do_train:
        if hparams.multi:
            # use <prompt_x> to init <meta_prompt_x>, x range from 0 to 99
            model.initialize_metaprompt_by_trained_prompt({i: i for i in range(hparams.num_domain_prompt)})
            model.training_prompt_name = 'meta_prompt'
            val_dataloader = model.prepare_val_dataloader()
            start = time.time()
            trainer = Trainer(
                default_root_dir=hparams.saving_dir,
                accumulate_grad_batches=hparams.gradient_accumulation_steps,
                gradient_clip_val=hparams.max_norm,
                max_epochs=hparams.n_epochs,
                callbacks=[pl.callbacks.EarlyStopping(monitor='jga', patience=5, verbose=True, mode='max'),
                           pl.callbacks.ModelCheckpoint(filename='{jga:.4f}-{epoch}', monitor='jga', mode='max',
                                                        save_top_k=1)],
                reload_dataloaders_every_epoch=True,
                gpus=[0],
                precision=16,
                num_sanity_val_steps=4
            )
            trainer.fit(model, val_dataloaders=val_dataloader)
            end = time.time()
            print("Time elapsed:", end - start)
            model.model.save_pretrained(f'{hparams.saving_dir}')
            model.tokenizer.save_pretrained(f'{hparams.saving_dir}')

        elif hparams.continual:

            meta_prompt_test_res = OrderedDict({task_id: OrderedDict() for task_id in dataset_order})
            backward_test_res = OrderedDict({task_id: OrderedDict() for task_id in dataset_order})
            meta_prompt_jga = OrderedDict()

            memsize_for_task = [len(model.dataset['train'].dialogs[task_id]) for task_id in dataset_order]
            if hparams.no_memory_uniform_dist:
                total_dials = sum(memsize_for_task)
                memsize_for_task = [int(_ / total_dials * len(dataset_order) * hparams.episodic_mem_size) for _ in memsize_for_task]
            else:
                memsize_for_task = [min(_, hparams.episodic_mem_size) for _ in memsize_for_task]

            for task_num, task_id in enumerate(dataset_order):
                model.task_list_seen.append(task_id)
                model.cur_domain = task_id
                if task_num == 0 and hparams.CL != 'RAND_PROMPT':
                    model.training_prompt_name = 'first_prompt'
                else:
                    model.training_prompt_name = 'prompt'

                # training prompt
                print(f"TASK:{task_id}")
                max_epochs = hparams.first_epochs if model.training_prompt_name == 'first_prompt' else hparams.n_epochs
                val_dataloader = model.prepare_val_dataloader()
                seed_everything(hparams.seed)
                start = time.time()
                trainer = Trainer(
                    default_root_dir=f'{hparams.saving_dir}/{task_num}_{task_id}',
                    accumulate_grad_batches=hparams.gradient_accumulation_steps,
                    gradient_clip_val=hparams.max_norm,
                    max_epochs=max_epochs,
                    callbacks=[pl.callbacks.EarlyStopping(monitor='jga', patience=5, verbose=True, mode='max'),
                               pl.callbacks.ModelCheckpoint(filename='{jga:.4f}-{epoch}', monitor='jga', mode='max',
                                                            save_top_k=1)],
                    reload_dataloaders_every_epoch=True,
                    gpus=[0],
                    precision=16,
                    num_sanity_val_steps=-1
                )
                trainer.fit(model, val_dataloaders=val_dataloader)
                end = time.time()
                print("Time elapsed:", end - start)
                best_model_path = trainer.checkpoint_callback.best_model_path
                trainer.optimizers = None  # IMPORTANT! release GPU memory
                del trainer

                # load best model since there is early stopping
                checkpoint = torch.load(best_model_path)
                print("load from:", best_model_path)
                checkpoint['state_dict'] = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
                model.model.load_state_dict(checkpoint['state_dict'])
                del checkpoint

                # testing the model by generating the answers
                if hparams.test_every_step:
                    test_dataset = {model.cur_domain: test_datasets[model.cur_domain]}
                    test_dataset = model.dataset_class(
                        tokenizer=model.tokenizer,
                        type_path='test',
                        dialogs=test_dataset,
                        domain2slot=domain2slot,
                        num_domain_prompt=hparams.num_domain_prompt,
                        small_sample_run=hparams.small_sample_run)
                    pred_metrics = prompt5_predict(model.model, test_dataset, model.tokenizer, hparams,
                                                   [model.cur_domain])
                    # print('=============final joint acc on test===================')
                    print(pred_metrics, file=open(os.path.join(hparams.saving_dir, 'test_res.txt'), 'a'))

                # delete models at this step to save space
                checkpoints = os.path.dirname(best_model_path)
                print("rm checkpoints:", checkpoints)
                shutil.rmtree(checkpoints)

                # save some training data into the episodic memory
                if hparams.CL in ['FW_PROMPT', 'FWBW_PROMPT'] and task_num < len(dataset_order) - 1:
                    # TODO: memory selection strategy
                    cur_domain_train_dialogs = model.dataset['train'].dialogs[model.cur_domain]
                    seed_everything(hparams.seed)
                    model.dataset['train'].replay_memory[model.cur_domain] = sample(cur_domain_train_dialogs,
                                                                                    memsize_for_task[task_num])
                    print('memsize for {}: {}'.format(task_id, memsize_for_task[task_num]))
                    # DONE: evaluate domain prompt on current domain memory
                    # log down test jga, memory loss and jga
                    if hparams.test_on_memory:
                        assert hparams.test_every_step
                        for test_domain in [model.cur_domain]:
                            print('eval memory of', test_domain)
                            model.zero_grad()
                            model.eval()
                            dials = model.dataset['train'].replay_memory[test_domain]
                            num_dialog_per_batch = hparams.train_batch_size
                            mem_loss, mem_jga = 0, 0
                            for s in tqdm(range(0, len(dials), num_dialog_per_batch), desc=str(test_domain)):
                                test_batch = []
                                batch_dials = dials[s: s + num_dialog_per_batch]
                                for dial in batch_dials:
                                    test_batch.append(test_dataset.prepare_for_generation(dial, test_domain))
                                batch = test_dataset.collate_fn(test_batch)
                                batch = {k: v.to(model.model.device) if isinstance(v, torch.Tensor) else v
                                         for k, v in batch.items()}
                                metric = model.validation_step(batch, batch_idx=s)
                                mem_loss += metric['loss'].item() * len(batch_dials)
                                mem_jga += metric['acc']
                                del batch
                            mem_loss /= len(dials)
                            mem_jga /= len(dials)
                            meta_prompt_test_res[test_domain][model.cur_domain] = {
                                'test_jga': pred_metrics[test_domain],
                                'mem_loss': mem_loss, 'mem_jga': mem_jga}
                            backward_test_res[test_domain][model.cur_domain] = {'test_jga': pred_metrics[test_domain],
                                                                                'mem_loss': mem_loss,
                                                                                'mem_jga': mem_jga}

                    # for n-1 domains, train meta-prompt in next domain
                    model.training_prompt_name = 'meta_prompt'
                    model.forward_domains = model.task_list_seen
                    # TODO: meta prompt init strategy
                    if hparams.select_init:
                        next_domain_id = dataset_order[task_num + 1]
                        print('trying to choose best prompt initialization for {} domain'.format(next_domain_id))
                        assert model.model.training
                        model.zero_grad()
                        model.eval()

                        min_loss_domain_id = None
                        min_loss = float('inf')
                        for prompt_domain in dataset_order[:task_num + 1]:
                            domain_val_loss_given_prompt = 0
                            print('=' * 50)
                            print('calculating val loss using {} domain prompt'.format(prompt_domain))
                            dials = val_dataset.dialogs[next_domain_id]

                            num_dialog_per_batch = hparams.eval_batch_size // 2
                            assert num_dialog_per_batch > 0

                            # prepare batches
                            if hparams.small_sample_run:
                                dials = dials[:10]

                            for s in tqdm(range(0, len(dials), num_dialog_per_batch), desc=str(next_domain_id)):
                                test_batch = []
                                batch_dials = dials[s: s + num_dialog_per_batch]
                                for dial in batch_dials:
                                    test_batch.append(
                                        val_dataset.convert_dial_to_example(
                                            dial,
                                            next_domain_id,
                                            val_dataset.domain2soft_prompt(prompt_domain)))
                                batch = val_dataset.collate_fn(test_batch)
                                batch = {k: v.to(model.model.device) if isinstance(v, torch.Tensor) else v
                                         for k, v in batch.items()}
                                domain_val_loss_given_prompt += model.validation_step(batch, batch_idx=s, only_loss=True)[
                                    'loss'].item()
                                del batch
                            print('val_loss = {}'.format(domain_val_loss_given_prompt))
                            if domain_val_loss_given_prompt < min_loss:
                                min_loss_domain_id = prompt_domain
                                min_loss = domain_val_loss_given_prompt
                        print(
                            'choose prompt initialization, decide to use {} domain prompt as initialization'.format(
                                min_loss_domain_id))
                        model.initialize_metaprompt_by_trained_prompt(
                            model.dataset['train'].get_prompt_init_dict(from_cl_domain=min_loss_domain_id))
                        model.train()
                    else:
                        model.initialize_metaprompt_by_trained_prompt(
                            model.dataset['train'].get_prompt_init_dict(from_cl_domain=model.cur_domain))
                    model.cur_domain = dataset_order[task_num + 1]

                    # TEST meta-prompt in all seen domains
                    test_dataset = OrderedDict({d: test_datasets[d] for d in dataset_order[:task_num + 2]})

                    test_dataset = model.dataset_class(
                        tokenizer=model.tokenizer,
                        type_path='test',
                        dialogs=test_dataset,
                        domain2slot=domain2slot,
                        num_domain_prompt=hparams.num_domain_prompt,
                        small_sample_run=hparams.small_sample_run,
                        multitask=True)

                    # print("==================meta prompt test before training==================", model.cur_domain)
                    # pred_metrics = prompt5_predict(model.model, test_dataset, model.tokenizer, hparams, [model.cur_domain])
                    # print(pred_metrics, file=open(os.path.join(hparams.saving_dir, 'test_res_meta_before.txt'), 'a'))

                    print('training meta_prompts in {} domain'.format(model.cur_domain))
                    seed_everything(hparams.seed)
                    val_dataloader = model.prepare_val_dataloader()
                    trainer = Trainer(
                        default_root_dir=f'{hparams.saving_dir}/{task_num + 1}_{model.cur_domain}_meta',
                        accumulate_grad_batches=hparams.gradient_accumulation_steps,
                        gradient_clip_val=hparams.max_norm,
                        max_epochs=hparams.meta_n_epochs,
                        callbacks=[pl.callbacks.EarlyStopping(monitor='jga', patience=5, verbose=True, mode='max'),
                                   pl.callbacks.ModelCheckpoint(filename='{jga:.4f}-{epoch}', monitor='jga', mode='max',
                                                                save_top_k=1)],
                        reload_dataloaders_every_epoch=True,
                        gpus=[0],
                        precision=16,
                        num_sanity_val_steps=-1
                    )
                    trainer.fit(model, val_dataloaders=val_dataloader)
                    best_model_path = trainer.checkpoint_callback.best_model_path
                    trainer.optimizers = None  # IMPORTANT! release GPU memory
                    del trainer

                    # load best model since there is early stopping
                    checkpoint = torch.load(best_model_path)
                    print("load from:", best_model_path)
                    checkpoint['state_dict'] = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
                    model.model.load_state_dict(checkpoint['state_dict'])
                    del checkpoint

                    # init cur domain prompt by meta prompt
                    model.initialize_prompt_by_trained_metaprompt(
                        model.dataset['train'].get_prompt_init_dict(to_cl_domain=dataset_order[task_num + 1]))

                    # print("==================meta prompt test after training==================", model.cur_domain)
                    # pred_metrics = prompt5_predict(model.model, test_dataset, model.tokenizer, hparams,
                    #                                dataset_order[:task_num + 2])
                    # print(pred_metrics, file=open(os.path.join(hparams.saving_dir, 'test_res_meta_after.txt'), 'a'))

                    # delete models at this step to save space
                    checkpoints = os.path.dirname(best_model_path)
                    print("rm checkpoints:", checkpoints)
                    shutil.rmtree(checkpoints)

                    # DONE: evaluate meta prompt on previous domain memory
                    # log down test jga, memory loss and jga
                    if hparams.episodic_mem_size > 0 and hparams.test_on_memory:
                        assert hparams.test_every_step
                        for test_domain in model.task_list_seen:
                            print('eval memory of', test_domain)
                            model.zero_grad()
                            model.eval()
                            dials = model.dataset['train'].replay_memory[test_domain]
                            num_dialog_per_batch = hparams.train_batch_size
                            mem_loss, mem_jga = 0, 0
                            for s in tqdm(range(0, len(dials), num_dialog_per_batch), desc=str(test_domain)):
                                test_batch = []
                                batch_dials = dials[s: s + num_dialog_per_batch]
                                for dial in batch_dials:
                                    test_batch.append(test_dataset.prepare_for_generation(dial, test_domain))
                                batch = test_dataset.collate_fn(test_batch)
                                batch = {k: v.to(model.model.device) if isinstance(v, torch.Tensor) else v
                                         for k, v in batch.items()}
                                metric = model.validation_step(batch, batch_idx=s)
                                mem_loss += metric['loss'].item() * len(batch_dials)
                                mem_jga += metric['acc']
                                del batch
                            mem_loss /= len(dials)
                            mem_jga /= len(dials)
                            meta_prompt_test_res[test_domain][model.cur_domain] = {
                                'test_jga': pred_metrics[test_domain],
                                'mem_loss': mem_loss, 'mem_jga': mem_jga}
                        # print(meta_prompt_test_res)
                        table = []
                        for test_domain in model.task_list_seen:
                            min_mem_loss = meta_prompt_test_res[test_domain][test_domain]['mem_loss']
                            max_mem_jga = meta_prompt_test_res[test_domain][test_domain]['mem_jga']
                            test_jga = meta_prompt_test_res[test_domain][test_domain]['test_jga']
                            best_domain = test_domain
                            res = {'dom': test_domain, 'mem_loss': min_mem_loss, 'mem_jga': max_mem_jga,
                                   'test_jga': test_jga}
                            for train_domain in meta_prompt_test_res[test_domain]:
                                mem_loss = meta_prompt_test_res[test_domain][train_domain]['mem_loss']
                                mem_jga = meta_prompt_test_res[test_domain][train_domain]['mem_jga']
                                if mem_loss < min_mem_loss and mem_jga >= max_mem_jga:
                                    min_mem_loss = mem_loss
                                    max_mem_jga = mem_jga
                                    test_jga = meta_prompt_test_res[test_domain][train_domain]['test_jga']
                                    best_domain = train_domain
                            res.update(
                                {'best dom': best_domain, 'best mem_loss': min_mem_loss, 'best mem_jga': max_mem_jga,
                                 'best test_jga': test_jga})
                            meta_prompt_jga[test_domain] = test_jga
                            table.append(res)
                        print(tabulate(table, headers="keys"))
                        json.dump(meta_prompt_test_res,
                                  open(os.path.join(hparams.saving_dir, 'meta_prompt_test_res.json'), 'w'), indent=4)

            model.model.save_pretrained(f'{hparams.saving_dir}')
            model.tokenizer.save_pretrained(f'{hparams.saving_dir}')

            if hparams.episodic_mem_size > 0 and hparams.test_on_memory:
                json.dump(backward_test_res, open(os.path.join(hparams.saving_dir, 'backward_test_res.json'), 'w'),
                          indent=4)
                json.dump(meta_prompt_test_res,
                          open(os.path.join(hparams.saving_dir, 'meta_prompt_test_res.json'), 'w'),
                          indent=4)
                meta_prompt_jga['avg_joint_acc'] = sum(meta_prompt_jga.values()) / len(meta_prompt_jga)
                print('meta prompt jga')
                print(meta_prompt_jga)
                print(meta_prompt_jga, file=open(os.path.join(hparams.saving_dir, 'meta_prompt_mem.txt'), 'a'))

    if hparams.do_eval:
        model.model.load_state_dict(torch.load(os.path.join(hparams.saving_dir, 'pytorch_model.bin')))
        model.eval()
        test_dataset = OrderedDict({d: test_datasets[d] for d in dataset_order})
        test_dataset = model.dataset_class(
            tokenizer=model.tokenizer,
            type_path='test',
            dialogs=test_dataset,
            domain2slot=domain2slot,
            num_domain_prompt=hparams.num_domain_prompt,
            small_sample_run=hparams.small_sample_run,
            multitask=hparams.multi)
        pred_metrics = prompt5_predict(model.model, test_dataset, model.tokenizer, hparams)
        # print('=============final joint acc on test===================')
        print(pred_metrics, file=open(os.path.join(hparams.saving_dir, 'test_res.txt'), 'a'))

    if hparams.do_eval_fwt:
        model.model.load_state_dict(torch.load(os.path.join(hparams.saving_dir, 'pytorch_model.bin')))
        model.eval()
        test_dataset = OrderedDict({d: test_datasets[d] for d in dataset_order})
        test_dataset = model.dataset_class(
            tokenizer=model.tokenizer,
            type_path='test',
            dialogs=test_dataset,
            domain2slot=domain2slot,
            num_domain_prompt=hparams.num_domain_prompt,
            small_sample_run=hparams.small_sample_run,
            multitask=hparams.multi)
        pred_metrics = prompt5_predict_fwt(model.model, test_dataset, model.tokenizer, hparams, dataset_order)
        # print('=============final joint acc on test===================')
        os.makedirs(os.path.join(hparams.saving_dir, 'fwt_predictions'), exist_ok=True)
        print(pred_metrics, file=open(os.path.join(hparams.saving_dir, 'fwt_predictions', 'test_res.txt'), 'a'))

    if hparams.do_eval_selectinit_fwt:
        model.model.load_state_dict(torch.load(os.path.join(hparams.saving_dir, 'pytorch_model.bin')))
        model.eval()
        test_dataset = OrderedDict({d: test_datasets[d] for d in dataset_order})
        test_dataset = model.dataset_class(
            tokenizer=model.tokenizer,
            type_path='test',
            dialogs=test_dataset,
            domain2slot=domain2slot,
            num_domain_prompt=hparams.num_domain_prompt,
            small_sample_run=hparams.small_sample_run,
            multitask=hparams.multi)

        target_domain_to_init_domain = {}
        for domain_idx, domain in enumerate(dataset_order):
            if domain_idx == 0:
                continue
            if domain_idx == 1:
                target_domain_to_init_domain[domain] = dataset_order[0]
                continue

            print('trying to choose best prompt initialization for {} domain'.format(domain))
            model.zero_grad()
            model.eval()

            init_domain = None
            min_loss = float('inf')
            for prompt_domain in dataset_order[:domain_idx]:
                domain_val_loss_given_prompt = 0
                print('=' * 50)
                print('calculating val loss using {} domain prompt'.format(prompt_domain))
                dials = val_dataset.dialogs[domain]

                num_dialog_per_batch = hparams.eval_batch_size // 2
                assert num_dialog_per_batch > 0

                # prepare batches
                if hparams.small_sample_run:
                    dials = dials[:10]

                for s in tqdm(range(0, len(dials), num_dialog_per_batch), desc=str(domain)):
                    test_batch = []
                    batch_dials = dials[s: s + num_dialog_per_batch]
                    for dial in batch_dials:
                        test_batch.append(
                            val_dataset.convert_dial_to_example(
                                dial,
                                domain,
                                val_dataset.domain2soft_prompt(prompt_domain)))
                    batch = val_dataset.collate_fn(test_batch)
                    batch = {k: v.to(model.model.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    domain_val_loss_given_prompt += model.validation_step(batch, batch_idx=s, only_loss=True)['loss'].item()
                    del batch
                print('val_loss = {}'.format(domain_val_loss_given_prompt))
                if domain_val_loss_given_prompt < min_loss:
                    init_domain = prompt_domain
                    min_loss = domain_val_loss_given_prompt
            target_domain_to_init_domain[domain] = init_domain
        print('selected teachers:')
        print(target_domain_to_init_domain)
        pred_metrics = prompt5_predict_select_fwt(model.model, test_dataset, model.tokenizer, hparams, target_domain_to_init_domain)
        # print('=============final joint acc on test===================')
        os.makedirs(os.path.join(hparams.saving_dir, 'fwt_predictions'), exist_ok=True)
        print(pred_metrics, file=open(os.path.join(hparams.saving_dir, 'fwt_predictions', 'test_res.txt'), 'w'))



def prepare_prompt_params(parser):
    parser = prepare_MemT5PromptDST_parser(parser)
    parser.add_argument("--dataset_list", type=str, default="SGD,TM19,TM20,MWOZ", help="Path for saving")
    parser.add_argument('--CL', type=str, default="MULTI")
    parser.add_argument('--task_type', type=str, default="NLG")
    parser.add_argument('--samples_per_domain', default=-1, type=int, help="restrict samples per domain for fast run")
    parser.add_argument("--use_cache", action='store_true', help="use cached dataloader")
    parser.add_argument("--test_every_step", action='store_true', help="continual baseline")
    parser.add_argument("--debug", action='store_true', help="continual baseline")
    parser.add_argument("--setting", type=str, default="single", help="Path for saving")
    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument('--test_file_path', default=None)
    parser.add_argument("--do_eval", action='store_true', help="evaluation")
    parser.add_argument('--inclusive_domains', default=None)
    parser.add_argument('--gt_domain_test', action='store_true', help='use ground truth domain predictions')
    parser.add_argument("--episodic_mem_size", type=int, default=50,
                        help="number of batch/sample put in the episodic memory")
    parser.add_argument("--meta_lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--meta_n_epochs", type=int, default=5, help="training epoch for meta loop")
    parser.add_argument("--first_lr", type=float, default=0.5, help="Learning rate for the first domain")
    parser.add_argument("--first_epochs", type=int, default=20, help="training epoch for meta loop")
    parser.add_argument("--backward_lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--backward_epochs", type=int, default=1, help="training epoch for meta loop")
    parser.add_argument("--max_num_teachers", type=int, default=1, help="how many teacher for KD")
    parser.add_argument("--permute_desc", action='store_true', help="permute slots' descriptions during training")
    parser.add_argument('--small_sample_run', action='store_true')
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument('--embedding_initialization', default='vocab_sample')
    parser.add_argument('--model_type')

    parser.add_argument('--same_pos_emb_for_prompts', action='store_true',
                        help='use same pos embedding for each prompt token')

    parser.add_argument('--max_train_dials_per_domain', type=int, help='limit dials for each domain')
    parser.add_argument('--choose_teacher_meta_prompt', action='store_true')
    parser.add_argument('--metaprompt_reinit', action='store_true')
    parser.add_argument("--teacher_loss_th", type=float, default=100,
                        help="use teacher if its loss in current domain < th")

    # parser.add_argument('--ewc_prompt', action='store_true')
    # parser.add_argument('--ewc_prompt_mem_size', default=None, type=int)
    # parser.add_argument('--choose_best_initialization', action='store_true')
    #
    # parser.add_argument('--ewc_meta_prompt', action='store_true', help='init by clinit style and ewc on last domain')

    # parser.add_argument('--use_fake_distill', action='store_true', help='use fake distill at mem_meta_prompt training')
    # parser.add_argument('--multitask_include_curdomain', action='store_true',
    #                     help='use current domain in multitask-pretrain')
    # parser.add_argument('--mem_per_domain_for_memmetainit', type=int, default=50,
    #                     help='mem size for every domain when cl=mem_meta_prompt ')
    parser.add_argument('--aug_train_metaprompt', action='store_true',
                        help='shuffle/add/drop descriptions to augment train metaprompt')
    parser.add_argument('--generate_fake_example', action='store_true',
                        help='use dial+prev_prompt+prev_desc to generate fake training examples')

    parser.add_argument('--dataset_order', type=int, help='choose dataset_order for one run')

    parser.add_argument('--test_on_memory', action='store_true')
    parser.add_argument('--select_init', action='store_true')
    parser.add_argument('--num_domain_prompt', type=int, help="num of prompt token per domain", required=True)
    parser.add_argument("--do_eval_fwt", action='store_true', help="evaluation")
    parser.add_argument("--do_eval_selectinit_fwt", action='store_true', help="evaluation")

    parser.add_argument('--no_memory_uniform_dist', action='store_true')

    args = parser.parse_args()

    args.saving_dir = args.output_dir
    args.valid_batch_size = args.eval_batch_size
    args.test_batch_size = args.eval_batch_size
    args.gradient_accumulation_steps = args.accumulate_grad_batches
    args.n_epochs = args.max_epochs
    args.lr = args.learning_rate
    args.max_norm = args.gradient_clip_val
    args.pred_batch_size = args.eval_batch_size

    args.max_history = 100
    return args


if __name__ == '__main__':
    parser = ArgumentParser()
    hyperparams = prepare_prompt_params(parser)
    print(hyperparams)
    main(hyperparams)
