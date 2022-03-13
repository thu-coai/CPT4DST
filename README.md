# Continual Prompt Tuning for Dialog State Tracking

This is the official code for "Continual Prompt Tuning for Dialog State Tracking" (ACL 2022).

## Requirements

- transformers==4.6.1
- torch==1.8.1
- pytorch-lightning==1.2.5


## Datasets

We conduct experiments on [Schema-Guided Dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue). We choose 15 services from the dataset and do most experiments on them. To prepare dataset, download the dataset and put it in `data/` path.

If you are interested in the pre-processing, please check ```utils/preprocess.py``` and ```utils/dataloader.py```.

## Reproduce results in our paper

Our baseline scripts are contained in `baseline_scripts` and our prompt-tuning scripts are in `prompt_scripts`.

To reproduce results in `Table 1`, refer to scripts:


| Methods                  | avg. JGA |                   Script Path                    |
| :----------------------- | :------: | :----------------------------------------------: |
| Fine-tuning              |   14.3   |       `baseline_scripts/run_t5_vanilla.sh`       |
| EWC                      |   13.9   |         `baseline_scripts/run_t5_ewc.sh`         |
| Replay                   |   58.6   |       `baseline_scripts/run_t5_replay.sh`        |
| AdapterCL (20x)          |   49.8   |       `baseline_scripts/run_t5_adapter.sh`       |
| AdapterCL (1x)           |   30.6   |    `baseline_scripts/run_t5_small_adapter.sh`    |
| Prompt Tuning            |   48.1   |    `prompt_scripts/run_train_pt_randinit.sh`     |
| Continual Prompt Tuning  |   59.5   |      `prompt_scripts/run_train_cpt_augq.sh`      |
| w/ memory                |   60.7   |   `prompt_scripts/run_train_cpt_w_mem_augq.sh`   |
| w/ memory & backward     |   61.2   | `prompt_scripts/run_train_cpt_w_mem_augq_bwd.sh` |
| Multi-Task Prompt Tuning |   64.0   |       `prompt_scripts/run_multiprompt.sh`        |


To reproduce results in `Table 2`, refer to scripts:

|    Techniques    | avg. JGA |                 Script Path                  |
| :--------------: | :------: | :------------------------------------------: |
|        -         |   29.6   |     `prompt_scripts/run_train_noMSR.sh`      |
|      CLInit      |   41.8   |    `prompt_scripts/run_train_noMSR_cl.sh`    |
|       MSR        |   48.1   |  `prompt_scripts/run_train_pt_randinit.sh`   |
|    MSR+CLInit    |   57.6   |      `prompt_scripts/run_train_cpt.sh`       |
|  MSR+CLInit+QF   |   59.5   |    `prompt_scripts/run_train_cpt_augq.sh`    |
|  MSR+CLInit+MR   |   60.4   |   `prompt_scripts/run_train_cpt_w_mem.sh`    |
| MSR+CLInit+QF+MR |   60.7   | `prompt_scripts/run_train_cpt_w_mem_augq.sh` |


To reproduce results in `Table 3`, refer to scripts:

| Initialization | avg. JGA on 5 seeds |                 Script Path                 |
|:--------------:|:-------------------:|:-------------------------------------------:|
| RandomInit     |                48.1 | `prompt_scripts/run_train_pt_randinit.sh`   |
| SelectInit     |                54.5 | `prompt_scripts/run_train_pt_selectinit.sh` |
| CLInit         |                57.6 | `prompt_scripts/run_train_cpt.sh`           |


To reproduce results in `Table 4`, refer to `prompt_scripts/run_train_cpt.sh` and set `--dataset_order=30/1/31/32`.  

To reproduce results in `Figure 3` and `Table 5`, refer to `prompt_scripts/run_train_cpt_augq.sh` and set `--model_name_or_path` to paths to your `t5-base` or `t5-large` pre-trained weights. We use RTX-2080 GPU with 11GB RAM. In our experiments, we choose 4 batch size and 4 accumulation steps for `t5-base`
and 2 batch size, 8 accumulation steps for `t5-large`. We use fp32 training for both of them.

To reproduce results in `Table 6` and `Table 7`, set the `MEM_SIZE` accordingly. To make domain's memory size proportional to its training data size, use `--no_memory_uniform_dist` flag.

## Evaluation 

For evaluation for non-prompt experiments, refer to `gather_res_baseline.py`

For evaluation for prompt-tuning experiments, refer to `gather_res_prompt.py`

For FWT calculation for non-prompt experiments, refer to `gather_res_baseline_fwt.py`

For FWT calculation for prompt-tuning experiments, refer to `gather_res_prompt_fwt.py`

## Citation

```
@inproceedings{zhu-etal-2022-cpt4dst,
    title = "Continual Prompt Tuning for Dialog State Tracking",
    author = "Zhu, Qi and Li, Bing and Mi, Fei and Zhu, Xiaoyan and Huang, Minlie",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    publisher = "Association for Computational Linguistics",
}
```

