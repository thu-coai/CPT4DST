set -e
GPU=0

CL_TYPE="FWBW_PROMPT"
NUM_DOMAIN_PROMPT=100
INIT_STYLE="vocab_sample"

LR=0.5
EPOCH=10
FIRST_LR=0.5
FIRST_EPOCH=20
META_LR=0.5
META_EPOCH=10
BACK_LR=1e-2
BACK_EPOCH=5
MEM_SIZE=50
NUM_ACC_STEP=2
BATCH_SIZE=8
DEV_BATCH_SIZE=32
SEED=1

MODEL_TYPE=t5
PRETRAINED_MODEL=${T5_PATH}

for ORDER in 1 2 3 4 5
do
  # need to run the `run_train_cpt_w_mem_augq.sh` with corresponding setting first
  FW_MODEL="output/FW_PROMPT/prompt100_augq_order${ORDER}_vocab_sample_mem50_lr0.5_epoch10_firstlr0.5_firstEpoch20_metalr0.5_metaEpoch10_BZ8_ACC2_seed1"
  added_params="--CL ${CL_TYPE} --num_domain_prompt ${NUM_DOMAIN_PROMPT}  --embedding_initialization ${INIT_STYLE}"
  train_prompt_params="--episodic_mem_size ${MEM_SIZE} --meta_lr ${META_LR} --meta_n_epochs ${META_EPOCH}
                      --first_lr ${FIRST_LR} --first_epochs ${FIRST_EPOCH} --learning_rate=${LR}
                      --num_train_epochs=${EPOCH} --accumulate_grad_batches=${NUM_ACC_STEP}
                      --train_batch_size=${BATCH_SIZE} --eval_batch_size=${DEV_BATCH_SIZE}"
  testing_params="--trained_model_path ${FW_MODEL} --backward_lr ${BACK_LR} --backward_epochs ${BACK_EPOCH}"
  OUTPUT_DIR_NAME="${FW_MODEL}/backward_save_epoch${BACK_EPOCH}"

  mkdir -p $OUTPUT_DIR_NAME
  cp $0 $OUTPUT_DIR_NAME

  CUDA_VISIBLE_DEVICES=${GPU} python prompt_train_backward.py \
    --dataset_list SGD \
    --task_type DST \
    --do_train \
    --do_eval \
    --test_every_step \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path=${PRETRAINED_MODEL} \
    --output_dir=$OUTPUT_DIR_NAME \
    --gpus=1  \
    --seed ${SEED} \
    --dataset_order ${ORDER} \
    ${added_params} \
    ${train_prompt_params} \
    ${testing_params}
done