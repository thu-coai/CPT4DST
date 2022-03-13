set -e
GPU=0

CL_TYPE="RAND_PROMPT"
NUM_DOMAIN_PROMPT=100
INIT_STYLE="vocab_sample"

LR=0.5
EPOCH=20
FIRST_LR=0.5
FIRST_EPOCH=20
META_LR=0.5
META_EPOCH=10
MEM_SIZE=0
NUM_ACC_STEP=2
BATCH_SIZE=8
DEV_BATCH_SIZE=32
SEED=1
DATASET_ORDER=1

MODEL_TYPE=t5
PRETRAINED_MODEL=${T5_PATH}

added_params="--CL ${CL_TYPE} --num_domain_prompt ${NUM_DOMAIN_PROMPT}  --embedding_initialization ${INIT_STYLE}"
train_prompt_params="--episodic_mem_size ${MEM_SIZE} --meta_lr ${META_LR} --meta_n_epochs ${META_EPOCH} --first_lr ${FIRST_LR} --first_epochs ${FIRST_EPOCH} --learning_rate=${LR} --num_train_epochs=${EPOCH} --accumulate_grad_batches=${NUM_ACC_STEP} --train_batch_size=${BATCH_SIZE} --eval_batch_size=${DEV_BATCH_SIZE}"


for SEED in 1 2 3 4 5
do
  OUTPUT_DIR_NAME="output/${CL_TYPE}/genstate_randinit_prompt${NUM_DOMAIN_PROMPT}_order${DATASET_ORDER}_${INIT_STYLE}_mem${MEM_SIZE}_lr${LR}_epoch${EPOCH}_firstlr${FIRST_LR}_firstEpoch${FIRST_EPOCH}_metalr${META_LR}_metaEpoch${META_EPOCH}_BZ${BATCH_SIZE}_ACC${NUM_ACC_STEP}_seed${SEED}"

  mkdir -p $OUTPUT_DIR_NAME
  cp $0 $OUTPUT_DIR_NAME

  CUDA_VISIBLE_DEVICES=${GPU} python prompt_train_genstate.py \
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
    --dataset_order ${DATASET_ORDER} \
    ${added_params} \
    ${train_prompt_params}
done
