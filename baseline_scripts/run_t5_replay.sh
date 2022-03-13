SEED=1
for DATASET_ORDER in 1 2 3 4 5
do
  CUDA_VISIBLE_DEVICES=0 python mytrain.py \
    --do_train \
    --CL REPLAY \
    --task_type DST \
    --model_checkpoint ${T5_PATH} \
    --saving_dir output/t5_replay_seed${SEED}_order${DATASET_ORDER}_notuniform \
    --max_history 200 \
    --dataset_list SGD \
    --n_epochs 20 \
    --test_every_step \
    --train_batch_size 8 \
    --valid_batch_size 32 \
    --test_batch_size 32 \
    --lr 3e-5 \
    --seed ${SEED} \
    --episodic_mem_size 50 \
    --dataset_order ${DATASET_ORDER}
done

