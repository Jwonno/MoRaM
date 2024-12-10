#!/usr/bin/env bash         
export CUDA_VISIBLE_DEVICES=1
# hyperparameters
DATASET=sun397
SEED=214
EPOCHS=100
batch_size=16
LR=0.0015
WEIGHT_DECAY=0
warmup_lr=1e-7
warmup_epochs=10
min_lr=1e-8
drop_path=0
idx_path=./max_index

exp_name=vtab_vit_supervised_${LR}_${WEIGHT_DECAY}_${batch_size}_${EPOCHS}
python train.py \
    --data-path=./data/vtab-1k/${DATASET} \
    --data-set=${DATASET} \
    --model_name=vit_base_patch16_224_in21k \
    --output_dir=./saves_vtab/${DATASET}/${exp_name} \
    --batch-size=${batch_size} \
    --lr=${LR} \
    --epochs=${EPOCHS} \
    --weight-decay=${WEIGHT_DECAY} \
    --smoothing=0 \
    --seed=${SEED} \
    --exp_name=${exp_name} \
    --warmup-lr=${warmup_lr} \
    --warmup-epochs=${warmup_epochs} \
    --min-lr=${min_lr} \
    --drop-path=${drop_path} \
    --pre-idx=${idx_path} \
    --no-aug \
    --test