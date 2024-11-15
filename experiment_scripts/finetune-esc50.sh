#!/bin/bash

# CUDA 환경 설정
export CUDA_VISIBLE_DEVICES=0

# 기본 환경 설정
source /fsx/yusong/clap/bin/activate
cd /fsx/yusong/CLAP/src
export TRANSFORMERS_CACHE=/fsx/yusong/transformers_cache

python -m evaluate.eval_linear_probe \
    --save-frequency 50 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --precision="fp32" \
    --warmup 0 \
    --batch-size=160 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=100 \
    --workers=4 \
    --freeze-text \
    --amodel PANN-14 \
    --tmodel roberta \
    --report-to "wandb" \
    --wandb-notes "10.14-finetune-esc50" \
    --datasetnames "esc50" \
    --datasetinfos "train" \
    --seed 3407 \
    --datasetpath <your_dataset_path> \
    --logs /fsx/clap_logs \
    --lp-loss="ce" \
    --lp-metrics="acc" \
    --lp-lr=1e-4 \
    --lp-mlp \
    --class-label-path="../class_labels/ESC50_class_labels_indices_space.json" \
    --openai-model-cache-dir /fsx/yusong/transformers_cache \
    --pretrained="/fsx/clap_logs/2022_10_14-04_05_14-model_PANN-14-lr_0.0001-b_160-j_6-p_fp32/checkpoints" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --optimizer "adam"