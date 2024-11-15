#!/bin/bash

# CUDA 환경 설정
export CUDA_VISIBLE_DEVICES=0

# 기본 환경 설정
source /fsx/yusong/clap/bin/activate
cd /fsx/yusong/CLAP/src
export TRANSFORMERS_CACHE=/fsx/yusong/transformers_cache

python -m evaluate.eval_retrieval_main \
    --save-frequency 5 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --precision="fp32" \
    --warmup 0 \
    --batch-size=512 \
    --wd=0.0 \
    --epochs=50 \
    --workers=6 \
    --freeze-text \
    --amodel HTSAT-tiny \
    --tmodel roberta \
    --report-to "wandb" \
    --wandb-notes "10.17-freesound-dataset-4#" \
    --datasetnames "freesound_no_overlap_noesc50" \
    --datasetinfos "train" \
    --seed 3407 \
    --datasetpath <your_dataset_path> \
    --logs /fsx/clap_logs \
    --openai-model-cache-dir /fsx/yusong/transformers_cache \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --pretrained="/fsx/clap_logs/2022_10_17-02_08_21-model_HTSAT-tiny-lr_0.0001-b_96-j_6-p_fp32/checkpoints"