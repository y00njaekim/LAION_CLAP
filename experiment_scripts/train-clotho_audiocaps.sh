#!/bin/bash

# CUDA 환경 설정
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/laion_clap/LAION_CLAP/src/laion_clap:${PYTHONPATH}"
export WANDB_API_KEY='402aa2b77bad2c5914a60b6c37a08ce3bc34374a'



/home/laion_clap/LAION_CLAP/.venv/bin/python /home/laion_clap/LAION_CLAP/src/laion_clap/training/main.py \
    --save-frequency 5 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --datasetpath="/home/laion_clap/LAION_CLAP/data" \
    --precision="fp32" \
    --batch-size=192 \
    --lr=1e-4 \
    --wd=0.0 \
    --epochs=45 \
    --workers=4 \
    --amodel HTSAT-tiny \
    --tmodel bert \
    --warmup 3200 \
    --datasetnames "clotho_kor_tar" "audiocaps_kor_tar" \
    --datasetinfos "train" "train" \
    --top-k-checkpoint-select-dataset="Clotho-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --logs '/home/laion_clap/LAION_CLAP/logs' \
    --seed 3407 \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --pretrained-audio '/home/laion_clap/LAION_CLAP/artifacts/HTSAT-fullset-imagenet-tiny-map=0.467.ckpt' \
    --prefetch-factor 2 \
    --report-to wandb