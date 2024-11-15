#!/bin/bash

# CUDA 환경 설정
export CUDA_VISIBLE_DEVICES=0

python -m laion_clap.training.main \
    --save-frequency 5 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --datasetpath="<to-your-directory-containing-Clotho-not-the-path-to-Clotho>" \
    --precision="fp32" \
    --batch-size=96 \
    --lr=1e-4 \
    --wd=0.0 \
    --epochs=45 \
    --workers=6 \
    --amodel HTSAT-tiny \
    --tmodel roberta \
    --warmup 3200 \
    --datasetnames "Clotho" \
    --datasetinfos "train" \
    --top-k-checkpoint-select-dataset="Clotho-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --logs 'logs' \
    --seed 3407 \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --pretrained-audio '<path-to>/HTSAT-fullset-imagenet-map=0.467.ckpt' \
    --prefetch-factor 2