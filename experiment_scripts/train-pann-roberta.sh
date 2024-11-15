#!/bin/bash

# CUDA 환경 설정
export CUDA_VISIBLE_DEVICES=0

# 기본 환경 설정
source /fsx/yusong/clap/bin/activate
cd /fsx/yusong/CLAP/src
export TRANSFORMERS_CACHE=/fsx/yusong/transformers_cache

python -m training.main \
    --save-frequency 5 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --precision="fp32" \
    --batch-size=96 \
    --lr=1e-4 \
    --wd=0.0 \
    --epochs=45 \
    --workers=6 \
    --amodel PANN-14 \
    --tmodel roberta \
    --warmup 500 \
    --report-to "wandb" \
    --wandb-notes "10.16-clap-dataset-1#-pann-roberta" \
    --datasetnames "Clotho" "audiocaps" \
    --datasetinfos "train" "unbalanced_train" \
    --top-k-checkpoint-select-dataset="Clotho-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --openai-model-cache-dir /fsx/yusong/transformers_cache \
    --logs /fsx/clap_logs \
    --seed 3407 \
    --datasetpath <your_dataset_path> \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --pretrained-audio /fsx/yusong/audio_pretrained_model/PANN-fullset-map=0.439.ckpt
