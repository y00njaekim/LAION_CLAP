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
    --amodel HTSAT-tiny \
    --tmodel roberta \
    --warmup 3200 \
    --report-to "wandb" \
    --wandb-notes "10.16-clap-dataset-2#-htsat-roberta-fusion" \
    --datasetnames "Clotho" "audiocaps" "BBCSoundEffects" "free_to_use_sounds" "paramount_motion" "sonniss_game_effects" "wesoundeffects" "freesound_no_overlap_noesc50" "audiostock" "epidemic_sound_effects" "fsd50k_class_label" "MACS" "WavText5K" \
    --full-train-dataset "BBCSoundEffects" "free_to_use_sounds" "paramount_motion" "sonniss_game_effects" "wesoundeffects" "audiostock" "epidemic_sound_effects" "fsd50k_class_label" \
    --exclude-eval-dataset "freesound_no_overlap_noesc50" "MACS" "WavText5K" "fsd50k_class_label" \
    --datasetinfos "train" "unbalanced_train" \
    --top-k-checkpoint-select-dataset="Clotho-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --openai-model-cache-dir /fsx/yusong/transformers_cache \
    --logs /fsx/clap_logs \
    --seed 3407 \
    --datasetpath <your_dataset_path> \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "fusion" \
    --enable-fusion \
    --fusion-type "aff_2d" \
    --pretrained-audio /fsx/yusong/audio_pretrained_model/HTSAT-fullset-imagenet-map=0.467.ckpt
