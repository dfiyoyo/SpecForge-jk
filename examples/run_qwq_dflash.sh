#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-64}

NUM_GPUS=${1:-4}
TP_SIZE=${2:-4}
ATTENTION_BACKEND=${3:-flex_attention}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$ROOT_DIR/cache/dataset/sharegpt_train.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/qwq-32b-dflash-sharegpt}
MASK_TOKEN_ID=${MASK_TOKEN_ID:-151662}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path Qwen/QwQ-32B \
    --target-model-backend sglang \
    --tp-size $TP_SIZE \
    --draft-config-path $ROOT_DIR/configs/qwq-dflash.json \
    --train-data-path $TRAIN_DATA_PATH \
    --build-dataset-num-proc $SPECFORGE_DATA_NUM_PROC \
    --output-dir $OUTPUT_DIR \
    --num-epochs 6 \
    --batch-size 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --block-size 16 \
    --mask-token-id $MASK_TOKEN_ID
