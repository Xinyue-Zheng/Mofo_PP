#!/usr/bin/env bash
# run_fixed_weather.sh — Ablation 1: fixed global period, no channel attention (original MoFo)
# MoFo++ ablation study — Weather dataset (10-min intervals, PERIODIC=144)

set -e

DATA="weather.csv"
SEQ_LEN=336
D_MODEL=24
PERIODIC=144
LR=0.01
PATIENCE=10

for HORIZON in 96 192 336 720; do
    echo "=== [Fixed] horizon=${HORIZON} ==="
    python ./scripts/run_benchmark.py \
        --config-path "rolling_forecast_config.json" \
        --data-name-list "${DATA}" \
        --strategy-args "{\"horizon\": ${HORIZON}}" \
        --model-hyper-params "{\"batch_size\": 16, \"d_model\": ${D_MODEL}, \"horizon\": ${HORIZON}, \"lr\": ${LR}, \"norm\": true, \"seq_len\": ${SEQ_LEN}, \"patience\": ${PATIENCE}, \"periodic\": ${PERIODIC}, \"bias\": 1, \"cias\": 1}" \
        --adapter "MoFo_adapter" \
        --model-name "time_series_library.MoFo" \
        --gpus 0 \
        --num-workers 1 \
        --timeout 60000 \
        --save-path "Weather/MoFo_fixed_h${HORIZON}"
done
