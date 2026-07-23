#!/bin/bash
# 该脚本用于simulator的mg scheduling plans获取
# 获取的shedules中的op都是粗粒度的，包含了所有send/recv通信，get_batch、fwd、bwd中的通信算子需要根据profile补齐 

train_iters=4
trace_iter_num=1 # trace_iter_num的范围<=train_iters-2
trace_start=$(($train_iters-$trace_iter_num+1)) # [start, train_iters]

# 检查trace_iter_num是否在合理的范围内
if [ $trace_iter_num -gt $((train_iters - 2)) ]; then
  echo "Error: trace_iter_num must be less than or equal to train_iters - 2"
  exit 1
fi

MODEL_SIZE=70 # "tiny"

if   [[ ${MODEL_SIZE} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_QUERY_GROUP=8; NUM_LAYERS=4; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

MAX_SEQ_LEN=2048

python mg_test.py \
    --local-size 8 \
    --world-size 8 \
    --micro-batch-size 2 \
    --global-batch-size 8 \
    --seq-length $MAX_SEQ_LEN \
    --hidden-size $HIDDEN_SIZE \
    --fp16 \
    --train-iters $train_iters \
    -pp 4 \
    -tp 8 \
    --trace-start $trace_start