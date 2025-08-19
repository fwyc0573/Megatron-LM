#!/bin/bash

train_iters=3
trace_iter_num=1 # trace_iter_num的范围<=train_iters-2
trace_start=$(($train_iters-$trace_iter_num+1)) # [start, train_iters]

# 检查trace_iter_num是否在合理的范围内
if [ $trace_iter_num -gt $((train_iters - 2)) ]; then
  echo "Error: trace_iter_num must be less than or equal to train_iters - 2"
  exit 1
fi

# MODEL_SIZE=13 # "tiny"

# if   [[ ${MODEL_SIZE} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
# elif [[ ${MODEL_SIZE} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
# elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
# elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_QUERY_GROUP=8; NUM_LAYERS=4; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
# else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
# fi

# size variables
MODEL_SIZE="tiny" # "tiny"

# if   [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=32; NUM_LAYERS=40;
# elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=80; NUM_LAYERS=80;
# elif [[ ${MODEL_SIZE} == 135 ]];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_LAYERS=96;
# elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_LAYERS=4;
# else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
# fi
if   [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=32; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_LAYERS=80;
elif [[ ${MODEL_SIZE} == 175 ]];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_LAYERS=96;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_LAYERS=4;
elif [[ ${MODEL_SIZE} == 30 ]];   then HIDDEN_SIZE=7680;  NUM_HEAD=48; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 40 ]];   then HIDDEN_SIZE=9216;  NUM_HEAD=72; NUM_LAYERS=40;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi



MAX_SEQ_LEN=2048

# New variables for parallelism and batch sizes
world_size=4
pp=2
tp=1

# Calculate dp and check if it's an integer
dp=$((world_size / (tp * pp)))
if [ $((world_size % (tp * pp))) -ne 0 ]; then
  echo "Error: dp (data parallelism) must be an integer. Please check world_size, tp, and pp."
  exit 1
fi

num_micbatch=1
micro_batch_size=2
global_batch_size=$((num_micbatch * micro_batch_size * dp))

# Run the python script with the calculated variables
python mg_test.py \
    --local-size 8 \
    --world-size $world_size \
    --micro-batch-size $micro_batch_size \
    --global-batch-size $global_batch_size \
    --seq-length $MAX_SEQ_LEN \
    --hidden-size $HIDDEN_SIZE \
    --train-iters $train_iters \
    --model-size $MODEL_SIZE \
    -pp $pp \
    -tp $tp \
    --trace-start $trace_start 
    #     --fp16 \