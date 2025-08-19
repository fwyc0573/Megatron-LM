#!/bin/bash

train_iters=3
trace_iter_num=1 # trace_iter_num的范围<=train_iters-2
trace_start=$(($train_iters-$trace_iter_num+1)) # [start, train_iters]

# 检查trace_iter_num是否在合理的范围内
if [ $trace_iter_num -gt $((train_iters - 2)) ]; then
  echo "Error: trace_iter_num must be less than or equal to train_iters - 2"
  exit 1
fi



world_size=16
pp=2
tp=1


# size variables
EP=2
NUM_EXPERTS=2
MODEL_SIZE="Mixtral_${NUM_EXPERTS}x1.75B"
if   [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=32; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_LAYERS=80;
elif [[ ${MODEL_SIZE} == 175 ]];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_LAYERS=96;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_LAYERS=4;
elif [[ ${MODEL_SIZE} == 30 ]];   then HIDDEN_SIZE=7680;  NUM_HEAD=48; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 40 ]];   then HIDDEN_SIZE=9216;  NUM_HEAD=72; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 6.7 ]];  then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_LAYERS=32;
elif [[ ${MODEL_SIZE} == "Mixtral_${NUM_EXPERTS}x1.75B" ]]; then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_LAYERS=8 ; FFN_HIDDEN_SIZE=14336;
elif [[ ${MODEL_SIZE} == "Mixtral_${NUM_EXPERTS}x7B" ]]; then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=14336;
elif [[ ${MODEL_SIZE} == "Mixtral_${NUM_EXPERTS}x22B" ]]; then HIDDEN_SIZE=6144;  NUM_HEAD=56; NUM_LAYERS=56; FFN_HIDDEN_SIZE=16384;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi


# Calculate dp and check if it's an integer
dp=$((world_size / (tp * pp)))
if [ $((world_size % (tp * pp))) -ne 0 ]; then
  echo "Error: dp (data parallelism) must be an integer. Please check world_size, tp, and pp."
  exit 1
fi



num_micbatch=4*${pp}
micro_batch_size=1
global_batch_size=$((num_micbatch * micro_batch_size * dp))
MAX_SEQ_LEN=2048



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
    -exp $EP \
    --num-experts $NUM_EXPERTS \
    --trace-start $trace_start 
    #     --fp16 \