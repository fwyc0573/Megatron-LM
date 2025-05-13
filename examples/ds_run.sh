#!/bin/bash

# 设置NCCL和其他依赖项的环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens81f0
export NCCL_P2P_DISABLE=0
export CUDA_HOME=/usr/local/cuda-12.1
export PYTHONUSERBASE=/research/d1/gds/ytyang/PYTHONUSER/python
export PYTHONPATH=/research/d1/gds/ytyang/yichengfeng/Megatron-LM/examples
export PATH="/usr/lib64/openmpi/bin:$PATH"
export PIP_CACHE_DIR=/research/d1/gds/ytyang
export GRB_LICENSE_FILE=/research/d1/gds/ytyang/gurobi.lic

# DeepSpeed启动命令
deepspeed \
    --hostfile /research/d1/gds/ytyang/yichengfeng/Megatron-LM/hostfile \
    --master_addr 192.168.50.186 \
    --master_port 25555 \
    /research/d1/gds/ytyang/yichengfeng/Megatron-LM/pretrain_llama.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --distributed-backend nccl \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --num-layers 3 \
    --hidden-size 128 \
    --num-attention-heads 4 \
    --group-query-attention \
    --num-query-groups 4 \
    --ffn-hidden-size 512 \
    --position-embedding-type rope \
    --max-position-embeddings 512 \
    --make-vocab-size-divisible-by 1 \
    --norm-epsilon 1e-5 \
    --normalization RMSNorm \
    --swiglu \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --train-iters 1 \
    --log-interval 1 \
    --optimizer adam \
    --exit-interval 100 \
    --use-mcore-models \
    --seed 1403 \
    --init-method-std 0.02 \
    --lr 3e-5 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.1 \
    --min-lr 3e-6 \
    --save-interval 1000 \
    --fp16 \
    --attention-softmax-in-fp32 \
    --eval-interval 1000 \
    --eval-iters 0 \
    --data-path /research/d1/gds/ytyang/yichengfeng/Megatron-LM/data/output_prefix_llama/output_prefix_llama_text_document \
    --split 949,50,1 \
    --seq-length 512 \
    --num-workers 0 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model /research/d1/gds/ytyang/yichengfeng/Megatron-LM/tokenizers/Llama2Tokenizer/tokenizer.model \
    --data-cache-path ./data_cache/llama2_tiny_pretrain_WS8_TP2_PP2 \
    --vocab-size 3200
