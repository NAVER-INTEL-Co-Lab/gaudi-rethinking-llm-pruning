#!/bin/sh
echo "Start"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PT_HPU_LAZY_MODE=0 ## eager mode
export PT_HPU_GPU_MIGRATION=1

python main.py \
    --config=./configs/opt.py \
    --config.prune_method='sparsegpt' \
    --config.sparsity_type='unstructured' \
    --config.sparsity_ratio=0.5 \
    # --config.use_cr=True \
    # --config.eval_zero_shot=True \
    # --config.M=4 \
    # --config.N=2 \ 

date

echo "##### END #####"