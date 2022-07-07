#!/usr/bin bash


for s in 0 1 2
do
    python DeepAligned.py \
        --dataset clinc \
        --known_cls_ratio 0.9 \
        --cluster_num_factor 1 \
        --seed $s \
        --seed2 10 \
        --freeze_bert_parameters \
        --save_model \
        --pretrain_dir pretrain_models_v1_0.9_seed42_scl \
        --pretrain \
        --model_dir deepaligned_v1_0.9_scl  \
        --gpu_id 1 \

done
