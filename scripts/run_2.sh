#!/usr/bin bash


for s in 0
do
    python DeepAligned.py \
        --dataset clinc \
        --known_cls_ratio 0.6 \
        --cluster_num_factor 1 \
        --seed $s \
        --seed2 20 \
        --freeze_bert_parameters \
        --pretrain \
        --save_model \
        --pretrain_dir pretrain_models_v2_seed20 \
        --model_dir deepaligned_INDnoise_v3 \
        --gpu_id 1 \
        --mode IND_noise

done