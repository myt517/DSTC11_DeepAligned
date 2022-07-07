#!/usr/bin bash


for s in 0
do
    python DeepAligned_pipeline.py \
        --dataset clinc \
        --known_cls_ratio 0.8 \
        --cluster_num_factor 1 \
        --type 0 \
        --seed $s \
        --seed2 20 \
        --freeze_bert_parameters \
        --pretrain \
        --method classify \
        --save_model \
        --pretrain_dir pretrain_models_cross_0.8_1 \
        --model_dir pipeline_v1_cross_1 \
        --gpu_id 1 \
        --mode cross_domain

done


for s in 0
do
    python DeepAligned_pipeline.py \
        --dataset clinc \
        --known_cls_ratio 0.6 \
        --cluster_num_factor 1 \
        --type 0 \
        --seed $s \
        --seed2 20 \
        --freeze_bert_parameters \
        --pretrain \
        --save_model \
        --method classify \
        --pretrain_dir pretrain_models_cross_0.6_1 \
        --model_dir pipeline_v2_cross_1 \
        --gpu_id 1 \
        --mode cross_domain

done



for s in 0
do
    python DeepAligned_pipeline.py \
        --dataset clinc \
        --known_cls_ratio 0.4 \
        --cluster_num_factor 1 \
        --type 0 \
        --seed $s \
        --seed2 20 \
        --freeze_bert_parameters \
        --pretrain \
        --save_model \
        --method classify \
        --pretrain_dir pretrain_models_cross_0.4_1 \
        --model_dir pipeline_v3_cross_1 \
        --gpu_id 1 \
        --mode cross_domain

done



