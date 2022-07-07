#!/usr/bin bash




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
        --pretrain_dir pretrain_models_v2_seed20 \
        --model_dir pipeline_v2_seed20_clinc \
        --gpu_id 1 \

done

