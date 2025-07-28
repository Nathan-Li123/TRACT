CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--test_path datasets/tao_tracklets/test_yolow_xl_masa_mix.json \
--model_path outputs/attention_cosine_lvis+tao/epoch_030.pth \
--out_path outputs/attention_cosine_lvis+tao \
--test_class_names datasets/class_names/lvis_classes_with_attributes.txt \
--fusion_name AttentionFusion \
--batch_size 1024 \
--clip_len 10 \
--loss cosine \
--eval_only