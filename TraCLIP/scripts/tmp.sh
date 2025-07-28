CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--test_path datasets/tao_tracklets/ovtb_ovtrack_masa.json \
--model_path outputs/attention_cosine_lvis+tao/epoch_030.pth \
--out_path outputs/attention_cosine_lvis+tao \
--test_class_names datasets/class_names/ovtb_class_names.txt \
--fusion_name AttentionFusion \
--batch_size 1024 \
--clip_len 5 \
--loss cosine \
--eval_only \
--out_name results.json

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--test_path datasets/tao_tracklets/ovtb_ovtrack_masa.json \
--model_path outputs/attention_cosine_lvis+tao/epoch_030.pth \
--out_path outputs/attention_cosine_lvis+tao \
--test_class_names datasets/class_names/ovtb_classes_with_attributes.txt \
--fusion_name AttentionFusion \
--batch_size 1024 \
--clip_len 5 \
--loss cosine \
--eval_only \
--out_name results_attr.json