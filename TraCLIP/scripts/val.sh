# evaluate using ResidualAttentionBlock
# CUDA_VISIBLE_DEVICES=0 python main.py \
# --model_path outputs/residual_attention_5/epoch_037.pth \
# --out_path outputs/residual_attention_5 \
# --test_class_names datasets/class_names/lvis_class_names.txt \
# --test_name validation_ovtrack.json \
# --fusion_name ResidualAttentionBlock \
# --clip_len 5 \
# --eval_only

# evaluate using AvgPool
# python main.py \
# --out_path outputs/cosine_10 \
# --test_class_names datasets/class_names/lvis_class_names.txt \
# --test_name validation_yolow_l_1203.json \
# --fusion_name AvgPool \
# --eval_only

# evaluate using Voting
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--out_path outputs \
--test_class_names datasets/class_names/lvis_classes_with_attributes.txt \
--test_path datasets/tao_tracklets/validation_masa_gdino.json \
--fusion_name Voting \
--batch_size 128 \
--clip_len 10 \
--loss cosine \
--eval_only

# evaluate using AttentionFusion
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--test_path datasets/tao_tracklets/validation_ovtrack.json \
--model_path outputs/attention_cosine_lvis+tao/epoch_030.pth \
--out_path outputs/attention_cosine_lvis+tao \
--test_class_names datasets/class_names/lvis_class_names.txt \
--fusion_name AttentionFusion \
--batch_size 1024 \
--clip_len 5 \
--loss contrastive \
--eval_only

# evaluation using CrossSimilarity
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--test_path datasets/tao_tracklets/validation_masa_gdino.json \
--model_path outputs/cross_similarity_tao/epoch_020.pth \
--out_path outputs/cross_similarity_tao \
--test_class_names datasets/class_names/lvis_class_names.txt \
--fusion_name CrossSimilarity \
--batch_size 512 \
--clip_len 10 \
--loss cross_entropy \
--eval_only

# evaluate using TransformerCLIP
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--test_path datasets/tao_tracklets/validation_masa_gdino_gt.json \
--model_path outputs/transformerclip_lvibis+ytvis+tao/epoch_031.pth \
--out_path outputs/transformerclip_lvis+ytvis+tao/attr \
--test_class_names datasets/class_names/lvis_classes_with_attributes.txt \
--fusion_name TransformerCLIP \
--batch_size 512 \
--clip_len 10 \
--loss cosine \
--eval_only

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
# --model_path outputs/transformerclip_cosine/epoch_059.pth \
# --out_path outputs/transformerclip_cosine \
# --test_class_names datasets/class_names/lvis_class_names.txt \
# --test_name validation_ovtrack.json \
# --fusion_name TransformerCLIP \
# --batch_size 32 \
# --loss cosine \
# --eval_only

# evaluation with AttentionFusion_v2
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--test_path datasets/tao_tracklets/validation_masa_gdino_gt.json \
--model_path outputs/attentionfusion_contrast_lvis+ytvis+tao/epoch_050.pth \
--out_path outputs/attentionfusion_v2_lvis+tao_pred \
--test_class_names datasets/class_names/lvis_class_names.txt \
--fusion_name AttentionFusion \
--batch_size 1024 \
--clip_len 10 \
--loss cosine \
--eval_only

