CUDA_VISIBLE_DEVICES=0 python main.py \
--train_path datasets/tao_tracklets/train.json \
--test_path datasets/tao_tracklets/validation.json \
--batch_size 128 \
--loss contrastive \
--out_path outputs/attentionfusion_v2 \
--fusion_name AttentionFusion_v2 \
--clip_len 10 \
--epochs 40 \
--no_augment

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--train_path datasets/tao_tracklets/train.json \
--test_path datasets/tao_tracklets/validation_masa_gdino_gt.json \
--batch_size 1024 \
--loss contrastive \
--out_path outputs/attentionfusion_contrast_lvis+ytvis+tao\
--fusion_name AttentionFusion \
--clip_len 10 \
--epochs 50 \
--model_path outputs/attentionfusion_contrast_lvis+ytvis/epoch_025.pth \
--no_augment \
--resume