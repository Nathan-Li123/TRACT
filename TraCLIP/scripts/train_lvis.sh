CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--train_path datasets/lvis_tracklets/train.json \
--test_path datasets/tao_tracklets/validation_masa_gdino_gt.json \
--batch_size 1024 \
--loss contrastive \
--out_path outputs/attentionfusion_lvis_contrast \
--fusion_name AttentionFusion \
--clip_len 10 \
--epochs 5