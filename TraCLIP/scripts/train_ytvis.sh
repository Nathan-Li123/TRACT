# transformer clip
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--train_path datasets/ytvis_tracklets/train.json \
--test_path datasets/tao_tracklets/validation.json \
--train_class_names datasets/class_names/ytvis_class_names.txt \
--batch_size 512 \
--loss cosine \
--out_path outputs/transformerclip_ytvis \
--fusion_name TransformerCLIP \
--clip_len 10 \
--save_epoch 5 \
--no_augment \
--epochs 40

# resume
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--train_path datasets/ytvis_tracklets/train.json \
--test_path datasets/tao_tracklets/validation_masa_gdino_gt.json \
--train_class_names datasets/class_names/ytvis_class_names.txt \
--model_path outputs/attentionfusion_lvis_contrast/epoch_005.pth \
--batch_size 1024 \
--loss contrastive \
--out_path outputs/attentionfusion_contrast_lvis+ytvis \
--fusion_name AttentionFusion \
--clip_len 10 \
--save_epoch 5 \
--resume \
--epochs 25 \
--no_augment