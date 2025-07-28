# LVIS pretrain, using TransformerCLIP
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--train_path datasets/lvis_tracklets/train.json \
--test_path datasets/tao_tracklets/validation.json \
--batch_size 512 \
--loss cosine \
--out_path outputs/transformerclip_cosine_lvis \
--fusion_name TransformerCLIP \
--clip_len 5 \
--epochs 10

# TAO finetune, using TransformerCLIP
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--train_path datasets/tao_tracklets/train.json \
--test_path datasets/tao_tracklets/validation.json \
--batch_size 512 \
--loss cosine \
--out_path outputs/transformerclip_cosine_lvis+tao \
--fusion_name TransformerCLIP \
--clip_len 10 \
--epochs 20 \
--model_path outputs/transformerclip_cosine_lvis/epoch_010.pth \
--no_augment \
--resume

# LVIS pretrain, using AttentionFusion
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--train_path datasets/lvis_tracklets/train.json \
--test_path datasets/tao_tracklets/validation.json \
--batch_size 512 \
--loss cosine \
--out_path outputs/attention_cosine_lvis \
--fusion_name AttentionFusion \
--clip_len 10 \
--epochs 10

# TAO finetune, using AttentionFusion
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--train_path datasets/tao_tracklets/train.json \
--test_path datasets/tao_tracklets/validation.json \
--batch_size 512 \
--loss cosine \
--out_path outputs/attention_cosine_lvis+tao \
--fusion_name AttentionFusion \
--clip_len 10 \
--model_path outputs/attention_cosine_lvis/epoch_010.pth \
--no_augment \
--epochs 30 \
--resume

# TAO training, using CrossSimilarity
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--train_path datasets/tao_tracklets/train.json \
--test_path datasets/tao_tracklets/validation.json \
--batch_size 128 \
--loss cross_entropy \
--out_path outputs/cross_similarity_tao \
--fusion_name CrossSimilarity \
--clip_len 10 \
--epochs 20

# LVIS pretrain, using AttentionFusion, R50 backbone
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--train_path datasets/lvis_tracklets/train.json \
--test_path datasets/tao_tracklets/validation.json \
--extraction_name R50 \
--batch_size 128 \
--loss cosine \
--out_path outputs/attention_r50_lvis \
--fusion_name AttentionFusion \
--clip_len 10 \
--epochs 10
