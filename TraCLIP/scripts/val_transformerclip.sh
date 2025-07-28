CUDA_VISIBLE_DEVICES=2,3 python main.py \
--test_path datasets/tao_tracklets/validation_masa_gdino_gt.json \
--model_path outputs/transformerclip_lvis+ytvis+tao/epoch_040.pth \
--out_path outputs/transformerclip_lvis+ytvis+tao \
--test_class_names datasets/class_names/lvis_classes_with_attributes.txt \
--fusion_name TransformerCLIP \
--batch_size 512 \
--clip_len 10 \
--loss cosine \
--eval_only