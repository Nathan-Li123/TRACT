CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_test.sh configs/masa-gdino/open_vocabulary_mot_test/masa_gdino_swinb_open_vocabulary_val_regionclip.py saved_models/masa_models/gdino_masa.pth 4
CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_test.sh configs/masa-gdino/open_vocabulary_mot_test/masa_gdino_swinb_open_vocabulary_test_regionclip.py saved_models/masa_models/gdino_masa.pth 4

CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_test.sh configs/masa-gdino/open_vocabulary_mot_test/masa_gdino_swinb_open_vocabulary_ovtb_ovtrack.py saved_models/masa_models/gdino_masa.pth 8
