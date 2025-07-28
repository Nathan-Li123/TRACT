import torch
import json
import tqdm
import clip
import numpy as np

USE_YOLO_TEXT = True

# 进行轨迹分类，使用YOLO-WOrld输出的特征
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    print('clip model loaded')

    path = '/data2/SMVU/OC_SORT/output/yolow_tao_val_with_feats/track_embeds.json'
    with open(path, 'r') as f:
        data = json.load(f)
    print('tracking data loaded')
    
    # out_path = './classifications.json'
    # out_dict = {}

    with open('/data2/resources/TAO/annotations/validation.json', 'r') as f:
        tao_data = json.load(f)
    cate_dict = {}
    for cate in tao_data['categories']:
        cate_dict[cate['id']] = cate['name']

    track_dict = {}
    for anno in tao_data['annotations']:
        if anno['track_id'] not in track_dict:
            track_dict[anno['track_id']] = cate_dict[anno['category_id']]
    
    track_transfer_path = '/data3/AttrTrack/CLIP-exp/match.json'
    with open(track_transfer_path, 'r') as f:
        track_transfer = json.load(f)
    print('transfer dict loaded.')

    text_path = '/data3/AttrTrack/YOLO-World/tools/class_names/lvis_class_names.txt'
    classes = np.loadtxt(text_path, dtype=str, delimiter='?')
    if USE_YOLO_TEXT:
        text_feats_path = 'text_features.json'
        with open(text_feats_path, 'r') as f:
            text_feats = json.load(f)
        text_feats = torch.tensor(text_feats).to(torch.float32)
        avg_pool = torch.nn.AdaptiveAvgPool1d(768)
        text_features = avg_pool(text_feats).to(device)
    else:
        text = [f'a photo of {class_name}' for class_name in classes]
        text = clip.tokenize(text).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        # import pdb; pdb.set_trace()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.to(torch.float32)
    print('class names processed')

    correct_1, correct_3, correct_5, correct_10, correct_50, num = 0, 0, 0, 0, 0, 0
    for video_name, video_feats in tqdm.tqdm(data.items()):
        for track_id, track_feats in video_feats.items():
            gt_track = track_transfer['val/' + video_name]
            track_id = str(track_id)
            if track_id not in gt_track:
                continue

            assert len(track_feats[0]) == 512
            track_feats = torch.tensor(track_feats)

            track_feats = torch.mean(track_feats, dim=0).unsqueeze(0)
            avg_pool = torch.nn.AdaptiveAvgPool1d(768)
            track_feats = avg_pool(track_feats).to(device)
            # import pdb; pdb.set_trace()
            similarity = (100.0 * track_feats @ text_features.T).softmax(dim=-1)

            gt_id = gt_track[track_id]
            gt_class = track_dict[gt_id]
            
            values, indices = similarity[0].topk(50)
            pred_classes_50 = [classes[i] for i in indices.cpu().numpy()]
            values, indices = similarity[0].topk(10)
            pred_classes_10 = [classes[i] for i in indices.cpu().numpy()]
            values, indices = similarity[0].topk(5)
            pred_classes_5 = [classes[i] for i in indices.cpu().numpy()]
            values, indices = similarity[0].topk(3)
            pred_classes_3 = [classes[i] for i in indices.cpu().numpy()]
            values, indices = similarity[0].topk(1)
            pred_classes_1 = [classes[i] for i in indices.cpu().numpy()]
            if gt_class in pred_classes_50:
                correct_50 += 1
            if gt_class in pred_classes_10:
                correct_10 += 1
            if gt_class in pred_classes_5:
                correct_5 += 1
            if gt_class in pred_classes_3:
                correct_3 += 1
            if gt_class in pred_classes_1:
                correct_1 += 1
            num += 1
    print(f'Accuracy_50: {correct_50 / num}')
    print(f'Accuracy_10: {correct_10 / num}')
    print(f'Accuracy_5: {correct_5 / num}')
    print(f'Accuracy_3: {correct_3 / num}')
    print(f'Accuracy_1: {correct_1 / num}')
