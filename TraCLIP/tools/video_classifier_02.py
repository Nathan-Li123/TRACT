import os
import torch
import json
import tqdm
import clip
import numpy as np
import PIL.Image as Image


def get_box_image(pred_data, video_name):
    track_dict = {}
    for row in pred_data:
        frame_id = int(row[0])
        track_id = int(row[1])
        if track_id not in track_dict:
            track_dict[track_id] = []
        bbox = [float(i) for i in row[2:6]]
        img_path = os.path.join('/data2/resources/TAO', index_dict[video_name][str(frame_id)])
        img = Image.open(img_path)
        bbox = [int(i) for i in bbox]
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        cropped_img = img.crop(bbox)
        track_dict[track_id].append(cropped_img)
    return track_dict


# 进行轨迹分类，使用YOLO-WOrld输出bbox抠图，然后使用CLIP进行特征提取并分类
if __name__ == '__main__':
    with open('/data2/resources/TAO/annotations/validation.json', 'r') as f:
        tao_data = json.load(f)
    tao_dict = {}
    for video in tao_data['videos']:
        tao_dict[video['name']] = []
    for image in tao_data['images']:
        tao_dict[image['video']].append(image['file_name'])
    index_dict = {}
    for vid_name, images in tao_dict.items():
        tmp = {}
        for i, image in enumerate(images):
            tmp[str(i+1)] = image
        index_dict[vid_name] = tmp

    cate_dict = {}
    for cate in tao_data['categories']:
        cate_dict[cate['id']] = cate['name']

    track_dict = {}
    for anno in tao_data['annotations']:
        if anno['track_id'] not in track_dict:
            track_dict[anno['track_id']] = cate_dict[anno['category_id']]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    print('clip model loaded')

    text_path = '/data3/AttrTrack/YOLO-World/tools/class_names/lvis_class_names.txt'
    classes = np.loadtxt(text_path, dtype=str, delimiter='?')
    text = [f'{class_name}' for class_name in classes]
    text = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    # import pdb; pdb.set_trace()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.to(torch.float32)
    print('class names processed')

    track_transfer_path = '/data3/AttrTrack/CLIP-exp/match.json'
    with open(track_transfer_path, 'r') as f:
        track_transfer = json.load(f)

    # out_path = './classifications.json'
    # out_dict = {}

    correct_1, correct_3, correct_5, correct_10, correct_50, num = 0, 0, 0, 0, 0, 0
    pred_path= '/data2/SMVU/OC_SORT/output/yolow_tao_val_with_feats'
    sub_sets = os.listdir(pred_path)
    for sub_set in sub_sets:
        sub_set_path = os.path.join(pred_path, sub_set)
        if not os.path.isdir(sub_set_path):
            continue
        videos = os.listdir(sub_set_path)
        for video in tqdm.tqdm(videos, desc=sub_set):
            video_path = os.path.join(sub_set_path, video)
            pred_data = np.loadtxt(video_path, delimiter=',', dtype=str)
            if len(pred_data) == 0:
                continue
            if pred_data.ndim == 1:
                pred_data = pred_data[np.newaxis, :]
            vid_name = 'val/' + sub_set + '/' + video.split('.')[0]
            # out_dict[sub_set + '/' + video.split('.')[0]] = {}
            
            # out_dict[vid_name] = {}
            track_data = get_box_image(pred_data, vid_name)
            for track_id, box_imgs in track_data.items():
                gt_track = track_transfer[vid_name]
                track_id = str(track_id)
                if track_id not in gt_track:
                    continue

                v_feats = []
                for box_img in box_imgs:
                    v_feat = preprocess(box_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        v_feat = model.encode_image(v_feat)
                    v_feat /= v_feat.norm(dim=-1, keepdim=True)
                    v_feats.append(v_feat)
                v_feats = torch.cat(v_feats, dim=0)
                v_feats = torch.mean(v_feats, dim=0).unsqueeze(0).to(torch.float32)
                similarity = (100.0 * v_feats @ text_features.T).softmax(dim=-1)

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
                # out_dict[sub_set + '/' + video.split('.')[0]][track_id] = [classes[i] for i in indices.cpu().numpy()]
    # print('classification done')
    # with open(out_path, 'w') as f:
    #     json.dump(out_dict, f)
