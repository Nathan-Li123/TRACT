import os
import json
import tqdm
import numpy as np


# 进行轨迹分类，直接食用YOLO-World的分类结果，使用投票的思路进行分类
if __name__ == '__main__':
    with open('/data2/resources/TAO/annotations/validation_ours_v1.json', 'r') as f:
        tao_data = json.load(f)
    track_dict = {}
    cate_dict = {}
    for cate in tao_data['categories']:
        cate_dict[cate['id']] = cate['name']
    # TODO: 需要确认一个点，YOLO-World输出的类别是从0开始还是从1开始
    for anno in tao_data['annotations']:
        if anno['track_id'] not in track_dict:
            track_dict[anno['track_id']] = anno['category_id']

    # TODO: 修改为match文件的路径
    track_transfer_path = 'datasets/matches/match_yolow_l.json'
    with open(track_transfer_path, 'r') as f:
        track_transfer = json.load(f)

    correct, num = 0, 0
    pred_root= '/data2/SMVU/OC_SORT/output/yolow_l_val_ovtao'
    cls_preds_path = os.path.join(pred_root, 'track_classes.json')
    with open(cls_preds_path, 'r') as f:
        cls_preds = json.load(f)
    
    sub_sets = os.listdir(pred_root)
    for sub_set in sub_sets:
        sub_set_path = os.path.join(pred_root, sub_set)
        if not os.path.isdir(sub_set_path):
            continue
        videos = os.listdir(sub_set_path)
        for video in tqdm.tqdm(videos, desc=sub_set):
            vid_name = sub_set + '/' + video.split('.')[0]
            if vid_name not in cls_preds:
                continue
            gt_track = track_transfer['val/' + vid_name]
            preds_data = cls_preds[vid_name]
            for track_id, classes in preds_data.items():
                if track_id not in gt_track:
                    continue
                gt_id = gt_track[track_id]
                gt_class = track_dict[gt_id]

                score_dict = {}
                for cls in classes:
                    s = cls[1]
                    c = cls[0]
                    if c not in score_dict:
                        score_dict[c] = 0
                    score_dict[c] += s
                sorted_score = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
                highest_score_class = sorted_score[0][0]
                highest_score = sorted_score[0][1]
                if highest_score_class == gt_class:
                    correct += 1
                num += 1
    print(f'Accuracy: {correct / num}')
