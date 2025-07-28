import os
import tqdm
import json
import numpy as np

import warnings
warnings.filterwarnings('ignore')


ROOT = '/data3/InsCap/imgs'


def calculate_iou(box1, box2):
    # 计算交集的坐标范围
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # 计算交集的面积
    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # 计算并集的面积
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area_box1 + area_box2 - intersection_area

    # 计算交并比
    iou = intersection_area / union_area
    return iou


def match_tracks_dir():
    transfer_dict = {}
    gt_path = '/data2/resources/TAO/annotations/validation_ours_v1.json'
    pred_path = '/data2/SMVU/OC_SORT/output/ovtrack_val_ovtao'
    out_path= './datasets/matches/match_ovtrack_oc.json'

    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    name_to_id = {}
    for video in gt_data['videos']:
        name_to_id[video['name']] = video['id']

    gt_tracks_dict = {}
    for tracks in gt_data['tracks']:
        video_id = tracks['video_id']
        if video_id not in gt_tracks_dict:
            gt_tracks_dict[video_id] = {}
    video_dict = {}
    for image in gt_data['images']:
        video_id = image['video_id']
        if video_id not in video_dict:
            video_dict[video_id] = []
        video_dict[video_id].append(image['id'])
    image_dict = {}
    for images in gt_data['images']:
        image_dict[images['id']] = video_dict[images['video_id']].index(images['id']) + 1
    for anno in gt_data['annotations']:
        # gt_tracks_dict[anno['video_id']][anno['track_id']].update({image_dict[anno['image_id']]: anno['bbox']})
        frm_idx = image_dict[anno['image_id']]
        if frm_idx not in gt_tracks_dict[anno['video_id']]:
            gt_tracks_dict[anno['video_id']][frm_idx] = {}
        gt_tracks_dict[anno['video_id']][frm_idx].update({anno['track_id']: anno['bbox']})

    for cate_dir in os.listdir(pred_path):
        cate_dir_path = os.path.join(pred_path, cate_dir)
        if not os.path.isdir(cate_dir_path):
            continue
        seqs = os.listdir(cate_dir_path)
        for seq in tqdm.tqdm(seqs, desc=cate_dir):
            seq_name = 'val/' + cate_dir + '/' + seq.split('.')[0]
            gt_tracks = gt_tracks_dict[name_to_id[seq_name]]

            pred_data = np.loadtxt(os.path.join(cate_dir_path, seq), dtype='str', delimiter=',')
            if len(pred_data) == 0:
                continue
            if pred_data.ndim == 1:
                pred_data = np.expand_dims(pred_data, axis=0)
            match_dict = {}
            for pred_row in pred_data:
                frame_id = int(pred_row[0])
                if frame_id not in gt_tracks:
                    continue
                
                pred_track_id = pred_row[1]
                if pred_track_id not in match_dict:
                    match_dict[pred_track_id] = {}
                pred_box = [float(i) for i in pred_row[2:6]]

                gts = gt_tracks[frame_id]
                for gt_lbl, gt_box in gts.items():
                    if gt_lbl not in match_dict[pred_track_id]:
                        match_dict[pred_track_id][gt_lbl] = 0
                    iou = calculate_iou(gt_box, pred_box)
                    match_dict[pred_track_id][gt_lbl] += iou
            
            seq_transfer_dict = {}
            for pred_id, ious in match_dict.items():
                max_sum, max_id = -100, 0
                for gt_id, sum_iou in ious.items():
                    if sum_iou >= max_sum:
                        max_sum = sum_iou
                        max_id = gt_id
                seq_transfer_dict[pred_id] = max_id
            transfer_dict[seq_name] = seq_transfer_dict
    with open(out_path, 'w') as f:
        json.dump(transfer_dict, f)


if __name__ == '__main__':
    match_tracks_dir()
