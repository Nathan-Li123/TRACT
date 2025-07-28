import os
import json
import tqdm
import numpy as np
from collections import defaultdict

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area
    return iou

def match_tracks(gt_data, tracklets_data):
    gt_tracks_dict = {tracks['video_id']: {} for tracks in gt_data['tracks']}
    video_dict = {image['video_id']: [] for image in gt_data['images']}
    for image in gt_data['images']:
        video_dict[image['video_id']].append(image['id'])
    image_dict = {images['id']: video_dict[images['video_id']].index(images['id']) + 1 for images in gt_data['images']}
    for anno in gt_data['annotations']:
        frm_idx = image_dict[anno['image_id']]
        if frm_idx not in gt_tracks_dict[anno['video_id']]:
            gt_tracks_dict[anno['video_id']][frm_idx] = {}
        gt_tracks_dict[anno['video_id']][frm_idx].update({anno['track_id']: anno['bbox']})

    transfer_dict = {}
    for track_id, tracklet in tqdm.tqdm(tracklets_data.items(), desc='matching...'):
        video_id = tracklet['video_id']
        gt_tracks = gt_tracks_dict[video_id]
        match_dict = {}
        for frame in tracklet['tracklet']:
            frame_id = image_dict[frame['image_id']]
            if frame_id not in gt_tracks:
                continue
            pred_box = frame['bbox']
            gts = gt_tracks[frame_id]
            for gt_lbl, gt_box in gts.items():
                gt_box = [gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]]
                if gt_lbl not in match_dict:
                    match_dict[gt_lbl] = 0
                iou = calculate_iou(gt_box, pred_box)
                match_dict[gt_lbl] += iou
        max_sum, max_id = -100, 0
        for gt_id, sum_iou in match_dict.items():
            if sum_iou >= max_sum:
                max_sum = sum_iou
                max_id = gt_id
        if max_id == 0:
            continue
        transfer_dict[track_id] = max_id
    return transfer_dict


def teta_to_tracklets(teta_file, tracklets_file, match=False, filter_len=None):
    # 读取teta格式的JSON文件
    with open(teta_file, 'r') as f:
        teta_data = json.load(f)
    
    # 创建一个字典来存储tracklets数据
    tracklets_data = defaultdict(lambda: {'tracklet': []})
    
    num = 0
    # 遍历teta数据，组织tracklets格式
    for obj in tqdm.tqdm(teta_data, desc='formatting data...'):
        score = obj['score']
        # if score < 0.0005:
        #     continue
        if obj['bbox'][2] < 1 or obj['bbox'][3] < 1:
            continue
        track_id = str(obj['track_id'])
        tracklets_data[track_id]['video_id'] = obj['video_id']
        tracklets_data[track_id]['category'] = int(obj['category_id']) - 1
        tracklets_data[track_id]['tracklet'].append({
            'bbox': [
                obj['bbox'][0],
                obj['bbox'][1],
                obj['bbox'][0] + obj['bbox'][2],
                obj['bbox'][1] + obj['bbox'][3]
            ],
            'score': score,
            'image_path': image_dict[obj['image_id']],
            'image_id': obj['image_id'],
            'ori_category': obj['original_category_id']
        })
        num += 1
    print(num)

    if filter_len is not None:
        out_tracklets_data = {}
        for key, value in tracklets_data.items():
            if len(value['tracklet']) >= filter_len:
                out_tracklets_data[key] = value
        tracklets_data = out_tracklets_data

    if match:
        # 进行和gt的匹配
        transfer_dict = match_tracks(tao_data, tracklets_data)
        
        gt_dict = {}
        for track in tao_data['tracks']:
            gt_dict[track['id']] = track['category_id']

        correct, n = 0, 0
        for track_id in tracklets_data.keys():
            if track_id not in transfer_dict:
                continue
            if tracklets_data[track_id]['category'] == gt_dict[transfer_dict[track_id]]:
                correct += 1
            n += 1
            tracklets_data[track_id]['category'] = gt_dict[transfer_dict[track_id]]
        print(f'Accuracy: {correct / len(transfer_dict)}')
    
    # 将组织好的数据写入tracklets格式的JSON文件
    with open(tracklets_file, 'w') as f:
        json.dump(tracklets_data, f, indent=4)


if __name__ == '__main__':
    with open('/data2/resources/TAO/annotations/validation_ours_v1.json', 'r') as f:
        tao_data = json.load(f)
    image_dict = {}
    for image in tao_data['images']:
        image_dict[image['id']] = os.path.join('/data2/resources/TAO', image['file_name'])

    src_path = '/data3/AttrTrack/masa/results/masa_results/masa-groundingdino-release-ovmot-test/tao_track.json'
    dst_path = 'datasets/tao_tracklets/validation_masa_gdino_gt.json'
    # 示例调用
    # teta_to_tracklets(src_path, dst_path, match=False, filter_len=5)
    teta_to_tracklets(src_path, dst_path, match=True)
