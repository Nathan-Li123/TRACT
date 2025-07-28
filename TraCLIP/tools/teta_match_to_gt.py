import os
import json


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


def main(src_path, dst_path):
    # with open('/data2/resources/TAO/annotations/validation_ours_v1.json', 'r') as f:
    #     tao_data = json.load(f)
    with open('/data3/resources/TAO/annotations/tao_test_burst_v1.json', 'r') as f:
        tao_data = json.load(f)

    with open(src_path, 'r') as f:
        tracklets_data = json.load(f)
    
    gt_dict = {}
    for anno in tao_data['annotations']:
        image_id = anno['image_id']
        bbox = anno['bbox']
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        track_id = anno['track_id']
        if image_id not in gt_dict:
            gt_dict[image_id] = []
        gt_dict[image_id].append({'track_id': track_id, 'bbox': bbox})
    cate_dict = {}
    for track in tao_data['tracks']:
        cate_dict[track['id']] = track['category_id']
    
    track_dict = {}
    for anno in tracklets_data:
        image_id = int(anno['image_id'])
        bbox = anno['bbox']
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        track_id = anno['track_id']
        if track_id not in track_dict:
            track_dict[track_id] = {}
        
        if image_id not in gt_dict:
            continue
        gt_info = gt_dict[image_id]
        for gt in gt_info:
            iou = calculate_iou(gt['bbox'], bbox)
            gt_id = gt['track_id']
            if gt_id not in track_dict[track_id]:
                track_dict[track_id][gt_id] = 0
            track_dict[track_id][gt_id] += iou
    
    filter_ids = [] 
    transfer_dict = {}
    for track_id, track_info in track_dict.items():
        max_iou = -1
        max_id = 0
        for gt_id, iou in track_info.items():
            if iou > max_iou:
                max_iou = iou
                max_id = gt_id
        if max_id == 0:
            filter_ids.append(track_id)
        if max_iou < 0.2:
            filter_ids.append(track_id)
            max_id = 0
        transfer_dict[track_id] = max_id
    
    corrct, num = 0, 0
    for i in range(len(tracklets_data)):
        old_cate_id = tracklets_data[i]['category_id']
        if tracklets_data[i]['track_id'] not in transfer_dict:
            print(f'Warning: track_id {tracklets_data[i]["track_id"]} not in transfer_dict')
            continue
        gt_id = transfer_dict[tracklets_data[i]['track_id']]
        if gt_id == 0:
            num += 1
            continue
        new_cate_id = cate_dict[gt_id]
        if old_cate_id == new_cate_id:
            corrct += 1
            continue
        # tracklets_data[i]['category_id'] = new_cate_id
    print(f'Accuracy: {corrct / len(tracklets_data)}')
    print(f'Warning: {num} tracklets not matched')
    with open(dst_path, 'w') as f:
        json.dump(tracklets_data, f)
    with open(filter_path, 'w') as f:
        json.dump(filter_ids, f)


if __name__ == '__main__':
    src_path = '/data3/AttrTrack/masa/results/masa_results/masa-yolow_xl-test_tie/tao_track.json'
    dst_path = 'datasets/test_yolow_xl.json'
    filter_path = 'datasets/test_yolow_xl_mismatches.json'
    main(src_path, dst_path)
