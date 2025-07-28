import os
import json
import random
import tqdm
import numpy as np


def main():
    if FILTER:
        with open(fil_path, 'r') as f:
            fil_data = json.load(f)
    with open(path, 'r') as f:
        data = json.load(f)
    print('annos:', len(data))
    with open(tao_path, 'r') as f:
        tao_data = json.load(f)
    classes = np.loadtxt(class_path, delimiter='?', dtype=str).tolist()
    base_classes = np.loadtxt(base_class_path, delimiter='?', dtype=str).tolist()

    cate_dict = {}
    for cate in tao_data['categories']:
        cate_dict[cate['id']] = cate['name']
        if cate['name'] == 'speaker_(stero_equipment)':
            cate_dict[cate['id']] = 'speaker_(stereo_equipment)'
    
    image_dict = {}
    for image in tao_data['images']:
        image_dict[image['id']] = image['file_name']
    
    track_dict = {}
    for det in tqdm.tqdm(data):
        track_id = det['track_id']
        if FILTER and track_id in fil_data:
            continue
        if track_id not in track_dict:
            cate_name = cate_dict[det['category_id']]
            if cate_name not in classes:
                continue
            cate_id = classes.index(cate_name)
            track_dict[track_id] = {'category': cate_id, 'video_id': det['video_id'], 'tracklet': [], 'ori_cate_ids': []}
        
        bbox = det['bbox']
        if bbox[2] < 1 or bbox[3] < 1:
            continue
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        bbox = [int(i) for i in bbox]
        track_dict[track_id]['tracklet'].append(
            {'image_path': os.path.join('/data3/resources/TAO/frames', image_dict[det['image_id']]), 
             'bbox': bbox,
             'score': det['score'],
             'image_id': det['image_id']}
        )
        track_dict[track_id]['ori_cate_ids'].append(det['original_category_id'])

    del_keys = []
    for k, v in track_dict.items():
        if len(v['tracklet']) == 0:
            del_keys.append(k)
    for k in del_keys:
        del track_dict[k]

    for k, v in track_dict.items():
        ori_cate_ids = v['ori_cate_ids']
        proportion = ori_cate_ids.count(max(ori_cate_ids, key=ori_cate_ids.count)) / len(ori_cate_ids)
        track_dict[k]['proportion'] = proportion

    base_selected_ids = []
    novel_selected_ids = []
    for track in tao_data['tracks']:
        cate_name = cate_dict[track['category_id']]
        if cate_name in base_classes and len(base_selected_ids) < 100:
            if random.random() < 0.1:
                base_selected_ids.append(track['id'])
        elif cate_name not in base_classes and len(novel_selected_ids) < 1:
            if random.random() < 0.1:
                novel_selected_ids.append(track['id'])
    print('base_selected_ids:', base_selected_ids)
    print('novel_selected_ids:', novel_selected_ids)
    num, n = max(list(track_dict.keys())), 1
    gt_tracks, transfer = {}, {}
    for anno in tao_data['annotations']:
        track_id = anno['track_id']
        if track_id not in base_selected_ids and track_id not in novel_selected_ids:
            continue
        if track_id not in transfer:
            transfer[track_id] = num + n
            n += 1
        track_id = transfer[track_id]
        if track_id not in gt_tracks:
            gt_tracks[track_id] = {'category': anno['category_id']-1, 'video_id': anno['video_id'], 'tracklet': [], 'ori_cate_ids': []}
        if len(gt_tracks[track_id]['tracklet']) > 10:
            continue
        if track_id in novel_selected_ids and len(gt_tracks[track_id]['tracklet']) > 5:
            continue
        bbox = anno['bbox']
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        image_id = anno['image_id']
        image_path = os.path.join('/data3/resources/TAO/frames', image_dict[image_id])
        score = 1
        gt_tracks[track_id]['tracklet'].append(
            {'image_path': image_path, 
             'bbox': bbox,
             'score': score,
             'image_id': image_id}
        )
        gt_tracks[track_id]['ori_cate_ids'].append(anno['category_id']-1)
    track_dict.update(gt_tracks)

    print('track num:', len(track_dict))
    with open(out_path, 'w') as f:
        json.dump(track_dict, f)

FILTER = False


if __name__ == '__main__':
    # tao_path = '/data3/resources/TAO/annotations/tao_test_burst_v1.json'
    tao_path = '/data3/resources/TAO/annotations/validation_ours_v1.json'
    if FILTER:
        fil_path = 'datasets/yolow+masa_tao_val_mismatches.json'
    path = '/data3/AttrTrack/masa/results/masa_results/masa-ovtrack-val_tie/tao_track.json'
    out_path = 'datasets/tao_tracklets/validation_ovtrack_masa_mix.json'
    class_path = 'datasets/class_names/lvis_class_names.txt'
    base_class_path = 'datasets/class_names/lvis_base_class_names.txt'
    main()
