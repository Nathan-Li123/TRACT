import os
import json
import tqdm

import numpy as np


PART_CLASSES = False


def main():
    track_transfer_path = './datasets/matches/match_yolow_xl_oc.json'
    out_path = 'datasets/tao_tracklets/validation_oyolow_xl_oc.json'
    pred_path= '/data2/SMVU/OC_SORT/output/yolow_xl_val_ovtao'
    
    if PART_CLASSES:
        cate_names_path = 'datasets/class_names/lvis_class_names_ovtao_val.txt'

    with open('/data2/resources/TAO/annotations/validation_ours_v1.json', 'r') as f:
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
    gt_track_dict = {}
    for track in tao_data['tracks']:
        # 这里需要检测器输入的类别是LVIS的1203个类别
        # id需要-1是因为标注的类别是从1开始的
        gt_track_dict[str(track['id'])] = track['category_id'] - 1

    with open(track_transfer_path, 'r') as f:
        track_transfer = json.load(f)

    video_name_to_id = {}
    for video in tao_data['videos']:
        video_name_to_id[video['name']] = video['id']
    image_name_to_id = {}
    for image in tao_data['images']:
        image_name_to_id[image['file_name']] = image['id']
    if PART_CLASSES:
        cate_id_to_name = {}
        for cate in tao_data['categories']:
            cate_id_to_name[cate['id']] = cate['name']
        cate_names = list(np.loadtxt(cate_names_path, dtype=str, delimiter='?'))

    miss_num = 0
    sub_sets = os.listdir(pred_path)
    track_dict = {}
    track_num = 0
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
            if vid_name not in track_transfer:
                miss_num += 1
                continue
            gt_track = track_transfer[vid_name]
            track_ids = []
            for row in pred_data:
                frame_id = int(row[0])
                track_id = row[1]
                if track_id not in gt_track:
                    continue
                if track_id not in track_ids:
                    track_ids.append(track_id)
                global_track_id = track_num + track_ids.index(track_id)
                try:
                    bbox = [int(float(i)) for i in row[2:6]]
                except:
                    print(vid_name)
                    print(row)
                    continue
                if bbox[2] < 1 or bbox[3] < 1:
                    continue
                bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                img_path = os.path.join('/data2/resources/TAO', index_dict[vid_name][str(frame_id)])

                gt_id = str(gt_track[track_id])
                gt_cate_id = gt_track_dict[gt_id]
                if global_track_id not in track_dict:
                    if PART_CLASSES:
                        gt_cate_name = cate_id_to_name[gt_cate_id+1]
                        gt_cate_id = cate_names.index(gt_cate_name)
                    track_dict[global_track_id] = {'category': gt_cate_id, 
                                                   'tracklet': [],
                                                   'video_id': video_name_to_id[vid_name]}
                track_dict[global_track_id]['tracklet'].append({'image_path': img_path, \
                                                                'bbox': bbox,
                                                                'score': float(row[6]),
                                                                'image_id': image_name_to_id[index_dict[vid_name][str(frame_id)]]})
            track_num += len(track_ids)
    with open(out_path, 'w') as f:
        json.dump(track_dict, f)
    print(f'missing {miss_num} videos')


if __name__ == '__main__':
    main()
