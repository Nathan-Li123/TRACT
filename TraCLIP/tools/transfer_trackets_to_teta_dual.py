import os
import json
import tqdm
import numpy as np


def main():
    with open(classify_path_1, 'r') as f:
        classify_data_1 = json.load(f)
    with open(classify_path_2, 'r') as f:
        classify_data_2 = json.load(f)
    with open(track_path, 'r') as f:
        track_data = json.load(f)
    # assert classify_data.keys() == track_data.keys()
    track_ids = list(track_data.keys())
    annos = []
    for track_id in track_ids:
        
        if USE_TRACT_ONLY:
            cate_id = track_data[track_id]['category']
        elif FILTER_ID and track_id not in filter_ids:
            cate_id = track_data[track_id]['category']
            # continue
        else:
            if track_id not in classify_data_1 and track_id not in classify_data_2:
                cate_id = track_data[track_id]['category']
            elif track_id in classify_data_1 and track_id in classify_data_2:
                classify_score_1 = classify_data_1[track_id]['score']
                classify_score_2 = classify_data_2[track_id]['score']
                if classify_score_1 > classify_score_2:
                    classify_score = classify_data_1[track_id]['score']
                    classify_result = classify_data_1[track_id]['class']
                else:
                    classify_score = classify_data_2[track_id]['score']
                    classify_result = classify_data_2[track_id]['class']
                track_prop = track_data[track_id]['proportion'] * 1.5 if 'proportion' in track_data[track_id] else 1
                if track_prop > classify_score:
                    cate_id = track_data[track_id]['category']
                else:
                    cate_id = classify_result
            elif track_id in classify_data_1:
                track_prop = track_data[track_id]['proportion'] * 1.5 if 'proportion' in track_data[track_id] else 1
                if track_prop > classify_data_1[track_id]['score']:
                    cate_id = track_data[track_id]['category']
                else:
                    cate_id = classify_data_1[track_id]['class']
            elif track_id in classify_data_2:
                track_prop = track_data[track_id]['proportion'] * 1.5 if 'proportion' in track_data[track_id] else 1
                if track_prop > classify_data_2[track_id]['score']:
                    cate_id = track_data[track_id]['category']
                else:
                    cate_id = classify_data_2[track_id]['class']

        track_info = track_data[track_id]
        tracklet = track_info['tracklet']
        # cate_id = track_info['category']
        for i in range(len(tracklet)):
            bbox = tracklet[i]['bbox']
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            if width < 1 or height < 1:
                continue
            annos.append({
                'track_id': int(track_id),
                'video_id': int(track_info['video_id']),
                'category_id': int(cate_id) + 1,
                'score': float(tracklet[i]['score']),
                'bbox': [x1, y1, width, height],
                'image_id': int(tracklet[i]['image_id'])
            })
    print('annos:', len(annos))
    with open(out_path, 'w') as f:
        json.dump(annos, f)


FILTER_ID = False
USE_TRACT_ONLY = False

if __name__ == '__main__':
    classify_path_1 = 'outputs/attention_cosine_lvis+tao/results.json'
    classify_path_2 = 'outputs/attention_cosine_lvis+tao/results_attr.json'
    track_path = 'datasets/tao_tracklets/ovtb_ovtrack_masa.json'
    out_path = 'outputs/attention_cosine_lvis+tao/tmp.json'
 
    print(classify_path_1)
    if FILTER_ID:
        filter_path = 'outputs/masa_gdino_filter_ids.json'
        with open(filter_path, 'r') as f:
            filter_ids = json.load(f)
    main()
