import os
import json
import tqdm
import numpy as np


def main():
    with open(classify_path, 'r') as f:
        classify_data = json.load(f)
    with open(track_path, 'r') as f:
        track_data = json.load(f)
    # assert classify_data.keys() == track_data.keys()
    track_ids = list(track_data.keys())
    annos = []
    for track_id in track_ids:
        
        if FILTER_ID and track_id not in filter_ids:
            cate_id = track_data[track_id]['category']
            # continue
        else:
            if track_id not in classify_data:
                cate_id = track_data[track_id]['category']
            else:
                classify_score = classify_data[track_id]['score']
                if 'proportion' not in track_data[track_id]:
                    cate_id = classify_data[track_id]['class']
                else:
                    track_prop = track_data[track_id]['proportion']
                    if track_prop > classify_score:
                        cate_id = track_data[track_id]['category']
                    else:
                        cate_id = classify_data[track_id]['class']

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
    with open(out_path, 'w') as f:
        json.dump(annos, f)


FILTER_ID = False

if __name__ == '__main__':
    classify_path = 'outputs/attention_cosine_lvis+tao/results_test_yolow_xl_masa.json'
    track_path = 'datasets/tao_tracklets/test_yolow_xl_masa_mix.json'
    out_path = 'outputs/attention_cosine_lvis+tao/tmp.json'
 
    if FILTER_ID:
        filter_path = 'outputs/masa_gdino_filter_ids.json'
        with open(filter_path, 'r') as f:
            filter_ids = json.load(f)
    main()
