import os
import json
import tqdm
import numpy as np


def main():
    if FILTER:
        with open(fil_path, 'r') as f:
            fil_data = json.load(f)
    with open(path, 'r') as f:
        data = json.load(f)
    with open(tao_path, 'r') as f:
        tao_data = json.load(f)
    classes = np.loadtxt(class_path, delimiter='?', dtype=str).tolist()

    cate_dict = {}
    for cate in tao_data['categories']:
        cate_dict[cate['id']] = cate['name']
        # if cate['name'] == 'speaker_(stero_equipment)':
        #     cate_dict[cate['id']] = 'speaker_(stereo_equipment)'
    
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
            {'image_path': os.path.join('/data2/resources/OVT-B/OVT-B', image_dict[det['image_id']]), 
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

    print('track num:', len(track_dict))
    with open(out_path, 'w') as f:
        json.dump(track_dict, f)

FILTER = False


if __name__ == '__main__':
    # tao_path = '/data3/resources/TAO/annotations/tao_test_burst_v1.json'
    # tao_path = '/data3/resources/TAO/annotations/validation_ours_v1.json'
    tao_path = '/data2/resources/OVT-B/ovtb_ann.json'
    if FILTER:
        fil_path = 'datasets/yolow+masa_tao_val_mismatches.json'
    path = '/data3/AttrTrack/masa/results/masa_results/masa-ovtrack-ovtb_tcr/tao_track.json'
    out_path = 'datasets/tao_tracklets/ovtb_ovtrack_masa.json'
    class_path = 'datasets/class_names/ovtb_class_names.txt'
    main()
