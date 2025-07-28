import os
import json
import tqdm

import numpy as np


def main():
    root = '/data2/resources/TAO'
    path = '/data2/resources/TAO/annotations/validation_ours_v1.json'
    class_names_path = 'datasets/class_names/lvis_base_class_names.txt'
    class_names = np.loadtxt(class_names_path, delimiter='?', dtype=str).tolist()

    with open(path, 'r') as f:
        data = json.load(f)
    # import pdb; pdb.set_trace()
    image_dict = {}
    for image in data['images']:
        image_id = image['id']
        image_dict[image_id] = {'file_name': image['file_name']}
    track_dict = {}
    cate_dict = {}
    for cate in data['categories']:
        cate_dict[cate['id']] = cate['name']
    for track in data['tracks']:
        track_id = track['id']
        cate_id = track['category_id']
        track_dict[track_id] = cate_id
    out_dict = {}
    cate_ids = []
    for anno in tqdm.tqdm(data['annotations'], desc='Processing annotations'):
        bbox = anno['bbox']
        track_id = anno['track_id']
        image_id = anno['image_id']
        cate_id = anno['category_id']
        cate_name = cate_dict[cate_id]
        if cate_name not in class_names:
            continue
        cate_id = class_names.index(cate_name)
        cate_ids.append(cate_id)
        file_name = image_dict[image_id]['file_name']

        image_path = os.path.join(root, file_name)
        # img = Image.open(image_path)
        bbox = [int(i) for i in bbox]
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        # cropped_img = img.crop(bbox)
        if track_id not in out_dict:
            out_dict[track_id] = {'category': cate_id, 'tracklet': []}
        else:
            assert out_dict[track_id]['category'] == cate_id
        out_dict[track_id]['tracklet'].append({'image_path': image_path, 'bbox': bbox})
    print('used class num:', str(len(list(set(cate_ids)))))
    out_path = 'datasets/tao_tracklets/validation_base.json'
    with open(out_path, 'w') as f:
        json.dump(out_dict, f)


if __name__ == '__main__':
    main()
