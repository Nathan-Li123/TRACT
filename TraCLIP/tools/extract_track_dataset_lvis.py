import os
import json
import tqdm

import numpy as np


def main():
    root = '/data3/resources/LVIS'
    train_img_dir = os.path.join(root, 'train2017')
    # val_img_dir = os.path.join(root, 'val2017')
    label_path = os.path.join(root, 'lvis_v1_train+coco_mask_v1_base.json')
    out_path = 'datasets/lvis_tracklets/train.json'
    class_names = np.loadtxt('datasets/class_names/lvis_base_class_names.txt', dtype=str, delimiter='?').tolist()

    with open(label_path, 'r') as f:
        data = json.load(f)
    print('Data loaded')
    
    image_dict = {}
    for image in data['images']:
        image_dict[image['id']] = image['file_name']
    cate_dict = {}
    for cate in data['categories']:
        cate_dict[cate['id']] = cate['name']
    
    num = 1
    out_dict = {}
    for anno in tqdm.tqdm(data['annotations']):
        image_id = anno['image_id']
        image_path = os.path.join(train_img_dir, image_dict[image_id])
        bbox = anno['bbox']
        bbox = [int(i) for i in bbox]
        if bbox[2] < 1 or bbox[3] < 1:
            continue
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        cate_id = anno['category_id']
        cate_name = cate_dict[cate_id]
        cate_name = cate_name.replace('speaker_(stero_equipment)', 'speaker_(stereo_equipment)')
        if cate_name not in class_names:
            continue
        cate_id = class_names.index(cate_name)
        out = {'category': cate_id, "tracklet": [{"image_path": image_path, "bbox": bbox}]}
        out_dict[str(num)] = out
        num += 1
    with open(out_path, 'w') as f:
        json.dump(out_dict, f, indent=4)
    print('Data written')


if __name__ == '__main__':
    main()
