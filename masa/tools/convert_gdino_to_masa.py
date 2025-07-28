import os
import json
import tqdm
import torch
import pickle

import numpy as np


def main():
    with open(path, 'r') as f:
        data = json.load(f)
    with open(tao_path, 'r') as f:
        tao_data = json.load(f)
    # breakpoint()
    image_dict = {}
    for video in tao_data['videos']:
        image_dict[video['name']] = {'video_id': video['id'], 'frames': []}
    for image in tao_data['images']:
        video_name = image['video']
        image_dict[video_name]['frames'].append(image['id'])

    id_to_name = {}
    for image in tao_data['images']:
        id_to_name[image['id']] = image['file_name']
    
    out_dict = {}
    for seq_name, seq_data in tqdm.tqdm(data.items(), desc='Processing detections'):
        image_ids = image_dict[seq_name]['frames']
        for i in range(len(seq_data)):
            image_data = seq_data[i]
            image_id = image_ids[i]
            if image_id not in out_dict:
                out_dict[image_id] = {'det_bboxes': [], 'det_labels': []}
            bboxes = np.array(image_data['boxes'])
            scores = image_data['scores']
            if len(bboxes) == 0:
                bboxes = np.array([[0, 0, 0, 0]])
                scores = np.array([0])
            if bboxes.ndim == 1:
                bboxes = np.expand_dims(bboxes, axis=0)
            bboxes = bboxes.tolist()
            det_bboxes = [bbox + [score] for bbox, score in zip(bboxes, scores)]

            det_bboxes = torch.tensor(det_bboxes)
            det_labels = torch.ones_like(det_bboxes[:, 0], dtype=int)
            out_dict[image_id]['det_bboxes'] = det_bboxes
            out_dict[image_id]['det_labels'] = det_labels
    
    for image_id, det in tqdm.tqdm(out_dict.items(), desc='Saving detections'):
        file_name = id_to_name[image_id]
        # subset, video_name, frame_name = file_name.split('/')
        _, subset, video_name, frame_name = file_name.split('/')
        subset_root = os.path.join(out_path, subset)
        os.makedirs(subset_root, exist_ok=True)
        video_root = os.path.join(subset_root, video_name)
        os.makedirs(video_root, exist_ok=True)
        frame_path = os.path.join(video_root, frame_name.replace('.jpg', '.pth').replace('.jpeg', '.pth').replace('.JPEG', '.pth'))
        with open(frame_path, 'wb') as f:
            pickle.dump(det, f)


if __name__ == '__main__':
    tao_path = 'data/tao/annotations/tao_val_lvis_v1_classes.json'
    path = 'results/masa_results/gdino_lasot_15_with_scores.json'
    out_path = 'results/public_dets/lasot_dets/gdino_lasot_det'
    os.makedirs(out_path, exist_ok=True)
    # out_path = os.path.join(out_path, 'test')
    # os.makedirs(out_path, exist_ok=True)
    main()      
