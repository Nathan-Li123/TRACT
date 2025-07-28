import os
import json
import tqdm
import torch
import pickle


def main():
    with open(path, 'r') as f:
        data = json.load(f)
    with open(tao_path, 'r') as f:
        tao_data = json.load(f)
    # breakpoint()
    image_dict = {}
    for image in tao_data['images']:
        image_dict[image['id']] = image['file_name']
    
    out_dict = {}
    for det in tqdm.tqdm(data, desc='Processing detections'):
        image_id = det['image_id']
        if image_id not in out_dict:
            out_dict[image_id] = {'det_bboxes': [], 'det_labels': []}
        det_score = det['score']
        bbox = det['bbox']
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        bbox.append(det_score)
        bbox = torch.tensor(bbox)
        out_dict[image_id]['det_bboxes'].append(bbox)

        label = torch.tensor(det['category_id'] - 1)
        out_dict[image_id]['det_labels'].append(label)
    
    for image_id, det in tqdm.tqdm(out_dict.items(), desc='Saving detections'):
        file_name = image_dict[image_id]
        subset, video_name, frame_name = file_name.split('/')
        # _, subset, video_name, frame_name = file_name.split('/')
        subset_root = os.path.join(out_path, subset)
        os.makedirs(subset_root, exist_ok=True)
        video_root = os.path.join(subset_root, video_name)
        os.makedirs(video_root, exist_ok=True)
        frame_path = os.path.join(video_root, frame_name.replace('.jpg', '.pth').replace('.jpeg', '.pth').replace('.JPEG', '.pth'))
        with open(frame_path, 'wb') as f:
            pickle.dump(det, f)


if __name__ == '__main__':
    tao_path = '/data3/AttrTrack/ovtrack/data/ovt-b/annotations/ovtb_ann.json'
    path = '/data3/AttrTrack/ovtrack/results/ovtrack_teta_results_ovtb/tao_bbox.json'
    out_path = 'results/public_dets/ovtb_dets/ovtrack_ovtb_det'
    os.makedirs(out_path, exist_ok=True)
    # out_path = os.path.join(out_path, 'test')
    # os.makedirs(out_path, exist_ok=True)
    main()      
