import os
import json
import tqdm
import torch
import pickle


def main():
    # Define paths
    # tao_path = '/data3/resources/TAO/annotations/validation_ours_v1.json'
    # tao_path = '/data3/resources/TAO/annotations/tao_test_burst_v1.json'
    # json_path = '/data3/AttrTrack/AED/data/detections/TAO_RegionCLIP_test.json'
    # out_root = 'results/public_dets/tao_test_dets/regionclip_tao_test_det'

    tao_path = '/data3/resources/TAO/annotations/tao_test_burst_v1.json'
    json_path = '/data3/AttrTrack/AED/data/detections/TAO_RegionCLIP_test.json'
    out_root = 'results/public_dets/tao_test_dets/regionclip_tao_test_det'

    os.makedirs(out_root, exist_ok=True)

    with open(tao_path, 'r') as f:
        tao_data = json.load(f)
    image_dict = {}
    for image in tao_data['images']:
        image_dict[image['id']] = image['file_name']

    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    out_dict = {}
    for det in tqdm.tqdm(data, desc='Processing detections'):
        image_id = det['image_id']
        if image_id not in out_dict:
            out_dict[image_id] = {'det_bboxes': [], 'det_labels': []}
        det_score = det['score'] / 10
        # if det_score < 0.001:
        #     continue
        bbox = det['bbox']
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        bbox.append(det_score)
        bbox = torch.tensor(bbox)
        out_dict[image_id]['det_bboxes'].append(bbox)

        label = torch.tensor(det['category_id'] - 1)
        if label < 0 or label > 1202:
            raise ValueError(label)
        out_dict[image_id]['det_labels'].append(label)

    for det in tqdm.tqdm(out_dict.values(), desc='Filtering detections'):
        det['det_bboxes'] = torch.stack(det['det_bboxes'])
        det['det_labels'] = torch.stack(det['det_labels'])
        if len(det['det_bboxes']) > 100:
            top_scores, top_indices = torch.topk(det['det_bboxes'][:, 4], 100)
            det['det_bboxes'] = det['det_bboxes'][top_indices]
            det['det_labels'] = det['det_labels'][top_indices]
    
    out_root = os.path.join(out_root, 'test')
    os.makedirs(out_root, exist_ok=True)
    for image_id, det in tqdm.tqdm(out_dict.items(), desc='Saving detections'):
        file_name = image_dict[image_id]
        _, subset, video_name, frame_name = file_name.split('/')
        subset_root = os.path.join(out_root, subset)
        os.makedirs(subset_root, exist_ok=True)
        video_root = os.path.join(subset_root, video_name)
        os.makedirs(video_root, exist_ok=True)
        frame_path = os.path.join(video_root, frame_name.replace('.jpg', '.pth'))
        with open(frame_path, 'wb') as f:
            pickle.dump(det, f)


if __name__ == '__main__':
    main()