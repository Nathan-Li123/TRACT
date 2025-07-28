import os
import json
import tqdm


if __name__ == '__main__':
    path = 'results/masa_results/masa-yolow-ovtb_0.3/tao_track.json'
    ann_path = 'data/ovt-b/annotations/ovtb_ann.json'
    out_root = 'results/masa_results/masa-yolow-ovtb_0.3/mot'
    os.makedirs(out_root, exist_ok=True)

    with open(path, 'r') as f:
        data = json.load(f)
    with open(ann_path, 'r') as f:
        ann = json.load(f)
    
    image_dict = {}
    for image in ann['images']:
        image_dict[image['id']] = (image['file_name'], image['frame_id'])
    
    for re in tqdm.tqdm(data):
        image_id = re['image_id']
        image_name, frame_id = image_dict[image_id]
        subset, video, frame = image_name.split('/')
        subset_dir = os.path.join(out_root, subset)
        os.makedirs(subset_dir, exist_ok=True)
        video_path = os.path.join(subset_dir, video + '.txt')
        with open(video_path, 'a') as f:
            f.write(f'{frame_id+1},{re["track_id"]},{re["bbox"][0]},{re["bbox"][1]},{re["bbox"][2]},{re["bbox"][3]},1,-1,-1,-1\n')
