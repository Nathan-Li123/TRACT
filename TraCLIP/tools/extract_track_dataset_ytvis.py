import os
import json
import tqdm


def main():
    anno_path = os.path.join(root, 'instances.json')
    with open(anno_path, 'r') as f:
        data = json.load(f)
    
    out_dict = {}
    video_dict = {}
    for video in data['videos']:
        video_dict[video['id']] = video['file_names']
    
    for anno in tqdm.tqdm(data['annotations']):
        track_id = anno['id']
        video_id = anno['video_id']
        bboxes = anno['bboxes']
        file_names = video_dict[video_id]

        tmp = {
            'category': anno['category_id'] - 1,
            'tracklet': []
        }
        assert len(bboxes) == len(file_names)
        for i in range(len(bboxes)):
            if bboxes[i] is None:
                continue
            image_path = os.path.join(root, 'JPEGImages', file_names[i])
            if not os.path.exists(image_path):
                continue
            box = bboxes[i]
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            tmp['tracklet'].append({
                'image_path': image_path,
                'bbox': box
            })
        if len(tmp['tracklet']) > 0:
            out_dict[track_id] = tmp
    with open(out_path, 'w') as f:
        json.dump(out_dict, f)


if __name__ == '__main__':
    root = '/data2/resources/YouTube-vis/train'
    out_path = 'datasets/ytvis_tracklets/train.json'
    main()
