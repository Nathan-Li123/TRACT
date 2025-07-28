# 计算轨迹分类精度
import json


if __name__ == '__main__':
    gt_path = '/data2/resources/TAO/annotations/validation.json'
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    cate_dict = {}
    for cate in gt_data['categories']:
        cate_dict[cate['id']] = cate['name']

    track_dict = {}
    for anno in gt_data['annotations']:
        if anno['track_id'] not in track_dict:
            track_dict[anno['track_id']] = cate_dict[anno['category_id']]

    track_transfer_path = './match.json'
    with open(track_transfer_path, 'r') as f:
        track_transfer = json.load(f)
    
    correct, num = 0, 0
    pred_path = './classifications.json'
    with open(pred_path, 'r') as f:
        pred_data = json.load(f)
    for video_name, video_dict in pred_data.items():
        for track_id, cates in video_dict.items():
            gt_track = track_transfer['val/' + video_name]
            if track_id not in gt_track:
                continue
            gt_id = gt_track[track_id]
            cate = track_dict[gt_id]
            if cate in cates:
                correct += 1
            num += 1
    print(f'Accuracy: {correct / num:.4f}')
