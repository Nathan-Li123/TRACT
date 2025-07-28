import json
import tqdm
from collections import Counter

def main():
    with open(path, 'r') as f:
        data = json.load(f)
    
    track_ids = []
    for track_id, track in tqdm.tqdm(list(data.items())):
        boxes = track['tracklet']
        if len(boxes) < 5:
            continue
        cates = []
        for box in boxes:
            cates.append(box['ori_category'])
        # 使用Counter来统计每个类别的出现次数
        cate_counter = Counter(cates)
        most_common_cate, most_common_count = cate_counter.most_common(1)[0]
        
        # 计算该类别的占比
        total_cates = len(cates)
        proportion = most_common_count / total_cates if total_cates > 0 else 0
        if proportion < 0.3:
            track_ids.append(track_id)
    print(len(track_ids))
    with open(out_path, 'w') as f:
        json.dump(track_ids, f)


if __name__ == '__main__':
    path = 'datasets/tao_tracklets/validation_masa_gdino_gt.json'
    out_path = 'outputs/masa_gdino_filter_ids.json'
    main()