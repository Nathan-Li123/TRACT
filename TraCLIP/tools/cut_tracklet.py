import os
import json
import random
import numpy as np
import PIL.Image as Image


def main():
    with open(anno_path, 'r') as f:
        data = json.load(f)
    tracklets = []
    for track_info in data.values():
        if len(track_info['tracklet']) < 20:
            continue
        tracklets.append(track_info['tracklet'])
    
    tracklets = random.sample(tracklets, 5)
    for i, tracklet in enumerate(tracklets):
        out_path = os.path.join(out_dir, str(i+1).zfill(2))
        os.makedirs(out_path, exist_ok=True)
        for j, obj in enumerate(tracklet):
            image_path = obj['image_path']
            image = Image.open(image_path)
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox
            image = image.crop((x1, y1, x2, y2))
            resized = image.resize((224, 224))
            resized.save(os.path.join(out_path, f'{str(j+1).zfill(2)}.jpg'))


if __name__ == '__main__':
    anno_path = 'datasets/tao_tracklets/validation_yolow_xl_masa.json'
    out_dir = 'outputs/tracklets'
    os.makedirs(out_dir, exist_ok=True)
    main()
