import os
import json


if __name__ == '__main__':
    path = '/data2/resources/YouTube-vis/train/instances.json'
    out_path = 'datasets/class_names/ytvis_class_names.txt'
    
    with open(path, 'r') as f:
        data = json.load(f)

    with open(out_path, 'w') as f:
        for cate in data['categories']:
            f.write(cate['name'] + '\n')     
