import json


if __name__ == '__main__':
    path = 'data/tao/annotations/tao_val_lvis_v1_classes.json'
    with open(path, 'r') as f:
        tao_data = json.load(f)
    novel, base  = [], []
    cates = {}
    for i in range(len(tao_data['categories'])):
        cates[tao_data['categories'][i]['id']] = tao_data['categories'][i]['name']
        if tao_data['categories'][i]['frequency'] == 'r':
            novel.append(tao_data['categories'][i]['id'])
        else:
            base.append(tao_data['categories'][i]['id'])
    print('Novel:', len(novel))
    print('Base:', len(base))

    images = {}
    for image in tao_data['images']:
        images[image['id']] = image['file_name']

    for anno in tao_data['annotations']:
        if anno['category_id'] in novel:
            print(images[anno['image_id']])
            print(cates[anno['category_id']])
            print(anno['bbox'])
            