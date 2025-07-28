import os
import json
import tqdm
import torch
import pickle


if __name__ == '__main__':
    # path = 'results/public_dets/tao_test_dets/ovtrack_tao_test_det/test/ArgoVerse/0f0d7759-fa6e-3296-b528-6c862d061bdd/ring_front_center_315974292036080872.pth'
    # with open(path, 'rb') as f:
    #     data = pickle.load(f)
    #     breakpoint()

    path = 'data/tao/annotations/tao_val_lvis_v1_classes.json'
    with open(path, 'r') as f:
        tao_data = json.load(f)
    breakpoint()

    # path = '/data3/AttrTrack/masa/results/public_dets/tao_val_dets/teta_50_internms/ovtrack_tao_val_det/val/LaSOT/car-4/00000571.pth'
    # with open(path, 'rb') as f:
    #     data = pickle.load(f)
    # det_bboxes = data['det_bboxes']
    # for i, bbox in enumerate(det_bboxes):
    #     text = f'1,{i},{bbox[0]},{bbox[1]},{bbox[2] - bbox[0]},{bbox[3] - bbox[1]},{bbox[4]},-1,-1,-1'
    #     print(text)

    # path = 'data/tao/annotations/tao_test_burst_v1.json'
    # with open(path, 'r') as f:
    #     tao_data = json.load(f)
    # for i in range(len(tao_data['categories'])):
    #     if tao_data['categories'][i]['name'] == 'speaker_(stero_equipment)':
    #         tao_data['categories'][i]['name'] = 'speaker_(stereo_equipment)'
    # with open(path, 'w') as f:
    #     json.dump(tao_data, f)

    # path = '/data3/AttrTrack/ovtrack/data/ovt-b/annotations/ovtb_ann.json'
    # with open(path, 'r') as f:
    #     data = json.load(f)
    # outs = []
    # for image in data['images']:
    #     postfix = image['file_name'].split('.')[-1]
    #     if postfix not in outs:
    #         outs.append(postfix)
    # print(outs)
