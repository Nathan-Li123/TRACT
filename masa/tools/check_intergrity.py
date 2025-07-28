import os
import json
import shutil


if __name__ == '__main__':
    root = 'results/public_dets/tao_val_dets/teta_50_internms/regionclip_tao_val_det'
    tao_path = '/data3/resources/TAO/annotations/validation_ours_v1.json'

    yolow_root = 'results/public_dets/tao_val_dets/teta_50_internms/yolow_xl_tao_val_det'

    with open(tao_path, 'r') as f:
        tao_data = json.load(f)

    # for video in tao_data['videos']:
    #     name = video['name']
    #     path = os.path.join(root, name)
    #     if not os.path.exists(path):
    #         print(path)

    images = []
    for image in tao_data['images']:
        name = image['file_name']
        path = os.path.join(root, name.replace('.jpg', '.pth'))
        if not os.path.exists(path):
            print(path)
            # shutil.copyfile(os.path.join(yolow_root, name.replace('.jpg', '.pth')), path)

    # for subset in os.listdir(os.path.join(root, 'val')):
    #     subset_path = os.path.join(root, 'val', subset)
    #     for seq in os.listdir(subset_path):
    #         seq_path = os.path.join(subset_path, seq)
    #         if len(os.listdir(seq_path)) == 0:
    #             print(seq_path)
