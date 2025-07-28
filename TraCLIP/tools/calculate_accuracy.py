import os
import json


def main(filter_ids=None):
    with open(results_path, 'r') as f:
        results = json.load(f)
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    correct, n = 0, 0
    base_correct, base_n = 0, 0
    novel_correct, novel_n = 0, 0
    for track_id in results.keys():
        if track_id not in gt:
            continue
        if filter_ids is not None and track_id in filter_ids:
            continue
        # if len(gt[track_id]['tracklet']) < 10:
        #     continue
        if results[track_id] == gt[track_id]['category']:
            correct += 1
            if gt[track_id]['category'] in base_class_ids:
                base_correct += 1
            else:
                novel_correct += 1
        if gt[track_id]['category'] in base_class_ids:
            base_n += 1
        else:
            novel_n += 1
        n += 1
    accuracy = correct / n
    print('Accuracy: {:.8f}'.format(accuracy))
    print('Base Accuracy: {:.8f}'.format(base_correct / base_n))
    print('Novel Accuracy: {:.8f}'.format(novel_correct / novel_n))

FILTER_ID = True


if __name__ == '__main__':
    results_path = 'outputs/attention_cosine_lvis+tao/results.json'
    gt_path = 'datasets/tao_tracklets/validation_masa_gdino_gt.json'

    if FILTER_ID:
        filter_path = 'outputs/masa_gdino_filter_ids.json'
        with open(filter_path, 'r') as f:
            filter_ids = json.load(f)
    else:
        filter_ids = None

    with open('/data2/resources/TAO/annotations/validation_ours_v1.json', 'r') as f:
        tao_data = json.load(f)
    cate_dict = {}
    for cate in tao_data['categories']:
        cate_dict[cate['name']] = cate['id']
    base_classes = ['aerosol_can', 'airplane', 'grocery_bag', 'guitar', 'gun', 'surfboard', 'sweater', 'armchair', 'sword', 'ax', 'hat', 'baby_buggy', 'army_tank', 'handbag', 'backpack', 'suitcase', 'ball', 'teacup', 'hippopotamus', 'telephone', 'horse', 'baseball_bat', 'basketball', 'igniter', 'jacket', 'toothbrush', 'bed', 'cow', 'cover', 'toolbox', 'toothpaste', 'towel', 'bedspread', 'toy', 'kayak', 'tray', 'bicycle', 'trousers', 'truck', 'bird', 'knife', 'knitting_needle', 'turtle', 'umbrella', 'ladle', 'blanket', 'lamp', 'vacuum_cleaner', 'boat', 'laptop_computer', 'lanyard', 'lawn_mower', 'book', 'walking_stick', 'booklet', 'bottle', 'bow_(weapon)', 'lion', 'bowl', 'lizard', 'magazine', 'mallet', 'briefcase', 'broom', 'bucket', 'bull', 'wineglass', 'matchbox', 'microphone', 'bus_(vehicle)', 'cab_(taxi)', 'zebra', 'minivan', 'money', 'monkey', 'camel', 'camera', 'motor_scooter', 'motorcycle', 'can', 'candle', 'canister', 'napkin', 'necklace', 'bottle_cap', 'car_(automobile)', 'notebook', 'carton', 'cat', 'cellular_telephone', 'packet', 'chair', 'paddle', 'chicken_(animal)', 'paper_towel', 'cigarette', 'pen', 'person', 'coffee_table', 'piano', 'pickup_truck', 'pigeon', 'pillow', 'control', 'pistol', 'convertible_(automobile)', 'plate', 'pliers', 'cornet', 'pot', 'cup', 'rabbit', 'curtain', 'racket', 'cylinder', 'rag_doll', 'deer', 'remote_control', 'dish', 'dog', 'drawer', 'dress_hat', 'drone', 'drum_(musical_instrument)', 'sandwich', 'saxophone', 'earphone', 'scissors', 'scraper', 'sculpture', 'refrigerator', 'elephant', 'faucet', 'sheep', 'shirt', 'shoe', 'shoulder_bag', 'fish', 'skateboard', 'flag', 'ski_pole', 'slipper_(footwear)', 'fork', 'spatula', 'spectacles', 'spider', 'sponge', 'spoon', 'squirrel', 'gift_wrap', 'giraffe']
    base_class_ids = [cate_dict[base_class] for base_class in base_classes]
    novel_class_ids = list(set(
        [
            c["id"]
            for c in tao_data["categories"]
            if c['name'] not in base_classes
        ]
    ))
    main(filter_ids)
