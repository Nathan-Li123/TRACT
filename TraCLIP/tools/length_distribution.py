import json
import tqdm


if __name__ == '__main__':
    path = 'datasets/tao_tracklets/validation_masa_gdino.json'
    with open(path, 'r') as f:
        data = json.load(f)

    track_length_counts = {}
    for track_info in tqdm.tqdm(data.values()):
        track_len = len(track_info['tracklet'])
        if track_len in track_length_counts:
            track_length_counts[track_len] += 1
        else:
            track_length_counts[track_len] = 1

    length_1 = 0
    length_2_5 = 0
    length_6_10 = 0
    length_gt_10 = 0

    for length, count in track_length_counts.items():
        if length == 1:
            length_1 += count
        elif 2 <= length <= 5:
            length_2_5 += count
        elif 6 <= length <= 10:
            length_6_10 += count
        else:
            length_gt_10 += count

    print(f"Track length 1: {length_1} occurrences")
    print(f"Track length 2-5: {length_2_5} occurrences")
    print(f"Track length 6-10: {length_6_10} occurrences")
    print(f"Track length >10: {length_gt_10} occurrences")
