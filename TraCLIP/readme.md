##### 运行步骤
1. 进行追踪结果和gt的匹配，运行match_tracks.py
2. 使用extract_track_dataset_pred(gt).py文件提取数据集，过程中需要用到的信息包括gt、pred以及两者的match结果
3. 运行train或者val
4. 使用transfer_tracklets_to_teta.py将results文件转换为teta格式

##### 文件记录
datasets/tao_tracklets/train.json：TAO训练集gt，使用1203个类别标签
datasets/tao_tracklets/validation.json：TAO验证集gt，使用1203个类别标签
datasets/tao_tracklets/validation_base.json：TAO验证集gt，使用866个base类标签
datasets/tao_tracklets/validation_ovtrack.json：TAO验证集的OVTrack追踪结果，使用1203个类别标签
datasets/tao_tracklets/validation_yolow_l_0.3.json：TAO验证集YOLOW-l+OCSORT追踪结果，追踪阈值0.3，使用296个标签
datasets/tao_tracklets/validation_yolow_l_1203.json：TAO验证集YOLOW-l+OCSORT追踪结果，追踪阈值0.1，使用1203个标签
datasets/tao_tracklets/validation_yolow_l_1230.json：TAO验证集YOLOW-l+OCSORT追踪结果，追踪阈值0.1，其中YOLOW使用的额是早期1230个类别训练的版本（暂时在HOTA指标上效果最好），使用的是1203个类别标签而不是文件名中的1230
datasets/tao_tracklets/validation_yolow_l.json：TAO验证集YOLOW-l+OCSORT追踪结果，追踪阈值0.1，使用296个标签
datasets/tao_tracklets/validation_ovtrack_l2.json：TAO验证集的OVTrack追踪结果，使用1203个类别标签，筛去了轨迹长度低于3的轨迹

datasets/class_names/lvis_05_class_names.txt：1230个类别
datasets/class_names/lvis_class_names.txt：1203个类别
datasets/class_names/lvis_base_class_names.txt：866个类别
datasets/class_names/lvis_class_names_ovtao_val.txt：TAO验证集中包含的296个类别

datasets/matches/match_yolow_l.json：TAO验证集YOLOW-l+OCSORT追踪结果，追踪阈值0.1
datasets/matches/match_yolow_l_0.3.json：TAO验证集YOLOW-l+OCSORT追踪结果，追踪阈值0.3
datasets/matches/match_yolow_l_0.05.json：TAO验证集YOLOW-l+OCSORT追踪结果，追踪阈值0.05
datasets/matches/match_yolow_l_1230.json：
