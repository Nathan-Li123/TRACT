import timm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import clip
import tqdm
import json
import numpy as np

from torchvision import transforms
from models.data.track_dataset import TrackletDataset
from models.track_classifier import TraCLIP

import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

# 训练函数
def train(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(tqdm.tqdm(train_loader)):
        input_batch = (inputs.to(device), labels.to(device))
        
        optimizer.zero_grad()
        loss = model(input_batch).mean()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        # if (i + 1) % 100 == 0:
        #     print(f'Iter {i+1}/{len(train_loader)} Loss: {running_loss / (i+1):.4f}')

    epoch_loss = running_loss / len(train_loader)
    print(f'Train Loss: {epoch_loss:.4f}')

# 测试函数
def test(model, test_loader, device):
    model.eval()
    correct_1, correct_3, correct_5, correct_10, correct_50, num = 0, 0, 0, 0, 0, 0
    
    results = {}
    with torch.no_grad():
        for _, (inputs, labels, track_ids) in enumerate(tqdm.tqdm(test_loader)):
            input_batch = (inputs.to(device), labels.to(device))
            similarity = model(input_batch)
            
            # _, pred_classes_50 = similarity.topk(50, dim=1)
            # _, pred_classes_10 = similarity.topk(10, dim=1)
            # _, pred_classes_5 = similarity.topk(5, dim=1)
            # _, pred_classes_3 = similarity.topk(3, dim=1)
            pred_score_1, pred_classes_1 = similarity.topk(1, dim=1)
            
            labels = labels.squeeze(1).to(device)  # Assuming labels is of shape (n, 1), make it (n,)

            # For each sample in the batch, check if any of the predicted classes match the true labels
            # correct_50 += (pred_classes_50 == labels.unsqueeze(1)).any(dim=1).sum().item()
            # correct_10 += (pred_classes_10 == labels.unsqueeze(1)).any(dim=1).sum().item()
            # correct_5 += (pred_classes_5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            # correct_3 += (pred_classes_3 == labels.unsqueeze(1)).any(dim=1).sum().item()
            correct_1 += (pred_classes_1 == labels.unsqueeze(1)).any(dim=1).sum().item()
            num += labels.size(0)

            results.update({track_ids[j].item(): {'class': pred_classes_1[j].cpu().item(), \
                                                  'score': pred_score_1[j].cpu().item()} 
                            for j in range(labels.size(0))})

    # print(f'Accuracy_50: {correct_50 / num}')
    # print(f'Accuracy_10: {correct_10 / num}')
    # print(f'Accuracy_5: {correct_5 / num}')
    # print(f'Accuracy_3: {correct_3 / num}')
    print(f'Accuracy_1: {correct_1 / num}')
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Model Training')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 10)')
    parser.add_argument('--backbone_lr', type=float, default=0.0001, help='backbone learning rate (default: 0.0001)')
    parser.add_argument('--fusion_lr', type=float, default=0.0001, help='fusion module learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--out_path', type=str, default='outputs', help='path to save the results')
    parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training (default: cuda)')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--model_path', type=str, help='path to model weight')
    parser.add_argument('--save_epoch', type=int, default=2, help='save model every n epochs')
    parser.add_argument('--out_name', type=str, default='results.json', help='output file name')

    # model args
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cosine', 'cross_entropy', 'contrastive'], help='loss function name')
    parser.add_argument('--extraction_name', type=str, default='clip', choices=['clip', 'CLIP', 'R50', 'R101'])
    parser.add_argument('--fusion_name', type=str, default='ResidualAttentionBlock', \
                        choices=['AvgPool', 'Voting', 'CrossSimilarity', 'AttentionFusion', \
                                 'TransformerCLIP', 'AttentionFusion_v2'])
    parser.add_argument('--train_class_names', type=str, default='datasets/class_names/lvis_base_class_names.txt')
    parser.add_argument('--test_class_names', type=str, default='datasets/class_names/lvis_class_names.txt', \
                        help='class names for test set, can be the path a file or class names separated by comma')
    parser.add_argument('--clip_len', type=int, default=10, help='tracklet clip length')
    parser.add_argument('--template', type=str, default='{}')

    # dataset
    parser.add_argument('--train_path', type=str, default='datasets/tao_tracklets/train.json', help='dataset train split path')
    parser.add_argument('--test_path', type=str, default='datasets/tao_tracklets/validation_ovtrack.json', help='dataset test split path')
    parser.add_argument('--no_augment', action='store_true', help='data augmentation')

    args = parser.parse_args()
    # args check
    os.makedirs(args.out_path, exist_ok=True)
    if args.fusion_name in ['AvgPool', 'Voting']:
        assert args.eval_only == True, 'AvgPool fusion works in evaluation mode'
    
    if args.fusion_name == 'CrossSimilarity':
        assert args.loss == 'cross_entropy', 'CrossSimilarity fusion works with cross entropy loss'
    
    return args


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    clip_model, preprocess = clip.load("ViT-L/14")
    if not args.no_augment:
        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # 随机裁剪并调整大小
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，概率为0.5
            transforms.RandomRotation(degrees=15),  # 随机旋转，角度范围为 -15 到 15 度
            *preprocess.transforms,  # 结合CLIP原始预处理操作
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # 随机擦除，概率为0.5
        ])
    else:
        train_preprocess = preprocess
    print('CLIP model loaded')

    if args.extraction_name == 'R50':
        backbone = models.resnet50(pretrained=True)
        backbone = nn.Sequential(
                    *list(backbone.children())[:-1],
                    nn.Flatten(),
                    nn.Linear(2048, 768))
    elif args.extraction_name in ['clip', 'CLIP']:
        backbone = None
    elif args.extraction_name == 'SwinB':
        backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    else:
        raise ValueError('Unknown extraction model name')

    for param in clip_model.parameters():
        param.requires_grad = False

    # 加载数据集
    train_dataset = TrackletDataset(path=args.train_path, transform=train_preprocess, \
                                    clip_len=args.clip_len, split='train')
    test_dataset = TrackletDataset(path=args.test_path, transform=preprocess, \
                                   clip_len=args.clip_len, split='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, \
                              num_workers=args.num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, \
                             num_workers=args.num_workers, pin_memory=False)
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    if os.path.isfile(args.train_class_names):
        train_class_names = np.loadtxt(args.train_class_names, dtype=str, delimiter='?')
    else:
        train_class_names = args.train_class_names.split(',')
    if os.path.isfile(args.test_class_names):
        test_class_names = np.loadtxt(args.test_class_names, dtype=str, delimiter='?')
    else:
        test_class_names = args.test_class_names.split(',')
    train_class_names = [str(args.template).format(train_class_name) for train_class_name in train_class_names]
    test_class_names = [str(args.template).format(test_class_name) for test_class_name in test_class_names]
    print(f'{len(train_class_names)} train class names processed')
    print(f'{len(test_class_names)} test class names processed')

    # 模型定义
    model = TraCLIP(extraction_name=args.extraction_name,
                    extraction_model=clip_model,
                    image_backbone=backbone,
                    fusion_name=args.fusion_name,
                    loss=args.loss,
                    train_class_names=train_class_names,
                    test_class_names=test_class_names,
                    kwargs=args)

    model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.fusion_lr)

    # 加载检查点
    start_epoch = 0
    if args.resume and os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    if args.eval_only:
        if args.model_path is not None and os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            results = test(model, test_loader, device)
        else:
            if args.fusion_name in ['AvgPool', 'Voting']:
                results = test(model, test_loader, device)
            else:
                raise ValueError('Model path not found.')
        with open(os.path.join(args.out_path, args.out_name), 'w') as f:
            json.dump(results, f)
        return
    
    # 训练和测试循环
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train(model, train_loader, optimizer, device)
        if epoch % args.save_epoch == 0 or epoch == args.epochs - 1:
            # 保存模型检查点
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.out_path, 'epoch_' + str(epoch+1).zfill(3) + '.pth'))
    test(model, test_loader, device)

if __name__ == '__main__':
    main()