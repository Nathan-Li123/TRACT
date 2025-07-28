import os
import torch
import clip
import time
from PIL import Image


def calculate_speed_01():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    img_root = '/data2/resources/TAO/val/LaSOT/dog-11/'
    imgs = os.listdir(img_root)
    img_num = len(imgs)
    print('total image num:', img_num)

    time_begin = time.time()
    for img in imgs:
        img_path = os.path.join(img_root, img)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        text = clip.tokenize(["a diagram", "a dog", "a cat", "a mouse", "a bear", "a car", "a person"]).to(device)
        with torch.no_grad():
            # image_features = model.encode_image(image)
            # text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    time_end = time.time()
    total_time = time_end - time_begin
    print('total time:', total_time)
    print('average time:', total_time / img_num)


def calculate_speed_02():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    img_root = '/data2/resources/TAO/val/LaSOT/dog-11/'
    imgs = os.listdir(img_root)
    img_num = len(imgs)
    print('total image num:', img_num)

    text = clip.tokenize(["a diagram", "a dog", "a cat", "a mouse", "a bear", "a car", "a person"]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    # import pdb; pdb.set_trace()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    time_begin = time.time()
    for img in imgs:
        img_path = os.path.join(img_root, img)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        # import pdb; pdb.set_trace()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # values, indices = similarity[0].topk(3)
    time_end = time.time()
    total_time = time_end - time_begin
    print('total time:', total_time)
    print('average time:', total_time / img_num)
        

if __name__ == '__main__':
    calculate_speed_02()