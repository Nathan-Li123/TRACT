import os
import cv2
import matplotlib.pyplot as plt


image_paths = [
    "val/LaSOT/bear-8/00001081.jpg",
    "val/LaSOT/bear-8/00001081.jpg",
    # "val/LaSOT/tank-4/00000961.jpg",
    # "val/LaSOT/tank-4/00000961.jpg",
    "val/Charades/DOQ9Y/frame1231.jpg",
    "val/Charades/DOQ9Y/frame1231.jpg"
]
colors = [
    (0, 255, 0),
    (255, 0, 0),
    # (0, 255, 0),
    # (255, 0, 0),
    (0, 255, 0),
    (255, 0, 0)
]

total_boxes = [
    (408.0, 362.0, 290.0, 85.0, "walrus"),
    (408.0, 362.0, 290.0, 85.0, "bear"),
    # (452.0, 336.0, 125.0, 44.0, "army_tank"),
    # (452.0, 336.0, 125.0, 44.0, "vehicle"),
    (682.0, 538.0, 94.0, 157.0, "cylinder"),
    (682.0, 538.0, 94.0, 157.0, "plate")
]

gt_texts = [
    "GT: walrus",
    "GT: walrus",
    "GT: cylinder",
    "GT: cylinder"
]

out_paths = [
    "tools/images/00001081_right.pdf",
    "tools/images/00001081_wrong.pdf",
    # "tools/images/00000961_right.pdf",
    # "tools/images/00000961_wrong.pdf",
    "tools/images/frame1231_right.pdf",
    "tools/images/frame1231_wrong.pdf"
]

gt_colors = [
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0)
]

for i in range(len(image_paths)):
    # 读取图像
    image_path = image_paths[i]
    image_path = os.path.join('data/tao/frames', image_path)
    image_name = image_path.split('/')[-1]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

    # 定义框的位置和标签 (x, y, width, height, label)
    boxes = [total_boxes[i]]

    # 绘制边框和标签
    for (x, y, w, h, label) in boxes:
        x, y, w, h = int(x), int(y), int(w), int(h)
        # 绘制矩形框（颜色为红色，线宽为2）
        cv2.rectangle(image, (x, y), (x + w, y + h), color=colors[i], thickness=15)
        # 添加标签（位置为框的左上角，字体为Hershey字体）
        cv2.putText(image, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=4.0, color=colors[i], thickness=15)
        # 在图片左上角写文字
        cv2.putText(image, gt_texts[i], (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=4.0, color=gt_colors[i], thickness=15)

    # 可视化结果（可选）
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(out_paths[i], bbox_inches='tight', pad_inches=0)