# -*- coding: utf-8 -*-
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from matplotlib.gridspec import GridSpec

# ========== 中文显示设置 ==========
plt.rcParams['font.family'] = 'sans-serif'
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
except:
    plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 生成标准示例图像 ==========
def create_sample_image():
    """创建符合图3-1标准的示例图像"""
    # 黑色背景 (0,0,0)
    img = Image.new('RGB', (256, 256), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 橙色方块 (R=255, G=80, B=0)
    square_size = 100
    x_center, y_center = 128, 128
    square_coords = [
        (x_center - square_size // 2, y_center - square_size // 2),
        (x_center + square_size // 2, y_center + square_size // 2)
    ]
    draw.rectangle(square_coords, fill=(255, 80, 0))

    # 添加文字标注
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), "Fire_001.jpg", font=font, fill=(255, 255, 255))

    return img


# ========== 数据增强流程 ==========
def get_transforms_visual():
    return {
        'original': transforms.Compose([transforms.ToTensor()]),
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


# ========== 主绘图函数 ==========
def plot_augmentation_pipeline():
    trans = get_transforms_visual()
    orig_img = create_sample_image()

    # 固定随机种子保证可复现性
    torch.manual_seed(2024)
    train_img = trans['train'](orig_img)
    val_img = trans['val'](orig_img)

    # 创建白色背景画布
    fig = plt.figure(figsize=(16, 12), dpi=120, facecolor='white')
    gs = GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.3)

    # 原始输入区域
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(orig_img)
    ax0.set_title("原始输入\n(256×256)", fontsize=10, pad=8, color='black')
    ax0.axis('off')

    # 几何变换流程
    geo_steps = [
        ("Resize(256)", transforms.Resize(256)),
        ("RandomCrop(224)", transforms.RandomCrop(224)),
        ("RandomFlip", transforms.RandomHorizontalFlip(p=1))
    ]

    img_geo = orig_img
    for i, (title, t) in enumerate(geo_steps):
        ax = fig.add_subplot(gs[0, i + 1])
        img_geo = t(img_geo)
        ax.imshow(img_geo)
        ax.set_title(f"几何变换\n{title}", fontsize=10, pad=6, color='black')
        ax.axis('off')

    # 颜色扰动示例
    color_cases = [
        ("亮度增强", {'brightness': 0.5}),
        ("对比度增强", {'contrast': 0.5}),
        ("饱和度增强", {'saturation': 0.5})
    ]

    for i, (title, params) in enumerate(color_cases):
        ax = fig.add_subplot(gs[1, i])
        t = transforms.ColorJitter(**params)
        img_color = t(orig_img)
        ax.imshow(img_color)
        ax.set_title(f"颜色扰动: {title}", fontsize=10, pad=6, color='black')
        ax.axis('off')

    # 归一化直方图
    ax_hist1 = fig.add_subplot(gs[2, :2])
    ax_hist2 = fig.add_subplot(gs[3, :2])

    orig_tensor = trans['original'](orig_img).numpy().flatten()
    ax_hist1.hist(orig_tensor, bins=50, color='blue', alpha=0.7)
    ax_hist1.set_title("归一化前像素分布", fontsize=10, pad=8, color='black')
    ax_hist1.tick_params(colors='black')
    ax_hist1.set_xlim(-0.5, 1.5)

    norm_img = trans['train'](orig_img).numpy().flatten()
    ax_hist2.hist(norm_img, bins=50, color='red', alpha=0.7)
    ax_hist2.set_title("归一化后像素分布", fontsize=10, pad=8, color='black')
    ax_hist2.tick_params(colors='black')
    ax_hist2.set_xlim(-2.5, 2.5)

    # 训练/验证对比
    ax_train = fig.add_subplot(gs[2:, 2:4])
    ax_train.imshow(train_img.permute(1, 2, 0).numpy().clip(0, 1))
    ax_train.set_title("训练集增强结果\n(随机裁剪+翻转+颜色抖动)", fontsize=11, pad=10, color='black')
    ax_train.axis('off')

    ax_val = fig.add_subplot(gs[2:, 4])
    ax_val.imshow(val_img.permute(1, 2, 0).numpy().clip(0, 1))
    ax_val.set_title("验证集预处理\n(中心裁剪+归一化)", fontsize=11, pad=10, color='black')
    ax_val.axis('off')

    # 全局标题
    plt.suptitle("图3-1 数据增强策略可视化流程", y=0.92,
                 fontsize=14, weight='bold', color='black')

    # 保存图像
    save_path = r"D:\学习\毕设\训练结果\data_aug_visualization.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"图表已保存至：{save_path}")


if __name__ == '__main__':
    plot_augmentation_pipeline()