# -*- coding: utf-8 -*-
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 必须在所有库导入之前设置
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
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 优先使用系统字体
except:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 后备字体
plt.rcParams['axes.unicode_minus'] = False


# ========== 生成模拟数据 ==========
def create_sample_image():
    """创建带火焰标注的示例图像"""
    img = Image.new('RGB', (256, 256), color=(40, 40, 40))  # 深灰色背景
    draw = ImageDraw.Draw(img)

    # 绘制火焰区域
    fire_coords = [(180, 200), (210, 180), (240, 220), (200, 240)]
    draw.polygon(fire_coords, fill=(255, 80, 0))  # 橙色火焰

    # 添加文字标注
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), "Fire_001.jpg", font=font, fill=(255, 255, 255))

    return img


# ========== 数据增强流程 ==========
def get_transforms_visual():
    """获取可视化专用变换"""
    return {
        'original': transforms.Compose([transforms.ToTensor()]),
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }


# ========== 主绘图函数 ==========
def plot_augmentation_pipeline():
    # 初始化变换
    trans = get_transforms_visual()
    orig_img = create_sample_image()

    # 生成增强结果
    torch.manual_seed(2024)  # 使用torch的随机种子
    train_img = trans['train'](orig_img)
    val_img = trans['val'](orig_img)

    # 创建画布
    fig = plt.figure(figsize=(16, 12), dpi=120)
    gs = GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.3)

    # 原始图像
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(orig_img)
    ax0.set_title("原始输入\n(256×256)", fontsize=10, pad=8)
    ax0.axis('off')

    # 几何变换流程
    geo_steps = [
        ("Resize(256)", transforms.Resize(256)),
        ("RandomCrop(224)", transforms.RandomCrop(224)),
        ("RandomFlip", transforms.RandomHorizontalFlip(p=1))  # 强制显示翻转
    ]

    img_geo = orig_img
    for i, (title, t) in enumerate(geo_steps):
        ax = fig.add_subplot(gs[0, i + 1])
        img_geo = t(img_geo)
        ax.imshow(img_geo)
        ax.set_title(f"几何变换\n{title}", fontsize=10, pad=6)
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
        ax.set_title(f"颜色扰动: {title}", fontsize=10, pad=6)
        ax.axis('off')

    # 归一化直方图
    ax_hist1 = fig.add_subplot(gs[2, :2])
    ax_hist2 = fig.add_subplot(gs[3, :2])

    # 原始直方图
    orig_tensor = trans['original'](orig_img).numpy().flatten()
    ax_hist1.hist(orig_tensor, bins=50, color='blue', alpha=0.7)
    ax_hist1.set_title("归一化前像素分布", fontsize=10, pad=8)
    ax_hist1.set_xlim(-0.5, 1.5)

    # 归一化后直方图
    norm_img = trans['train'](orig_img).numpy().flatten()
    ax_hist2.hist(norm_img, bins=50, color='red', alpha=0.7)
    ax_hist2.set_title("归一化后像素分布", fontsize=10, pad=8)
    ax_hist2.set_xlim(-2.5, 2.5)

    # 训练/验证对比
    ax_train = fig.add_subplot(gs[2:, 2:4])
    ax_train.imshow(train_img.permute(1, 2, 0).numpy().clip(0, 1))
    ax_train.set_title("训练集增强结果\n(随机裁剪+翻转+颜色抖动)", fontsize=11, pad=10)
    ax_train.axis('off')

    ax_val = fig.add_subplot(gs[2:, 4])
    ax_val.imshow(val_img.permute(1, 2, 0).numpy().clip(0, 1))
    ax_val.set_title("验证集预处理\n(中心裁剪+归一化)", fontsize=11, pad=10)
    ax_val.axis('off')

    # 添加全局标题
    plt.suptitle("", y=0.92, fontsize=14, weight='bold')

    # 保存图像
    save_path = r"D:\学习\毕设\训练结果\data_aug_visualization.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"图表已保存至：{save_path}")


if __name__ == '__main__':
    plot_augmentation_pipeline()