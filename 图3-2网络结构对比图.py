# -*- coding: utf-8 -*-
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

# ========== 中文显示设置 ==========
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_resnet_comparison():
    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12), dpi=300)
    plt.subplots_adjust(wspace=0.3)

    # ================= 原始ResNet-18结构 =================
    ax1.set_title("原始ResNet-18结构", fontsize=14, pad=20)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 12)
    ax1.axis('off')

    # 结构组件参数
    layers_orig = [
        {"name": "输入\n(224×224×3)", "pos": (0.5, 10), "color": "#4BACC6"},
        {"name": "Conv1\n7×7, 64", "pos": (0.5, 9), "color": "#1F497D"},
        {"name": "MaxPool\n3×3", "pos": (0.5, 8), "color": "#1F497D"},
        {"name": "Layer1\n2×残差块", "pos": (0.5, 7), "color": "#8064A2"},
        {"name": "Layer2\n2×残差块", "pos": (0.5, 6), "color": "#8064A2"},
        {"name": "Layer3\n2×残差块", "pos": (0.5, 5), "color": "#8064A2"},
        {"name": "Layer4\n2×残差块", "pos": (0.5, 4), "color": "#8064A2"},
        {"name": "全局池化", "pos": (0.5, 3), "color": "#C0504D"},
        {"name": "全连接层\n1000维", "pos": (0.5, 2), "color": "#C0504D"},
        {"name": "输出\n(ImageNet)", "pos": (0.5, 1), "color": "#4BACC6"}
    ]

    # 绘制原始结构
    for layer in layers_orig:
        ax1.add_patch(Rectangle((layer["pos"][0] - 0.4, layer["pos"][1] - 0.4), 0.8, 0.8,
                                facecolor=layer["color"], edgecolor='black', lw=1))
        ax1.text(layer["pos"][0], layer["pos"][1] - 0.45, layer["name"],
                 ha='center', va='top', fontsize=10, color='white')

    # 绘制连接箭头
    for i in range(len(layers_orig) - 1):
        ax1.add_patch(FancyArrow(layers_orig[i]["pos"][0], layers_orig[i]["pos"][1] - 0.4,
                                 0, -0.8, width=0.05, color='#7F7F7F'))

    # ================= 改进后结构 =================
    ax2.set_title("改进火灾检测网络结构", fontsize=14, pad=20)
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 12)
    ax2.axis('off')

    layers_improved = [
        {"name": "输入\n(224×224×3)", "pos": (0.5, 10), "color": "#4BACC6"},
        {"name": "Conv1\n（冻结）", "pos": (0.5, 9), "color": "#A5A5A5"},
        {"name": "MaxPool\n（冻结）", "pos": (0.5, 8), "color": "#A5A5A5"},
        {"name": "Layer1\n（冻结）", "pos": (0.5, 7), "color": "#A5A5A5"},
        {"name": "Layer2\n（冻结）", "pos": (0.5, 6), "color": "#A5A5A5"},
        {"name": "Layer3\n（冻结）", "pos": (0.5, 5), "color": "#A5A5A5"},
        {"name": "Layer4\n（可训练）", "pos": (0.5, 4), "color": "#1F497D"},
        {"name": "全局池化", "pos": (0.5, 3), "color": "#C0504D"},
        {"name": "Dropout(0.5)", "pos": (0.5, 2.5), "color": "#9BBB59"},
        {"name": "FC→512+ReLU", "pos": (0.5, 2), "color": "#C0504D"},
        {"name": "Sigmoid输出", "pos": (0.5, 1), "color": "#4BACC6"}
    ]

    # 绘制改进结构
    for layer in layers_improved:
        ax2.add_patch(Rectangle((layer["pos"][0] - 0.4, layer["pos"][1] - 0.4), 0.8, 0.8,
                                facecolor=layer["color"], edgecolor='black', lw=1))
        ax2.text(layer["pos"][0], layer["pos"][1] - 0.45, layer["name"],
                 ha='center', va='top', fontsize=10, color='white')

    # 绘制连接箭头（含Dropout特殊标记）
    for i in range(len(layers_improved) - 1):
        start_y = layers_improved[i]["pos"][1] - 0.4
        end_y = layers_improved[i + 1]["pos"][1] - 0.4 if i != 7 else 2.1
        ax2.add_patch(FancyArrow(layers_improved[i]["pos"][0], start_y,
                                 0, end_y - start_y - 0.8, width=0.05, color='#7F7F7F'))

    # 添加图例
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#A5A5A5', edgecolor='black', label='冻结层'),
        Rectangle((0, 0), 1, 1, facecolor='#1F497D', edgecolor='black', label='可训练卷积层'),
        Rectangle((0, 0), 1, 1, facecolor='#C0504D', edgecolor='black', label='分类模块'),
        Rectangle((0, 0), 1, 1, facecolor='#9BBB59', edgecolor='black', label='正则化模块')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1),
               fontsize=10, title='图例说明', title_fontsize=12)

    # 保存图像
    save_path = r"D:\学习\毕设\训练结果\network_comparison.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"网络结构对比图已保存至：{save_path}")


if __name__ == '__main__':
    draw_resnet_comparison()