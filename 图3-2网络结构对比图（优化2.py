# -*- coding: utf-8 -*-
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects  # 正确导入路径效果
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyArrow

# ========== 中文显示设置 ==========
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_improved_comparison():
    # 创建画布
    fig = plt.figure(figsize=(24, 12), dpi=300)
    gs = GridSpec(1, 2, width_ratios=[1, 1.2])

    # ===== 原始ResNet-18结构 =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("原始ResNet-18结构", fontsize=18, pad=20, color='black', weight='bold')
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 10)
    ax1.axis('off')

    resnet_layers = [
        {"name": "输入\n(224×224×3)", "pos": (0.5, 9), "color": "#E6F3FF"},
        {"name": "Conv1\n7×7,64", "pos": (0.5, 8), "color": "#B8D4FF"},
        {"name": "MaxPool\n3×3", "pos": (0.5, 7), "color": "#B8D4FF"},
        {"name": "Layer1\n2×残差块", "pos": (0.5, 6), "color": "#D5E8D4"},
        {"name": "Layer2\n2×残差块", "pos": (0.5, 5), "color": "#D5E8D4"},
        {"name": "Layer3\n2×残差块", "pos": (0.5, 4), "color": "#D5E8D4"},
        {"name": "Layer4\n2×残差块", "pos": (0.5, 3), "color": "#D5E8D4"},
        {"name": "全局池化", "pos": (0.5, 2), "color": "#FFE6CC"},
        {"name": "FC-1000", "pos": (0.5, 1), "color": "#FFE6CC"}
    ]

    # ===== 改进火灾检测结构 =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("改进火灾检测网络结构", fontsize=18, pad=20, color='black', weight='bold')
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    improved_layers = [
        {"name": "输入\n(224×224×3)", "pos": (0.5, 9), "color": "#E6F3FF"},
        {"name": "Conv1\n(冻结)", "pos": (0.5, 8), "color": "#F0F0F0"},
        {"name": "MaxPool\n(冻结)", "pos": (0.5, 7), "color": "#F0F0F0"},
        {"name": "Layer1-3\n(冻结)", "pos": (0.5, 6), "color": "#F0F0F0"},
        {"name": "Layer4\n(可训练)", "pos": (0.5, 5), "color": "#B8D4FF"},
        {"name": "全局池化", "pos": (0.5, 4), "color": "#FFE6CC"},
        {"name": "Dropout\n(p=0.5)", "pos": (0.5, 3), "color": "#FFF2CC"},
        {"name": "FC-512", "pos": (0.5, 2), "color": "#FFE6CC"}
    ]

    # ===== 通用绘图函数 =====
    def draw_structure(ax, layers):
        for layer in layers:
            rect = Rectangle((layer["pos"][0] - 0.45, layer["pos"][1] - 0.45),
                             0.9, 0.9, facecolor=layer["color"], edgecolor='gray', lw=1)
            ax.add_patch(rect)
            # 修正路径效果调用方式
            ax.text(layer["pos"][0], layer["pos"][1] - 0.5, layer["name"],
                    ha='center', va='top', fontsize=14, color='black',
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])  # 正确引用

            # 添加连接箭头
            if layer != layers[-1]:
                ax.add_patch(FancyArrow(layer["pos"][0], layer["pos"][1] - 0.9,
                                        0, -0.8, width=0.08, color='#666666'))

    # 绘制两个结构
    draw_structure(ax1, resnet_layers)
    draw_structure(ax2, improved_layers)

    # ===== 添加图例 =====
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#F0F0F0', edgecolor='gray', label='冻结层'),
        Rectangle((0, 0), 1, 1, facecolor='#B8D4FF', edgecolor='gray', label='可训练卷积层'),
        Rectangle((0, 0), 1, 1, facecolor='#FFE6CC', edgecolor='gray', label='分类模块'),
        Rectangle((0, 0), 1, 1, facecolor='#FFF2CC', edgecolor='gray', label='正则化模块')
    ]
    ax2.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(1.1, 1), fontsize=12, labelcolor='black',
               title='图例说明', title_fontsize=14)

    # 保存图像
    save_path = r"D:\学习\毕设\训练结果\final_network_comparison.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"最终网络对比图已保存至：{save_path}")


if __name__ == '__main__':
    draw_improved_comparison()