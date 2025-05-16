# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyArrow

# ========== 中文显示设置 ==========
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_single_comparison():
    # 创建画布
    fig = plt.figure(figsize=(20, 8), dpi=300)
    gs = GridSpec(1, 2, width_ratios=[1, 1.2])  # 单行两列布局

    # ================= 结构对比图 =================
    ax1 = fig.add_subplot(gs[0, 0])  # 原始结构
    ax2 = fig.add_subplot(gs[0, 1])  # 改进结构

    # ================= 原始ResNet-18结构 =================
    def draw_original_structure(ax):
        ax.set_title("原始ResNet-18结构", fontsize=14, pad=15, fontweight='bold')
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # 结构组件参数
        layers = [
            {"name": "输入\n(224×224×3)", "pos": (0.5, 9), "color": "#4BACC6", "edge": "#2F5597"},
            {"name": "Conv1\n7×7,64", "pos": (0.5, 8), "color": "#2F5597", "edge": "#1F497D"},
            {"name": "MaxPool\n3×3", "pos": (0.5, 7), "color": "#2F5597", "edge": "#1F497D"},
            {"name": "Layer1\n2×残差块", "pos": (0.5, 6), "color": "#8064A2", "edge": "#4F2D7F"},
            {"name": "Layer2\n2×残差块", "pos": (0.5, 5), "color": "#8064A2", "edge": "#4F2D7F"},
            {"name": "Layer3\n2×残差块", "pos": (0.5, 4), "color": "#8064A2", "edge": "#4F2D7F"},
            {"name": "Layer4\n2×残差块", "pos": (0.5, 3), "color": "#8064A2", "edge": "#4F2D7F"},
            {"name": "全局池化", "pos": (0.5, 2), "color": "#C0504D", "edge": "#9B2335"},
            {"name": "FC-1000", "pos": (0.5, 1), "color": "#C0504D", "edge": "#9B2335"}
        ]

        # 绘制模块
        for layer in layers:
            rect = Rectangle((layer["pos"][0]-0.4, layer["pos"][1]-0.4), 0.8, 0.8,
                           facecolor=layer["color"], edgecolor=layer["edge"], lw=1.5)
            ax.add_patch(rect)
            ax.text(layer["pos"][0], layer["pos"][1]-0.45, layer["name"],
                   ha='center', va='top', fontsize=10, color='white')

        # 绘制连接箭头
        for i in range(len(layers)-1):
            ax.add_patch(FancyArrow(layers[i]["pos"][0], layers[i]["pos"][1]-0.8,
                                  0, -0.8, width=0.05, color='#666666'))

    # ================= 改进结构 =================
    def draw_improved_structure(ax):
        ax.set_title("改进火灾检测网络结构", fontsize=14, pad=15, fontweight='bold')
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # 结构组件参数
        layers = [
            {"name": "输入\n(224×224×3)", "pos": (0.5, 9), "color": "#4BACC6", "edge": "#2F5597"},
            {"name": "Conv1\n(冻结)", "pos": (0.5, 8), "color": "#B0C4DE", "edge": "#7F9DB9"},
            {"name": "MaxPool\n(冻结)", "pos": (0.5, 7), "color": "#B0C4DE", "edge": "#7F9DB9"},
            {"name": "Layer1-3\n(冻结)", "pos": (0.5, 6), "color": "#B0C4DE", "edge": "#7F9DB9"},
            {"name": "Layer4\n(可训练)", "pos": (0.5, 5), "color": "#2F5597", "edge": "#1F497D"},
            {"name": "全局池化", "pos": (0.5, 4), "color": "#C0504D", "edge": "#9B2335"},
            {"name": "Dropout\n(p=0.5)", "pos": (0.5, 3), "color": "#EFC050", "edge": "#D4A017"},
            {"name": "FC-512", "pos": (0.5, 2), "color": "#C0504D", "edge": "#9B2335"}
        ]

        # 绘制模块
        for layer in layers:
            rect = Rectangle((layer["pos"][0]-0.4, layer["pos"][1]-0.4), 0.8, 0.8,
                           facecolor=layer["color"], edgecolor=layer["edge"], lw=1.5)
            ax.add_patch(rect)
            ax.text(layer["pos"][0], layer["pos"][1]-0.45, layer["name"],
                   ha='center', va='top', fontsize=10, color='white')

        # 绘制连接箭头
        for i in range(len(layers)-1):
            ax.add_patch(FancyArrow(layers[i]["pos"][0], layers[i]["pos"][1]-0.8,
                                  0, -0.8, width=0.05, color='#666666'))

        # 添加图例
        legend_elements = [
            Rectangle((0,0),1,1, facecolor='#B0C4DE', edgecolor='black', label='冻结层'),
            Rectangle((0,0),1,1, facecolor='#2F5597', edgecolor='black', label='可训练卷积层'),
            Rectangle((0,0),1,1, facecolor='#C0504D', edgecolor='black', label='分类模块'),
            Rectangle((0,0),1,1, facecolor='#EFC050', edgecolor='black', label='正则化模块')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1),
                 fontsize=9, title='图例说明', title_fontsize=10)

    # ================= 执行绘图 =================
    draw_original_structure(ax1)
    draw_improved_structure(ax2)

    # 保存图像
    save_path = r"D:\学习\毕设\训练结果\network_comparison_single.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"网络结构对比图已保存至：{save_path}")

if __name__ == '__main__':
    draw_single_comparison()