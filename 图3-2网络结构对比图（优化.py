# -*- coding: utf-8 -*-
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec  # 添加GridSpec导入
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.font_manager import FontProperties

# ========== 中文显示设置 ==========
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_enhanced_comparison():
    # 创建画布
    fig = plt.figure(figsize=(24, 14), dpi=300)
    gs = GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[4, 1], hspace=0.15)  # 正确使用GridSpec

    # ================= 结构对比图 =================
    ax1 = fig.add_subplot(gs[0, 0])  # 原始结构
    ax2 = fig.add_subplot(gs[0, 1])  # 改进结构
    ax_table = fig.add_subplot(gs[1, :])  # 参数表格

    # ================= 原始ResNet-18结构 =================
    def draw_original_structure(ax):
        ax.set_title("原始ResNet-18结构", fontsize=16, pad=20, fontweight='bold')
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 12)
        ax.axis('off')

        layers = [
            {"name": "输入\n(224×224×3)", "pos": (0.5, 10), "color": "#4BACC6", "edge": "#2F5597"},
            {"name": "Conv1\n7×7,64", "pos": (0.5, 9), "color": "#2F5597", "edge": "#1F497D"},
            {"name": "MaxPool\n3×3", "pos": (0.5, 8), "color": "#2F5597", "edge": "#1F497D"},
            {"name": "Layer1\n2×残差块", "pos": (0.5, 7), "color": "#8064A2", "edge": "#4F2D7F"},
            {"name": "Layer2\n2×残差块", "pos": (0.5, 6), "color": "#8064A2", "edge": "#4F2D7F"},
            {"name": "Layer3\n2×残差块", "pos": (0.5, 5), "color": "#8064A2", "edge": "#4F2D7F"},
            {"name": "Layer4\n2×残差块", "pos": (0.5, 4), "color": "#8064A2", "edge": "#4F2D7F"},
            {"name": "全局池化", "pos": (0.5, 3), "color": "#C0504D", "edge": "#9B2335"},
            {"name": "FC-1000", "pos": (0.5, 2), "color": "#C0504D", "edge": "#9B2335"},
            {"name": "输出", "pos": (0.5, 1), "color": "#4BACC6", "edge": "#2F5597"}
        ]

        for layer in layers:
            rect = Rectangle((layer["pos"][0] - 0.45, layer["pos"][1] - 0.45), 0.9, 0.9,
                             facecolor=layer["color"], edgecolor=layer["edge"], lw=2)
            ax.add_patch(rect)
            ax.text(layer["pos"][0], layer["pos"][1] - 0.5, layer["name"],
                    ha='center', va='top', fontsize=12, color='white',
                    fontproperties=FontProperties(weight='bold'))

        for i in range(len(layers) - 1):
            ax.arrow(layers[i]["pos"][0], layers[i]["pos"][1] - 0.9,
                     0, -0.8, head_width=0.15, head_length=0.2,
                     fc='#7F7F7F', ec='#404040', lw=1.5)

    # ================= 改进结构 =================
    def draw_improved_structure(ax):
        ax.set_title("改进火灾检测网络结构", fontsize=16, pad=20, fontweight='bold')
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 12)
        ax.axis('off')

        layers = [
            {"name": "输入\n(224×224×3)", "pos": (0.5, 10), "color": "#4BACC6", "edge": "#2F5597"},
            {"name": "Conv1\n(冻结)", "pos": (0.5, 9), "color": "#B0C4DE", "edge": "#7F9DB9"},
            {"name": "MaxPool\n(冻结)", "pos": (0.5, 8), "color": "#B0C4DE", "edge": "#7F9DB9"},
            {"name": "Layer1-3\n(冻结)", "pos": (0.5, 7), "color": "#B0C4DE", "edge": "#7F9DB9"},
            {"name": "Layer4\n(可训练)", "pos": (0.5, 4), "color": "#FF6F61", "edge": "#E6584D"},
            {"name": "全局池化", "pos": (0.5, 3), "color": "#88B04B", "edge": "#5A7247"},
            {"name": "Dropout\n(p=0.5)", "pos": (0.5, 2.5), "color": "#EFC050", "edge": "#D4A017"},
            {"name": "FC-512", "pos": (0.5, 2), "color": "#88B04B", "edge": "#5A7247"},
            {"name": "输出", "pos": (0.5, 1), "color": "#4BACC6", "edge": "#2F5597"}
        ]

        for layer in layers:
            rect = Rectangle((layer["pos"][0] - 0.45, layer["pos"][1] - 0.45), 0.9, 0.9,
                             facecolor=layer["color"], edgecolor=layer["edge"], lw=2)
            ax.add_patch(rect)
            ax.text(layer["pos"][0], layer["pos"][1] - 0.5, layer["name"],
                    ha='center', va='top', fontsize=12, color='white',
                    fontproperties=FontProperties(weight='bold'))

        ax.text(0.5, 5.5, "动态解冻区域", ha='center', va='center',
                fontsize=14, color='#E6584D', weight='bold',
                bbox=dict(facecolor='white', edgecolor='#E6584D', boxstyle='round'))

        ax.arrow(0.5, 4 - 0.9, 0, -2.8, head_width=0.15, head_length=0.2,
                 fc='#E6584D', ec='#8B0000', lw=2, linestyle='--')

    # ================= 参数表格 =================
    def create_parameter_table(ax):
        ax.axis('off')
        columns = ("网络组件", "原始参数量", "改进参数量")
        cell_text = [
            ("卷积层", "11.2M", "0.9M (冻结)"),
            ("全连接层", "0.5M", "0.3M"),
            ("正则化模块", "-", "0.1M"),
            ("总计", "11.7M", "1.3M")
        ]

        table = ax.table(cellText=cell_text,
                         colLabels=columns,
                         loc='center',
                         cellLoc='center',
                         colColours=['#F2F2F2'] * 3,
                         colWidths=[0.25, 0.25, 0.25])

        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2)

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#2F5597')
                cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor('#F2F2F2')
            cell.set_edgecolor('#D3D3D3')

    # ================= 执行绘图 =================
    draw_original_structure(ax1)
    draw_improved_structure(ax2)
    create_parameter_table(ax_table)

    # 保存图像
    save_path = r"D:\学习\毕设\训练结果\enhanced_network_comparison.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"优化后的网络对比图已保存至：{save_path}")


if __name__ == '__main__':
    draw_enhanced_comparison()