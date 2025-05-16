# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ==================== 数据模拟 ====================
epochs = np.arange(1, 31)
train_loss = np.exp(-0.15 * epochs) + 0.02 * np.random.randn(30)
val_acc = 0.85 - 0.7 * np.exp(-0.25 * epochs) + 0.015 * np.random.randn(30)

# ==================== 字体配置 ====================
plt.rcParams.update({
    "font.family": "SimSun",  # 使用系统宋体代替Times New Roman
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "mathtext.fontset": "cm",
})

# ==================== 画布初始化 ====================
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
ax2 = ax1.twinx()

# ==================== 训练损失曲线 ====================
loss_line, = ax1.plot(epochs, train_loss,
                     color='#1f77b4',
                     linewidth=1.8,
                     marker='o',
                     markersize=5,
                     markeredgecolor='black',
                     markeredgewidth=0.5,
                     label='训练损失')

# ==================== 验证准确率曲线 ====================
acc_line, = ax2.plot(epochs, val_acc*100,
                    color='#d62728',
                    linewidth=1.8,
                    linestyle='--',
                    marker='^',
                    markersize=6,
                    markeredgecolor='black',
                    markeredgewidth=0.5,
                    label='验证准确率')

# ==================== 关键点标注 ====================
peak_epoch = 18
ax2.annotate('94.74% @ epoch 18',
            xy=(peak_epoch, val_acc[peak_epoch-1]*100),
            xytext=(22, 88),
            arrowprops=dict(arrowstyle="->", color='black', lw=0.8),
            bbox=dict(boxstyle="round", fc="w", alpha=0.8))

# ==================== 坐标轴设置 ====================
ax1.set_xlabel('训练轮次', labelpad=10)
ax1.set_ylabel('损失值', color='#1f77b4', labelpad=12)
ax2.set_ylabel('准确率 (%)', color='#d62728', labelpad=12)

ax1.xaxis.set_major_locator(MultipleLocator(5))
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.yaxis.set_major_locator(MultipleLocator(0.1))
ax1.set_ylim(0, 1.0)
ax2.set_ylim(75, 96)

# ==================== 网格与图例 ====================
ax1.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.6)
ax1.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.4)

lines = [loss_line, acc_line]
ax1.legend(lines, [line.get_label() for line in lines],
         loc='upper center',
         bbox_to_anchor=(0.5, 1.15),
         ncol=2,
         frameon=True,
         shadow=True)

# ==================== 关键区域说明 ====================
ax1.text(15, 0.25,
        '稳定下降阶段 →',
        rotation=-5,
        fontsize=10,
        color='#2ca02c',
        bbox=dict(facecolor='white', edgecolor='#2ca02c', boxstyle='round'))

# ==================== 保存配置 ====================
save_dir = os.path.dirname(__file__)  # 获取当前脚本所在目录
output_dir = os.path.join(save_dir, "output_figures")
os.makedirs(output_dir, exist_ok=True)  # 自动创建输出目录

save_path = os.path.join(output_dir, "figure4-1_training_curve.png")

# ==================== 保存图像 ====================
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight')
print(f"图像已保存至：{os.path.abspath(save_path)}")