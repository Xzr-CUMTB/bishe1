# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ========== 中文显示设置 ==========
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

# 生成模拟训练数据
epochs = np.arange(1, 31)
train_loss = np.exp(-epochs / 8) + np.random.normal(0, 0.02, 30)
val_acc = 0.82 + 0.15 * (1 - np.exp(-epochs / 10)) + np.random.normal(0, 0.01, 30)
best_epoch = 18  # 最佳检查点
stop_epoch = 23  # 早停触发点

# 创建画布
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)

# ===== 训练损失曲线 =====
ax1.plot(epochs, train_loss, color='#2F5597', lw=2, label='训练损失')
ax1.set_ylabel('训练损失', fontsize=12)
ax1.set_ylim(0, 1.0)
ax1.grid(True, ls=':', alpha=0.5)

# ===== 验证准确率曲线 =====
ax2.plot(epochs, val_acc, color='#C00000', lw=2, label='验证准确率')
ax2.set_ylabel('验证准确率', fontsize=12)
ax2.set_xlabel('训练轮次 (Epoch)', fontsize=12)
ax2.set_ylim(0.8, 0.98)
ax2.grid(True, ls=':', alpha=0.5)


# ===== 标注关键事件 =====
def add_annotation(ax, y, text, color):
    ax.annotate(text,
                xy=(best_epoch, y),
                xytext=(best_epoch + 3, y - 0.05 if ax == ax1 else y - 0.03),
                arrowprops=dict(arrowstyle="->", color=color),
                fontsize=10, color=color)


# 最佳检查点
for ax, y_pos in zip([ax1, ax2], [0.25, 0.92]):
    ax.axvline(best_epoch, color='#00B050', ls='--', lw=1.5)
    add_annotation(ax, y_pos, f'最佳检查点 (epoch {best_epoch})', '#00B050')

# 早停触发点
for ax, y_pos in zip([ax1, ax2], [0.4, 0.88]):
    ax.axvline(stop_epoch, color='#FF6600', ls='--', lw=1.5)
    ax.annotate(f'早停触发 (epoch {stop_epoch})',
                xy=(stop_epoch, y_pos),
                xytext=(stop_epoch - 8, y_pos + 0.05 if ax == ax1 else y_pos + 0.02),
                arrowprops=dict(arrowstyle="->", color='#FF6600'),
                fontsize=10, color='#FF6600')

# 保存区域标注
ax2.fill_betweenx([0.85, 0.97], best_epoch, stop_epoch, color='#FFF2CC', alpha=0.3)
ax2.text((best_epoch + stop_epoch) / 2, 0.95, '检查点保存区',
         ha='center', va='center', color='#D4A017', fontsize=11)

# 添加图例
ax1.legend(loc='upper right')
ax2.legend(loc='lower right')

# 保存图像
save_path = r"D:\学习\毕设\训练结果\checkpoint_illustration.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"检查点示意图已保存至：{save_path}")