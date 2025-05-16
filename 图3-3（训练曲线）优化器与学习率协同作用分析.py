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
lr_change_epoch = 15

# AdamW数据
adamw_train_loss = np.exp(-epochs/10) + np.random.normal(0, 0.02, 30)
adamw_val_acc = 0.85 + 0.1*(1 - np.exp(-epochs/8)) + np.random.normal(0, 0.01, 30)
adamw_val_acc[15:] += 0.02  # 学习率下降后提升效果

# Adam数据
adam_train_loss = np.exp(-epochs/12) + np.random.normal(0, 0.025, 30)
adam_val_acc = 0.83 + 0.09*(1 - np.exp(-epochs/10)) + np.random.normal(0, 0.015, 30)
adam_val_acc[15:] -= 0.005  # 学习率下降后性能下降

# 创建画布
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)

# ===== 训练损失子图 =====
ax1.plot(epochs, adamw_train_loss, color='#2F5597', lw=2, label='AdamW')
ax1.plot(epochs, adam_train_loss, color='#C00000', lw=2, ls='--', label='Adam')
ax1.set_ylabel('训练损失', fontsize=12)
ax1.set_ylim(0, 1.2)
ax1.grid(True, ls=':', alpha=0.5)

# ===== 验证准确率子图 =====
ax2.plot(epochs, adamw_val_acc, color='#2F5597', lw=2)
ax2.plot(epochs, adam_val_acc, color='#C00000', lw=2, ls='--')
ax2.set_ylabel('验证准确率', fontsize=12)
ax2.set_ylim(0.8, 0.98)
ax2.set_xlabel('训练轮次 (Epoch)', fontsize=12)
ax2.grid(True, ls=':', alpha=0.5)

# ===== 标注关键事件 =====
for ax in [ax1, ax2]:
    ax.axvline(lr_change_epoch, color='#00B050', ls=':', lw=2, alpha=0.8)
    ax.annotate('学习率调整点',
                xy=(lr_change_epoch, 0.05 if ax == ax1 else 0.82),
                xytext=(lr_change_epoch+2, 0.1 if ax == ax1 else 0.85),
                arrowprops=dict(arrowstyle="->", color='#00B050'),
                color='#00B050', fontsize=10)

# ===== 图例与标题 =====
ax1.legend(loc='upper right', fontsize=10)
plt.suptitle("", y=0.95, fontsize=14, weight='bold')

# 调整布局
plt.tight_layout()

# 保存图像
save_path = r"D:\学习\毕设\训练结果\optimizer_lr_synergy.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"图表已保存至：{save_path}")