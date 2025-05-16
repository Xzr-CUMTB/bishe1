# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ==================== 中文配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False    # 显示负号

# ==================== 数据生成 ====================
np.random.seed(2023)

# 生成梯度分布数据（精确匹配图示形态）
def generate_gradients(mean, scale, size):
    base = np.random.gamma(shape=0.5, scale=scale, size=size)  # Gamma分布模拟尖峰
    return np.clip(base + mean, 0, 3)

grad_early = generate_gradients(0.2, 0.3, 5000)  # 早期阶段（0-3梯度值）
grad_mid = generate_gradients(1.0, 0.5, 5000)    # 中期阶段
grad_late = generate_gradients(2.0, 0.6, 5000)    # 后期阶段

# KL散度数据（完全匹配图示点位）
epochs = [2, 3, 4, 5, 6, 7, 8, 9]
kl_values = [0.35, 0.28, 0.22, 0.25, 0.18, 0.12, 0.09, 0.10]

# ==================== 可视化引擎 ====================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                              gridspec_kw={'height_ratios': [2, 1]})

# ===== 梯度分布曲线 =====
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
labels = ['早期阶段 (epoch1-10)', '中期阶段 (epoch11-20)', '后期阶段 (epoch21-30)']

for data, color, label in zip([grad_early, grad_mid, grad_late], colors, labels):
    kde = gaussian_kde(data, bw_method=0.15)
    x = np.linspace(0, 3, 300)
    ax1.plot(x, kde(x)*16,  # 精确匹配0-16密度范围
            color=color,
            lw=2.5,
            label=label)

ax1.set(xlim=(0, 3), ylim=(0, 16),
       xlabel='梯度值', ylabel='概率密度',
       xticks=np.arange(0, 3.1, 0.5))
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right', frameon=False)

# ===== KL散度曲线 =====
ax2.plot(epochs, kl_values,
        color='black',
        linestyle='--',
        marker='o',
        markersize=8,
        markerfacecolor='red',
        markeredgecolor='black')

# 动态调整阶段标注
ax2.fill_betweenx(y=[0, 0.35], x1=5, x2=8,
                 color='#ffc0cb', alpha=0.3,
                 label='动态调整阶段')

# 数值标注（精确到小数点后两位）
for epoch, value in zip(epochs, kl_values):
    ax2.text(epoch, value+0.01, f'{value:.2f}',
            ha='center',
            color='red',
            fontsize=10)

ax2.set(xlim=(2, 9), ylim=(0, 0.4),
       xlabel='训练轮次', ylabel='KL散度',
       xticks=np.arange(2, 10))
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

# ==================== 保存输出 ====================
plt.tight_layout()
plt.savefig('gradient_analysis_final.png',
           bbox_inches='tight',
           dpi=300,
           facecolor='white')
print("图表已保存：gradient_analysis_final.png")