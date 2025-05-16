# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import matplotlib.font_manager as fm

# ==================== 跨平台字体配置 ====================
# 自动选择可用中文字体
try:
    # Windows优先使用微软雅黑
    font_path = fm.findfont(fm.FontProperties(family='Microsoft YaHei'))
except:
    try:
        # macOS使用苹方字体
        font_path = fm.findfont(fm.FontProperties(family='PingFang HK'))
    except:
        # Linux使用文泉驿字体
        font_path = fm.findfont(fm.FontProperties(family='WenQuanYi Zen Hei'))

# 配置全局字体参数
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],  # 英文字体备用
    "mathtext.fontset": "cm",  # 数学符号字体
    "axes.unicode_minus": False  # 解决负号显示问题
})

# 创建中文字体对象
cn_font = fm.FontProperties(fname=font_path, size=12)

# ==================== 模拟梯度数据 ====================
np.random.seed(2024)
epochs = np.arange(1, 31)

# 生成梯度数据（改用对数正态分布避免负值）
early_grad = np.random.lognormal(mean=-1.0, sigma=0.5, size=(10, 1000))
mid_grad = np.random.lognormal(mean=-1.5, sigma=0.3, size=(10, 1000))
late_grad = np.random.lognormal(mean=-2.0, sigma=0.2, size=(10, 1000))
grad_data = np.vstack([early_grad, mid_grad, late_grad])

# ==================== 计算KL散度 ====================
base_dist = np.histogram(grad_data[4:9], bins=50, density=True)[0]
kl_divergence = [entropy(np.histogram(grad_data[i], bins=50, density=True)[0],
                         qk=base_dist) for i in range(len(grad_data))]

# ==================== 创建画布 ====================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300,
                               gridspec_kw={'height_ratios': [2, 1]})

# ==================== 梯度直方图 ====================
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
labels = ['早期阶段 (epoch1-10)', '中期阶段 (epoch11-20)', '后期阶段 (epoch21-30)']
for idx, grad in enumerate([grad_data[:10], grad_data[10:20], grad_data[20:30]]):
    ax1.hist(grad.flatten(), bins=50, density=True,
             alpha=0.6, color=colors[idx], label=labels[idx])

ax1.set_title('梯度分布直方图', fontproperties=cn_font)
ax1.set_ylabel('概率密度', fontproperties=cn_font)
ax1.legend(prop=cn_font)
ax1.grid(True, linestyle='--', alpha=0.6)

# ==================== KL散度曲线 ====================
ax2.plot(epochs, kl_divergence,
         color='#d62728',
         linewidth=2,
         marker='o',
         markersize=6,
         markeredgecolor='black')

ax2.fill_between(epochs, kl_divergence,
                 alpha=0.2,
                 color='#d62728',
                 where=np.array(kl_divergence) < 0.15)

ax2.set_xlabel('训练轮次', fontproperties=cn_font)
ax2.set_ylabel('KL散度', fontproperties=cn_font)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.set_ylim(0, 0.35)

# ==================== 保存配置 ====================
save_dir = os.path.join(os.path.dirname(__file__), "output_figures")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "figure4-2_gradient_analysis.png")

# ==================== 保存图像 ====================
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight')
print(f"梯度分析图已保存至：{os.path.abspath(save_path)}")