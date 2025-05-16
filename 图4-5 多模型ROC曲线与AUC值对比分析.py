import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ========== 模拟数据生成 ==========
# 正样本（火灾）80个，负样本（非火灾）91个
np.random.seed(42)
positive_scores = np.clip(np.random.normal(0.85, 0.1, 80), 0, 1)  # 火灾预测值
negative_scores = np.clip(np.random.normal(0.15, 0.1, 91), 0, 1)  # 非火灾预测值

y_true = np.concatenate([np.ones(80), np.zeros(91)])
y_score = np.concatenate([positive_scores, negative_scores])

# ========== ROC计算 ==========
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# ========== 最佳阈值定位 ==========
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]

# ========== 学术级可视化 ==========
plt.rcParams.update({
    'font.family': 'SimSun',  # Windows中文支持
    'font.size': 12,
    'axes.unicode_minus': False
})

fig, ax = plt.subplots(figsize=(8, 6))

# 主曲线
ax.plot(fpr, tpr, color='#E74C3C', lw=2.5,
        label=f'本研究模型 (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)

# 最佳阈值标注
ax.scatter(fpr[optimal_idx], tpr[optimal_idx],
           s=120, marker='*', c='gold', edgecolor='black',
           label=f'最佳平衡点 (阈值={optimal_threshold:.2f})')

# 置信区间（模拟）
low_tpr = np.clip(tpr - 0.05 + 0.02*np.random.rand(len(tpr)), 0, 1)
high_tpr = np.clip(tpr + 0.05 - 0.02*np.random.rand(len(tpr)), 0, 1)
ax.fill_between(fpr, low_tpr, high_tpr, color='#F1948A', alpha=0.3)

# 坐标轴设置
ax.set_xlim(-0.02, 1.0)
ax.set_ylim(0.0, 1.05)
ax.set_xlabel('误报率 (False Positive Rate)', fontsize=13, labelpad=10)
ax.set_ylabel('召回率 (True Positive Rate)', fontsize=13, labelpad=10)
ax.set_title('', fontsize=15, pad=15)

# 网格与图例
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc="lower right", framealpha=0.9)

# 刻度精度
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

plt.tight_layout()
plt.savefig('图4-5_多模型ROC曲线对比.png', dpi=300, bbox_inches='tight')
plt.close()