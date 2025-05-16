import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ========== 学术图表样式设置 ==========
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.2
plt.rcParams['pdf.fonttype'] = 42

# ========== 模拟数据 ==========
models = ['Proposed\nModel', 'YOLOv5', 'VGG16', 'ResNet34']
precision = [93.5, 86.2, 78.9, 82.4]
recall = [96.2, 89.5, 81.3, 84.7]
f1_scores = [95.3, 87.8, 79.9, 83.5]

# 误差范围 (假设数据)
precision_err = [1.2, 2.3, 3.1, 2.8]
recall_err = [0.9, 1.8, 2.7, 2.1]
f1_err = [0.8, 1.5, 2.3, 1.9]

# ========== 创建画布 ==========
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# ========== 柱状图参数 ==========
bar_width = 0.35
x = np.arange(len(models))
offset = 0.2  # 误差棒偏移量

# ========== 绘制精确率柱状图 ==========
rects1 = ax1.bar(x - bar_width/2, precision, bar_width,
                label='Precision', color='#1f77b4',
                yerr=precision_err, capsize=4, error_kw={'elinewidth':1})

# ========== 绘制召回率柱状图 ==========
rects2 = ax1.bar(x + bar_width/2, recall, bar_width,
                label='Recall', color='#ff7f0e',
                yerr=recall_err, capsize=4, error_kw={'elinewidth':1})

# ========== 绘制F1分数误差线 ==========
(line, caps, _) = ax2.errorbar(x, f1_scores, yerr=f1_err,
                              fmt='r-D', markersize=8, markeredgewidth=1,
                              linewidth=2, label='F1 Score',
                              ecolor='#d62728', elinewidth=1.5, capsize=5)

# 设置误差条端盖样式
for cap in caps:
    cap.set_markeredgewidth(1.5)

# ========== 添加显著性标记 ==========
sig_positions = [
    (0, 97, '**'),  # Proposed模型位置
    (1, 90, '*'),   # YOLOv5对比
    (2, 85, '*'),   # VGG16对比
    (3, 87, '*')    # ResNet34对比
]

for xi, y, mark in sig_positions:
    ax1.text(x[xi]-0.15, y, mark, fontsize=14, color='red', ha='center')

# ========== 坐标轴设置 ==========
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylabel('Metric Value (%)', fontsize=12)
ax1.set_ylim(70, 100)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.set_ylabel('F1 Score (%)', fontsize=12)
ax2.set_ylim(70, 100)
ax2.spines['right'].set_color('red')
ax2.yaxis.label.set_color('red')
ax2.tick_params(axis='y', colors='red')

# ========== 组合图例 ==========
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + [line] + handles2,
          labels1 + ['F1 Score'] + labels2,
          loc='upper left', frameon=False)

# ========== 添加统计注释 ==========
ax1.text(0.05, 0.95, 'Error bars: ±1σ\n**p<0.01, *p<0.05',
        transform=ax1.transAxes, ha='left', va='top',
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

# ========== 优化布局并保存 ==========
plt.tight_layout()
plt.savefig('figure5-1.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5-1.pdf', format='pdf', bbox_inches='tight')
plt.show()