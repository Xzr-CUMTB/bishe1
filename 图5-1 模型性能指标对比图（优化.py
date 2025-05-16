import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ========== 学术图表样式设置 ==========
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.5
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# ========== 模拟数据 ==========
models = ['Proposed\nModel', 'YOLOv5', 'VGG16', 'ResNet34']
precision = [93.5, 86.2, 78.9, 82.4]
recall = [96.2, 89.5, 81.3, 84.7]
f1_scores = [95.3, 87.8, 79.9, 83.5]

# 误差范围 (±1σ)
precision_err = [1.2, 2.3, 3.1, 2.8]
recall_err = [0.9, 1.8, 2.7, 2.1]
f1_err = [0.8, 1.5, 2.3, 1.9]

# ========== 创建画布 ==========
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=120)
ax2 = ax1.twinx()

# ========== 柱状图参数 ==========
bar_width = 0.3
x = np.arange(len(models))
offset = bar_width/2

# ========== 绘制精确率柱状图 ==========
rects1 = ax1.bar(x - offset, precision, bar_width,
                label='Precision', color='#2a4d69',
                yerr=precision_err, capsize=6,
                error_kw={'elinewidth':1.5, 'capthick':1.5})

# ========== 绘制召回率柱状图 ==========
rects2 = ax1.bar(x + offset, recall, bar_width,
                label='Recall', color='#4b86b4',
                yerr=recall_err, capsize=6,
                error_kw={'elinewidth':1.5, 'capthick':1.5})

# ========== 绘制F1分数误差线 ==========
(line, caps, _) = ax2.errorbar(x, f1_scores, yerr=f1_err,
                              fmt='s-', color='#e63946', markersize=10,
                              markeredgewidth=1.5, linewidth=2.5,
                              ecolor='#8a2be2', elinewidth=2, capsize=8,
                              label='F1 Score')

# ========== 显著性标记系统 ==========
sig_data = [
    (0, 98, '**', 14),  # Proposed模型
    (1, 90, '*', 12),   # YOLOv5
    (2, 83, '*', 12),   # VGG16
    (3, 86, '*', 12)    # ResNet34
]

for xi, y, mark, fsize in sig_data:
    ax1.text(x[xi], y, mark, fontsize=fsize, color='#d90429',
            ha='center', va='bottom', weight='bold')

# ========== 坐标轴优化 ==========
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)
ax1.set_ylabel('Precision/Recall (%)', fontsize=13, labelpad=12)
ax1.set_ylim(70, 102)
ax1.grid(True, linestyle=':', color='gray', alpha=0.6)

ax2.spines['right'].set_color('#e63946')
ax2.set_ylabel('F1 Score (%)', fontsize=13, color='#e63946', labelpad=12)
ax2.tick_params(axis='y', colors='#e63946', width=1.5)
ax2.set_ylim(70, 102)
ax2.set_yticks(np.arange(70, 101, 5))

# ========== 图例系统重构 ==========
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#2a4d69', lw=4, label='Precision'),
    Line2D([0], [0], color='#4b86b4', lw=4, label='Recall'),
    Line2D([0], [0], marker='s', color='#e63946', lw=2,
           markersize=8, label='F1 Score')
]

ax1.legend(handles=legend_elements, loc='upper right',
          frameon=True, fancybox=False,
          edgecolor='black', bbox_to_anchor=(0.82, 1))

# ========== 统计注释优化 ==========
ax1.text(0.03, 0.92, 'Error bars: ±1σ\n* p<0.05, ​** p<0.01',
        transform=ax1.transAxes, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='white',
                edgecolor='#2a4d69', alpha=0.8),
        fontsize=10)

# ========== 输出设置 ==========
plt.tight_layout()
plt.subplots_adjust(right=0.87)  # 为右侧坐标轴留出空间
plt.savefig('improved_figure5-1.png', dpi=300,
           bbox_inches='tight', pad_inches=0.2)
plt.savefig('improved_figure5-1.pdf', format='pdf',
           bbox_inches='tight', pad_inches=0.1)
plt.show()