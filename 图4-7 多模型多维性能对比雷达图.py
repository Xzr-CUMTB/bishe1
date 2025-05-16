# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'SimSun',  # 使用系统自带宋体
    'axes.unicode_minus': False
})


# ================= 主雷达图 =================
def radar_chart(ax, metrics, data, title):
    """ 绘制多边形雷达图 """
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(20)

    # 绘制框架
    ax.set_ylim(0, 100)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # 填充各模型数据
    colors = ['#2E86C1', '#7D3C98', '#27AE60']
    labels = ['本研究', 'YOLOv5s', 'EfficientDet']
    for idx, (model, values) in enumerate(data.items()):
        values += values[:1]
        ax.plot(angles, values, color=colors[idx], linewidth=2, label=labels[idx])
        ax.fill(angles, values, color=colors[idx], alpha=0.25)

    ax.set_title(title, y=1.08, fontsize=12)


# 数据准备
metrics = ['mAP@0.5', 'AP_small', 'TPR@模糊', 'FPS', '能效比']
model_data = {
    '本研究': [94.7, 63.8, 82.3, 42.7, 42],  # 比例转换：4.2/10 * 100=42
    'YOLOv5s': [89.2, 48.9, 62.5, 38.4, 25],
    'EfficientDet': [91.5, 54.1, 72.7, 27.9, 31]
}

# 创建画布
fig = plt.figure(figsize=(14, 6))

# 主雷达图
ax_radar = fig.add_subplot(121, polar=True)
radar_chart(ax_radar, metrics, model_data, "")
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# ================= 子图：复杂度分析 =================
ax_sub = fig.add_subplot(122)
ax_sub.grid(True, alpha=0.3, linestyle='--')

# 散点绘制
scatter_params = {
    '基线': (7.8, 28.4),
    '改进方案': (8.9, 20.1)
}
for label, (x, y) in scatter_params.items():
    ax_sub.scatter(x, y, s=200 if label == '基线' else 250,
                   c='#3498DB' if label == '基线' else '#E74C3C',
                   edgecolor='k', label=label)

# 连线标注
ax_sub.plot([7.8, 8.9], [28.4, 20.1], '--', color='#2ECC71', lw=2)
ax_sub.annotate('效率提升29.3%', xy=(8.35, 24), ha='center',
                bbox=dict(boxstyle="round", fc="white", ec="gray", pad=0.3))
ax_sub.text(7.6, 28, r'$\frac{\Delta FLOPs}{\Delta Params} = -3.12$',
            fontsize=12, color='#7D3C98')

# 坐标设置
ax_sub.set_xlabel("参数量 (M)", fontsize=10)
ax_sub.set_ylabel("计算量 (GFLOPs)", fontsize=10)
ax_sub.legend()

plt.tight_layout()
plt.savefig('图4-7_多模型对比分析.png', dpi=300, bbox_inches='tight')
plt.close()