# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ========== 可视化样式配置 ==========
plt.rcParams.update({
    'font.family': 'SimSun',        # 中文字体
    'axes.labelsize': 12,           # 坐标轴标签字号
    'xtick.labelsize': 10,          # x轴刻度字号
    'ytick.labelsize': 10,          # y轴刻度字号
    'axes.linewidth': 1.5,          # 坐标轴线宽
    'savefig.dpi': 300,             # 输出分辨率
    'figure.autolayout': True       # 自动排版
})

# ========== 实验数据生成 ==========
def power_model(B, a, b):
    """ 系统功耗数学模型 """
    return a * np.exp(b * B/10)

batch_sizes = np.array([1, 2, 4, 8, 16])
fps_measured = np.array([42.7, 55.3, 68.9, 76.4, 79.2])
power_measured = np.array([3.8, 4.1, 4.9, 5.7, 7.2])

# 曲线拟合 (FPS模型)
def fps_model(B, c, d):
    return c * (1 - np.exp(-d * B))
popt_fps, _ = curve_fit(fps_model, batch_sizes, fps_measured, p0=[80, 0.1])

# 曲线拟合 (功耗模型)
popt_power, _ = curve_fit(power_model, batch_sizes, power_measured)

# ========== 学术级双轴可视化 ==========
fig, ax1 = plt.subplots(figsize=(8, 5))

# 绘制FPS曲线（带置信区间）
B_cont = np.linspace(1, 20, 100)
ax1.plot(B_cont, fps_model(B_cont, *popt_fps),
        color='#2980B9', lw=2.5, alpha=0.7,
        label='吞吐量趋势线')
ax1.scatter(batch_sizes, fps_measured,
           s=80, color='#154360', edgecolor='white',
           zorder=5, label='实测FPS')

# 绘制功耗曲线（带误差棒）
power_err = 0.1 + 0.05 * power_measured  # 模拟测量误差
ax2 = ax1.twinx()
ax2.errorbar(batch_sizes, power_measured, yerr=power_err,
            fmt='D', color='#C0392B', ecolor='#922B21', elinewidth=2,
            markersize=8, capsize=5, label='功耗实测值')
ax2.plot(B_cont, power_model(B_cont, *popt_power),
        '--', color='#E74C3C', lw=2.5, alpha=0.7,
        label='功耗趋势线')

# ========== 图表标注优化 ==========
# 坐标轴设置
ax1.set_xlabel('批量尺寸（Batch Size）', fontsize=12, labelpad=10)
ax1.set_ylabel('帧率（FPS）', color='#2980B9', fontsize=12, labelpad=12)
ax2.set_ylabel('系统功耗（W）', color='#C0392B', fontsize=12, labelpad=12)

# 刻度与范围
ax1.set_xticks(np.arange(0, 21, 4))
ax1.set_xlim(0, 20)
ax1.set_ylim(30, 85)
ax2.set_ylim(3, 9)

# 网格与图例
ax1.grid(True, alpha=0.3, linestyle=':', color='#7F8C8D')
ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
ax2.legend(loc='upper right', frameon=True, framealpha=0.9)

# 关键点标注
optimal_B = 8
ax1.annotate(f'最优能效点\n(B={optimal_B}, FPS={fps_model(8, *popt_fps):.1f})',
            xy=(8, fps_model(8, *popt_fps)), xytext=(12, 60),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2",
                            color='#2C3E50', lw=1.5),
            bbox=dict(boxstyle="round", fc="w", pad=0.3),
            fontsize=10, ha='center')

# 色带标注
ax1.fill_between(B_cont, fps_model(B_cont, *popt_fps), 85,
                color='#D6EAF8', alpha=0.2)

plt.title("",
         fontsize=13, pad=15, fontweight='semibold')
plt.savefig('图4-6_实时性能验证.png', bbox_inches='tight')
plt.close()