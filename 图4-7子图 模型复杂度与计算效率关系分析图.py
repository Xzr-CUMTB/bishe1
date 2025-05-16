# ========== 独立子图代码 ==========
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'SimSun',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3
})

fig, ax = plt.subplots(figsize=(6, 4))

# 散点与连线
ax.scatter([7.8, 8.9], [28.4, 20.1], 
           s=[200, 250], c=['#3498DB', '#E74C3C'],
           edgecolors='k', zorder=5)
ax.plot([7.8, 8.9], [28.4, 20.1],
        '--', color='#2ECC71', lw=2)

# 标注与公式
ax.annotate('效率提升29.3%', xy=(8.4, 24),
            ha='center', fontsize=10,
            bbox=dict(boxstyle="round", fc="w"))
ax.text(7.6, 28.8, r'$\frac{\Delta FLOPs}{\Delta Params} = -3.12$',
        fontsize=12, color='#7D3C98')

# 坐标设置
ax.set_xlabel("参数量 (M)", fontsize=12)
ax.set_ylabel("计算量 (GFLOPs)", fontsize=12)
ax.set_xticks([7.5,8.0,8.5,9.0])
ax.set_xlim(7.5,9.0)

plt.savefig("图4-7子图_复杂度分析.png", dpi=300, bbox_inches='tight')