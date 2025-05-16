import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

# ========== 参数配置 ==========
plt.rcParams.update({
    'font.family': 'SimSun',    # 使用宋体
    'font.size': 12,
    'axes.unicode_minus': False,
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300
})

# ========== 数学建模 ==========
def dynamic_learning_rate(batch_size, vram_usage):
    """
    动态学习率与批量尺寸的非线性关系模型
    参数说明：
    batch_size - 批量尺寸 (16-128)
    vram_usage - GPU显存利用率 (0.3-1.0)
    """
    base_lr = 1e-4
    scaled_lr = base_lr * (batch_size/32)**0.8  # 非线性缩放因子
    memory_factor = 1.2 / (1 + np.exp(-10*(vram_usage-0.7)))  # 显存压力调节项
    return scaled_lr * memory_factor

# ========== 数据生成 ==========
batch = np.linspace(16, 128, 50)  # 批量尺寸范围
vram = np.linspace(0.3, 1.0, 50)  # 显存利用率范围
B, V = np.meshgrid(batch, vram)
LR = dynamic_learning_rate(B, V)

# ========== 三维可视化 ==========
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 曲面绘制
surf = ax.plot_surface(B, V, LR*1e4, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True,
                       rstride=1, cstride=1, alpha=0.8)

# 等高线投影
cset = ax.contourf(B, V, LR*1e4, zdir='z', offset=0, cmap=cm.coolwarm)

# 视角设置
ax.view_init(elev=28, azim=135)  # 最佳观察角度

# 坐标轴设置
ax.set_xlabel('Batch Size\n(样本数)', labelpad=12)
ax.set_ylabel('VRAM Usage\n(显存利用率)', labelpad=12)
ax.set_zlabel('Learning Rate (×1e-4)', labelpad=12)
ax.zaxis.set_rotate_label(False)  # 防止z轴标签自动旋转

# 颜色映射条
cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label('Learning Rate Scale', rotation=270, labelpad=20)

# 网格优化
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle":":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle":":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle":":"})

# 标注关键特征点
ax.scatter(32, 0.75, dynamic_learning_rate(32,0.75)*1e4,
           c='red', s=80, marker='*', edgecolor='black',
           label='基准配置点')
ax.text(35, 0.75, 0.85, 'Optimal\nRegion', color='blue',
        fontsize=10, ha='left', va='center')

# 图例与标题
ax.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

plt.tight_layout()
plt.savefig('3d_learning_rate_surface.png', bbox_inches='tight', dpi=300)
plt.close()