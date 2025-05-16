import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# ========== 参数关系建模 ==========
def compute_learning_rate(B, V):
    """ 根据实际曲面形态构建的数学模型 """
    base_lr = 3.0e-4  # 对应z轴3.0×1e-4标注
    batch_factor = (B/80)**0.35  # 匹配80样本处的拐点
    vram_factor = 0.85 + 0.15 * np.tanh(8*(0.7 - V))  # 修正陡降区域
    return base_lr * batch_factor * vram_factor

# ========== 数据生成 ==========
batch = np.linspace(20, 120, 100)
vram = np.linspace(0.3, 1.0, 100)
B, V = np.meshgrid(batch, vram)
LR_scale = compute_learning_rate(B, V) * 1e4  # 转换为×1e-4单位

# ========== 学术规范可视化 ==========
plt.rcParams.update({
    'font.family': 'SimSun',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.labelweight': 'bold'
})

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 曲面绘制（匹配颜色渐变）
surf = ax.plot_surface(B, V, LR_scale, cmap=cm.coolwarm,
                       linewidth=0.5, edgecolor='k',
                       rstride=2, cstride=2,
                       vmin=0, vmax=3.5)

# 关键标注点（对应论文参数）
ax.scatter(80, 0.7, compute_learning_rate(80,0.7)*1e4,
           c='gold', s=150, marker='*', edgecolor='black',
           label='基准配置 (batch=80, VRAM=0.7)')

# 坐标轴优化
ax.set_xlabel('Batch Size\n(样本数)', labelpad=12)
ax.set_ylabel('VRAM Usage\n(显存利用率)', labelpad=12)
ax.set_zlabel('Learning Rate (×1e-4)', labelpad=10)
ax.set_zlim(0, 3.5)

# 颜色映射条
cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label('学习率尺度系数', rotation=270, labelpad=20)

# 刻度校准
ax.set_xticks(np.arange(20, 121, 20))
ax.set_yticks(np.arange(0.3, 1.1, 0.1))
ax.set_zticks(np.arange(0, 3.6, 0.5))

# 视角优化
ax.view_init(elev=32, azim=-135)

plt.legend(loc='upper left', bbox_to_anchor=(0.12, 0.92))
plt.tight_layout()
plt.savefig('图4-4_动态学习率参数响应曲面.png', dpi=300, bbox_inches='tight')
plt.close()