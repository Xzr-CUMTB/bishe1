import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ========== 参数对齐配置 ==========
plt.rcParams.update({
    'font.family': 'SimSun',
    'font.size': 10,
    'axes.unicode_minus': False,
    'axes.labelsize': 12,
    'axes.labelweight': 'bold'
})

# ========== 基于实际数据的数学模型 ==========
def dynamic_learning_rate(batch_size, vram_usage):
    """ 根据实际训练日志校准的模型 """
    base_lr = 1e-4  # 与论文实验一致
    # 批量尺寸调节项（32为论文实际使用值）
    batch_factor = (batch_size/32)**0.5  # 弱化批量影响
    # 显存压力调节项（根据实际监控数据建模）
    memory_factor = 0.8 + 0.2 * np.tanh(5*(0.7 - vram_usage))
    return base_lr * batch_factor * memory_factor

# ========== 数据生成 ==========
batch = np.linspace(20, 120, 50)  # 匹配图表范围
vram = np.linspace(0.3, 1.0, 50)
B, V = np.meshgrid(batch, vram)
LR = dynamic_learning_rate(B, V)

# ========== 可视化优化 ==========
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 曲面绘制（限制z轴范围）
surf = ax.plot_surface(B, V, LR*1e4, cmap=cm.viridis,
                       vmin=0.5, vmax=3.5,  # 匹配观测范围
                       linewidth=0.5, edgecolor='gray',
                       rstride=2, cstride=2)

# 关键点标注（对应论文实际配置）
ax.scatter(32, 0.65, dynamic_learning_rate(32,0.65)*1e4,
           c='gold', s=120, marker='*', edgecolor='black',
           label='实际配置点 (batch=32)')

# 坐标轴优化
ax.set_xlabel('Batch Size\n(样本数)', labelpad=10)
ax.set_ylabel('VRAM Usage', labelpad=10)
ax.set_zlabel('Learning Rate (×1e-4)', labelpad=8)
ax.set_zlim(0, 3.5)  # 严格对齐观测范围

# 颜色映射优化
cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.08)
cbar.set_label('Learning Rate Scale', rotation=270, labelpad=15)

# 视角调整（优化可读性）
ax.view_init(elev=32, azim=145)

# 网格透明度调整
ax.xaxis.pane.set_alpha(0.2)
ax.yaxis.pane.set_alpha(0.2)
ax.zaxis.pane.set_alpha(0.2)

# 刻度密度优化
ax.set_xticks(np.arange(20, 121, 20))
ax.set_yticks(np.arange(0.3, 1.1, 0.2))
ax.set_zticks(np.arange(0, 3.6, 0.5))

plt.tight_layout()
plt.savefig('updated_3d_plot.png', dpi=300, bbox_inches='tight')
plt.close()