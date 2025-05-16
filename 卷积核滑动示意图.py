import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
data = np.array([
    [0, 5, 10, 15, 20],
    [5, 10, 15, 20, 15],
    [10, 15, 20, 15, 10],
    [15, 20, 15, 10, 5],
    [20, 15, 10, 5, 0]
])

# 创建画布并调整布局参数
fig, ax = plt.subplots(figsize=(8, 7))  # 增加画布高度
fig.subplots_adjust(top=0.85)  # 为标题预留空间

# 绘制热力图
im = ax.imshow(data, cmap='Blues', interpolation='nearest')

# 添加卷积核标注
kernel_params = [
    {'pos': (0.5, 1.5), 'text': (1, 2)},
    {'pos': (1.5, 2.5), 'text': (2, 3)},
    {'pos': (2.5, 3.5), 'text': (3, 4)}
]

for param in kernel_params:
    rect = plt.Rectangle(
        param['pos'], 1, 1,
        edgecolor='limegreen',
        facecolor='none',
        linewidth=2,
        linestyle='--'
    )
    ax.add_patch(rect)
    ax.text(
        param['text'][0], param['text'][1],
        f'Kernel {kernel_params.index(param)+1}',
        ha='center', va='center',
        color='white',
        fontsize=10,
        weight='bold'
    )

# 坐标轴设置
ax.set_xlabel('Width direction', fontsize=12, labelpad=10)
ax.set_ylabel('Height direction', fontsize=12, labelpad=10)
ax.set_xticks(np.arange(5))
ax.set_yticks(np.arange(5))

# 优化标题设置
title = ax.set_title(
    'Convolution Kernel Sliding Demonstration (Stride=1)',
    fontsize=14,
    pad=20,  # 增加标题与图像的间距
    y=1.05,  # 微调垂直位置
    weight='bold'
)

# 颜色条设置
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Value', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# 添加网格线辅助观察
ax.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.5)
ax.set_xticks(np.arange(-0.5, 5), minor=True)
ax.set_yticks(np.arange(-0.5, 5), minor=True)

# 自动调整布局
plt.tight_layout()

# 保存高清图像
plt.savefig('conv_demo_fixed.png', dpi=300, bbox_inches='tight')
plt.show()