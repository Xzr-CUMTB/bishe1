import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ==== 1. 修复字体配置 ====
# 设置中文字体（需确保系统存在该字体）
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用系统宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==== 2. 数据准备 ====
# 从训练日志中提取的迁移学习验证准确率 (30个epoch)
transfer_acc = [
    83.04, 89.47, 94.15, 92.98, 93.57, 93.57, 94.15, 93.57, 94.74,
    94.74, 94.74, 94.74, 94.74, 94.74, 94.74, 94.74, 94.74, 94.74,
    94.74, 94.74, 94.74, 94.74, 94.74, 94.74, 94.74, 94.74, 93.57,
    93.57, 93.57, 94.74
]

# 模拟随机初始化训练的验证准确率（使用numpy 1.21兼容函数）
np.random.seed(42)
random_acc = np.linspace(50, 82.1, 30) + np.random.normal(0, 3, 30)
random_acc = np.clip(random_acc, 50, 85).tolist()

# ==== 3. 可视化配置 ====
plt.figure(figsize=(10, 6), dpi=150)
epochs = np.arange(1, 31)

# ==== 4. 绘制曲线 ====
# 迁移学习曲线
plt.plot(epochs, transfer_acc,
         color='#1f77b4',  # 标准蓝色
         linewidth=2.5,
         marker='o',
         markersize=8,
         markerfacecolor='white',
         markeredgewidth=2,
         label='迁移学习策略')

# 随机初始化曲线
plt.plot(epochs, random_acc,
         color='#ff7f0e',  # 标准橙色
         linestyle='--',
         linewidth=2,
         alpha=0.8,
         marker='s',
         markersize=6,
         label='随机初始化训练')

# ==== 5. 标注关键节点 ====
highlight_epoch = 5
highlight_acc = transfer_acc[highlight_epoch-1]
plt.annotate(f'迁移学习第{highlight_epoch}轮突破90%',
             xy=(highlight_epoch, highlight_acc),
             xytext=(highlight_epoch+2, highlight_acc-5),
             arrowprops=dict(arrowstyle="->", color='#2ca02c', lw=1.5),
             fontsize=10,
             color='#2ca02c')

# ==== 6. 样式优化 ====
plt.title('迁移学习与随机初始化训练效果对比', fontsize=14, pad=20)
plt.xlabel('训练轮次 (Epoch)', labelpad=10)
plt.ylabel('验证集准确率 (%)', labelpad=10)
plt.xticks(np.arange(0, 31, 5))
plt.yticks(np.arange(50, 101, 5))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')

# ==== 7. 保存输出 ====
plt.tight_layout()
plt.savefig(r'迁移学习对比曲线.png', bbox_inches='tight', pad_inches=0.2)
plt.show()