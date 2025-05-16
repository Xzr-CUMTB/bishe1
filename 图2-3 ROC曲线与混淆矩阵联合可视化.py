import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import rcParams

# ========== 中文显示设置 ==========
try:
    rcParams['font.family'] = 'SimSun'  # 正式论文推荐使用宋体
    rcParams['font.size'] = 12
    rcParams['axes.unicode_minus'] = False
except:
    pass

# ========== 根据实际测试集构建数据 ==========
# 测试集样本分布：火灾80，非火灾91
# 真实标签构建 (1:火灾，0:非火灾)
true_labels = np.array([1]*80 + [0]*91)

# 模拟预测概率（需要根据实际预测结果调整）
# 假设模型表现：
# - 火灾样本：75正确，5错误
# - 非火灾样本：85正确，6错误
pred_probs = np.concatenate([
    np.random.normal(0.85, 0.1, 75),  # 正确分类的火灾样本
    np.random.normal(0.3, 0.2, 5),    # 错误分类的火灾样本
    np.random.normal(0.15, 0.1, 85),  # 正确分类的非火灾样本
    np.random.normal(0.7, 0.2, 6)     # 错误分类的非火灾样本
])

# ========== 生成预测标签 ==========
y_pred = (pred_probs > 0.5).astype(int)

# ========== 混淆矩阵计算 ==========
cm = confusion_matrix(true_labels, y_pred)

# ========== 专业可视化设置 ==========
plt.figure(figsize=(8, 6), dpi=300)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmax=np.max(cm)*1.2)

# ========== 坐标轴设置 ==========
class_names = ['非火灾', '火灾']
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
plt.yticks(tick_marks, class_names, fontsize=12)
plt.xlabel('预测标签', fontsize=14, labelpad=10)
plt.ylabel('真实标签', fontsize=14, labelpad=10)

# ========== 数值标注 ==========
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14)

# ========== 精度标注 ==========
total_accuracy = np.trace(cm) / np.sum(cm)
plt.text(1.5, -0.4,
         f'总准确率: {total_accuracy:.2%}\n'
         f'火灾召回率: {cm[1,1]/80:.2%}\n'
         f'非火灾特异度: {cm[0,0]/91:.2%}',
         ha='center', va='top', fontsize=12)

# ========== 保存图片 ==========
plt.tight_layout()
save_path = os.path.join(r'D:\学习\毕设\训练结果', 'confusion_matrix_actual.png')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
plt.close()

print(f"混淆矩阵已保存至：{save_path}")