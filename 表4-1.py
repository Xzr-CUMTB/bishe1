import matplotlib.pyplot as plt

# ========== 表格数据定义 ==========
table_data = [
    ["Model",    "Accuracy (%)", "FP Rate", "FN Rate", "F1-Score", "Speed (ms)"],
    ["本研究",    94.7,          0.067,     0.052,     0.941,      18.7],
    ["YOLOv5",   89.2,          0.121,     0.083,     0.893,      23.4],
    ["EfficientDet", 91.5,      0.098,     0.071,     0.913,      27.9],
    ["ResNet-50",88.6,          0.153,     0.094,     0.882,      35.2]
]

# ========== 学术规范可视化 ==========
plt.rcParams.update({
    'font.family': 'SimSun',  # 确保中文字体显示
    'font.size': 10,
    'axes.unicode_minus': False
})

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111, frame_on=False)
ax.axis('off')

# 创建表格并设置样式
table = ax.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc='center',
    loc='center',
    colColours=['#B0C4DE']*6,  # 淡蓝色列标题背景
    cellColours=[
        ['#F5F5F5' if i%2==0 else 'white' for _ in row]  # 斑马条纹
        for i, row in enumerate(table_data[1:])
    ]
)

# 高亮最优值
best_cells = {
    (0,1): '#98FB98',  # Accuracy
    (0,2): '#FFB6C1',  # FP Rate
    (0,3): '#FFB6C1',  # FN Rate
    (0,4): '#98FB98',   # F1-Score
    (0,5): '#98FB98'    # Speed
}
for (row,col), color in best_cells.items():
    table[row, col].set_facecolor(color)

# 设置字体和尺寸
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)  # 调整表格宽高

# 添加注释
plt.text(0.5, 0.92, '表4-2 多模型性能对比',
         ha='center', va='center', fontsize=14,
         transform=fig.transFigure)

plt.savefig('table4-2.png', dpi=300, bbox_inches='tight')
plt.close()