# ========== 环境设置 ==========
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import random

# ========== 配置参数 ==========
model_path = r'D:\学习\毕设\训练结果\fire_detection_best.pth'
val_root = r'D:\学习\毕设\参考\火灾训练数据集\val'


# ========== 智能图像搜索 ==========
def find_sample_image(root_dir):
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    image_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(image_exts):
                image_files.append(os.path.join(dirpath, f))

    if not image_files:
        raise FileNotFoundError(
            f"验证集目录结构验证失败：\n"
            f"实际路径：{os.path.abspath(root_dir)}\n"
            f"应有结构：\n"
            f"{root_dir}\n"
            f"├── fire\n"
            f"│   └── *.jpg\n"
            f"└── nonfire\n"
            f"    └── *.jpg"
        )

    return random.choice(image_files)


# ========== 模型定义 ==========
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = resnet18(weights=None)

        # 参数冻结
        for name, param in self.base.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False

        # 分类头结构
        in_features = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # 特征图保存
        self.feature_maps = []

        def hook_fn(module, input, output):
            self.feature_maps.append(output.detach().cpu())

        target_layers = [
            self.base.layer1[0].conv1,
            self.base.layer2[0].conv1,
            self.base.layer3[0].conv1,
            self.base.layer4[0].conv1
        ]
        for layer in target_layers:
            layer.register_forward_hook(hook_fn)

    def forward(self, x):
        return torch.sigmoid(self.base(x))


# ========== 主程序 ==========
if __name__ == "__main__":
    # 多进程设置
    mp.set_start_method('spawn', force=True)

    # 目录验证
    if not os.path.exists(val_root):
        print(f"目录验证失败：{os.path.abspath(val_root)}")
        exit()

    # 图像选择
    try:
        image_path = find_sample_image(val_root)
        print(f"样本路径：{os.path.abspath(image_path)}")
    except Exception as e:
        print(str(e))
        exit()

    # 模型加载
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FeatureExtractor().to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        # 动态键名适配
        state_dict = {
            k.replace('base.fc.1', 'base.fc.3'): v
            for k, v in checkpoint['model_state_dict'].items()
        }
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as e:
        print(f"模型加载异常：{str(e)}")
        exit()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"图像处理异常：{str(e)}")
        exit()

    # 特征提取
    with torch.no_grad():
        prediction = model(input_tensor)
        feature_maps = model.feature_maps

    # ========== 可视化布局 ==========
    plt.figure(figsize=(20, 5))
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'SimHei',
        'axes.unicode_minus': False
    })

    # (a) 输入图像
    plt.subplot(1, 5, 1)
    plt.imshow(img)
    plt.title('(a) 输入图像\n尺寸: 256×256×3')
    plt.axis('off')

    # (b) 特征图组
    layer_params = [
        ('layer1', 64, 3, 1, 56),
        ('layer2', 128, 3, 2, 28),
        ('layer3', 256, 3, 2, 14),
        ('layer4', 512, 3, 2, 7)
    ]

    for i, (fm, params) in enumerate(zip(feature_maps, layer_params)):
        plt.subplot(1, 5, i + 2)
        channel_mix = torch.mean(fm[0, :8], dim=0).numpy()
        channel_mix = (channel_mix - channel_mix.min()) / (channel_mix.max() - channel_mix.min())

        plt.imshow(channel_mix, cmap='jet')
        title = (
            f'(b) {params[0]}\n'
            f'通道: {params[1]}\n'
            f'卷积核: {params[2]}×{params[2]}\n'
            f'步长: {params[3]}\n'
            f'输出尺寸: {params[4]}×{params[4]}'
        )
        plt.title(title)
        plt.axis('off')

    # (c) 分类结果（修改点1：标题文本）
    plt.subplot(1, 5, 5)
    prob = prediction[0][0].item()

    # 修改点2：类别标签
    class_labels = ['无火焰', '存在火焰']  # 原为 ['非火灾', '火灾']
    plt.barh(class_labels, [1 - prob, prob], color=['#1f77b4', '#ff7f0e'])

    # 修改点3：概率描述
    plt.title(f'(c) 分类概率\n存在火焰概率: {prob * 100:.1f}%')  # 原为 "火灾概率"
    plt.xlim(0, 1)
    plt.grid(axis='x', alpha=0.3)

    # 保存输出
    output_path = os.path.join(os.path.dirname(model_path), 'CNN_Feature_Visualization.png')
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n可视化文件路径：{os.path.abspath(output_path)}")
    print("运行状态：成功（退出代码0）")