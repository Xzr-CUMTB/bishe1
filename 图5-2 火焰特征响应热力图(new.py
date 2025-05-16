# ========== 前置环境设置 ==========
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from skimage.transform import resize  # 新增尺寸调整库


# ========== 模型定义 ==========
class FireResNet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.base = models.resnet18(weights=weights)

        # 参数冻结（与训练代码一致）
        for name, param in self.base.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False

        # 分类头结构
        in_features = self.base.fc.in_features
        self.base.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.base(x))


# ========== Grad-CAM 生成器 ==========
class GradCAMGenerator:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # 注册hook
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        # 前向传播
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward(torch.ones_like(output))

        # 计算特征权重
        pooled_grads = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        weighted_activations = self.activations * pooled_grads

        # 生成热力图（原始尺寸）
        heatmap = torch.mean(weighted_activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)

        # 调整热力图尺寸到输入图像大小
        heatmap = resize(heatmap, (224, 224),
                         order=1, mode='constant',
                         preserve_range=True, anti_aliasing=False)

        # 归一化处理
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap


# ========== 预处理管道 ==========
def build_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# ========== 可视化主函数 ==========
def visualize_gradcam(model_path, img_path, output_path):
    try:
        # ===== 文件验证 =====
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"输入图像不存在: {img_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ===== 设备配置 =====
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"运算设备: {device}")

        # ===== 模型加载 =====
        model = FireResNet(pretrained=False).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model.eval()

        # ===== 数据预处理 =====
        transform = build_transform()
        input_tensor = transform(Image.open(img_path).convert('RGB'))
        input_tensor = input_tensor.unsqueeze(0).to(device)

        # ===== 生成热力图 =====
        target_layer = model.base.layer4[1].conv2  # 最后一个卷积层
        cam_generator = GradCAMGenerator(model, target_layer)
        heatmap = cam_generator.generate(input_tensor)

        # ===== 可视化渲染 =====
        plt.figure(figsize=(18, 6))
        orig_img = Image.open(img_path).convert('RGB').resize((224, 224))

        # 子图1：原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(orig_img)
        plt.title('输入图像', fontsize=12)
        plt.axis('off')

        # 子图2：热力图
        plt.subplot(1, 3, 2)
        hm = plt.imshow(heatmap, cmap='jet')
        plt.colorbar(hm, fraction=0.046, pad=0.04)
        plt.title('特征响应热力图', fontsize=12)
        plt.axis('off')

        # 子图3：叠加效果
        plt.subplot(1, 3, 3)
        superimposed = (np.array(orig_img) * 0.4 +
                        plt.cm.jet(heatmap)[..., :3] * 255 * 0.6).astype(np.uint8)
        plt.imshow(superimposed)
        plt.title('热力叠加可视化', fontsize=12)
        plt.axis('off')

        # 保存结果
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"可视化结果已保存至: {os.path.abspath(output_path)}")

    except Exception as e:
        print("\n错误分析:")
        print(f"• 错误类型: {type(e).__name__}")
        print(f"• 详细描述: {str(e)}")
        print("\n故障排除步骤:")
        print("1. 检查输入图像是否为RGB三通道格式")
        print("2. 确认模型结构是否与训练时完全一致")
        print("3. 验证skimage库是否安装：pip install scikit-image")
        print("4. 尝试将输入图像缩放到224x224像素")


# ========== 主程序入口 ==========
if __name__ == "__main__":
    config = {
        "model_path": r"D:\学习\毕设\训练结果\fire_detection_best.pth",
        "img_path": r"D:\学习\毕设\参考\火灾训练数据集\val\fire_86.jpg",
        "output_path": r"D:\可视化结果\gradcam_output.png"
    }
    visualize_gradcam(**config)