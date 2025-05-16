import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from matplotlib import rcParams


# ========== 与训练代码完全一致的模型定义 ==========
class FireResNet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # 冻结前3个stage的参数
        for name, param in self.base.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False

        # 修改全连接层（与训练代码一致）
        in_features = self.base.fc.in_features
        self.base.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.base(x))


# ========== Grad-CAM 实现 ==========
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.gradients = []
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward(torch.ones_like(output))

        gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        activations = self.activations * gradients
        heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()

        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap


# ========== 预处理管道 ==========
def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(Image.open(image_path).convert('RGB')).unsqueeze(0)


# ========== 主执行流程 ==========
def visualize_heatmap(model_path, img_path, save_path):
    try:
        # 验证文件路径
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")

        # 设备配置
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载训练好的模型
        model = FireResNet(pretrained=False).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 获取目标层（与训练模型结构完全一致）
        target_layer = model.base.layer4[1].conv2

        # 生成热力图
        input_tensor = preprocess(img_path).to(device)
        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(input_tensor)

        # 可视化
        fig, ax = plt.subplots(1, 3, figsize=(20, 6))

        orig_img = Image.open(img_path).convert('RGB')
        ax[0].imshow(orig_img)
        ax[0].set_title('输入图像', fontsize=12)
        ax[0].axis('off')

        ax[1].imshow(heatmap, cmap='jet')
        ax[1].set_title('Grad-CAM热力图', fontsize=12)
        ax[1].axis('off')

        superimposed = Image.fromarray(
            (np.array(orig_img.resize((224, 224))) * 0.4 +
             plt.cm.jet(heatmap)[..., :3] * 255 * 0.6).astype(np.uint8)
        )
        ax[2].imshow(superimposed)
        ax[2].set_title('热力叠加效果', fontsize=12)
        ax[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"可视化结果已保存至: {os.path.abspath(save_path)}")

    except Exception as e:
        print(f"错误发生: {str(e)}")
        print("故障排除建议:")
        print("1. 检查模型文件路径是否正确")
        print("2. 确认测试图像文件存在")
        print("3. 验证模型结构与训练代码一致")
        print("4. 确保PyTorch版本>=1.12.0")


# ========== 执行示例 ==========
if __name__ == "__main__":
    # 使用实际存在的路径替换以下示例路径
    model_path = r'D:\学习\毕设\训练结果\fire_detection_best.pth'
    img_path = r'D:\学习\毕设\参考\火灾训练数据集\val\fire_86.jpg'
    save_path = 'figure5-2_corrected.png'

    visualize_heatmap(model_path, img_path, save_path)