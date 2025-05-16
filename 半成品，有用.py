import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('TkAgg')  # 强制指定图形后端
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== 环境版本检查 ==========
print("=" * 50)
print(f"Python 版本: {sys.version.split()[0]}")
print(f"NumPy 版本: {np.__version__}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"Torchvision 版本: {models.__version__ if hasattr(models, '__version__') else '未知'}")  # 兼容处理
print(f"Matplotlib 版本: {matplotlib.__version__}")
print("=" * 50)


# ========== 系统环境检查 ==========
def check_environment():
    # 检查MKL支持
    mkl_available = np.__config__.get_info('mkl_info')
    print(f"MKL 加速可用: {'是' if mkl_available else '否'}")

    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {'是' if cuda_available else '否'}")
    if cuda_available:
        print(f"当前 CUDA 设备: {torch.cuda.get_device_name(0)}")

    # 检查Tkinter支持
    try:
        import tkinter
        print("Tkinter 支持: 正常")
    except ImportError:
        print("Tkinter 支持: 缺失")


check_environment()


# ========== 配置参数 ==========
class Config:
    data_root = r'D:\学习\毕设\参考\火灾训练数据集'
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    # 训练参数
    batch_size = 32
    num_epochs = 30
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = 'fire_detection_best.pth'


# ========== 数据预处理 ==========
def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


# ========== 模型架构 ==========
class FireResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # 冻结前4个stage的参数
        for param in list(self.base.parameters())[:-5]:
            param.requires_grad = False

        # 修改全连接层
        in_features = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.base(x))


# ========== 训练流程 ==========
def train_model(config):
    # 数据加载
    transforms_dict = get_transforms()
    train_dataset = datasets.ImageFolder(config.train_dir, transform=transforms_dict['train'])
    val_dataset = datasets.ImageFolder(config.val_dir, transform=transforms_dict['val'])

    # Windows下建议设置num_workers=0，Linux可设为4
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # 模型初始化
    model = FireResNet().to(config.device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    # 训练循环
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0

        # 训练阶段
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}', unit='batch') as pbar:
            for images, labels in pbar:
                images = images.to(config.device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(config.device)

                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 反向传播
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 验证阶段
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config.device)
                labels = labels.float().unsqueeze(1).to(config.device)

                outputs = model(images)
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()

        # 记录指标
        epoch_loss = running_loss / len(train_dataset)
        val_acc = correct / len(val_dataset)
        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)

        # 学习率调整
        scheduler.step(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'acc': val_acc
            }, config.save_path)

        print(f"Epoch {epoch + 1:02d} | 训练损失: {epoch_loss:.4f} | 验证准确率: {val_acc:.2%}")

    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.title('训练损失曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('验证准确率')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()  # 关闭图形防止内存泄漏


# ========== 主程序入口 ==========
if __name__ == '__main__':
    # Windows下必须设置多进程启动方式
    mp.set_start_method('spawn')

    # 配置检查
    assert os.path.exists(Config.train_dir), f"训练目录不存在: {Config.train_dir}"
    assert os.path.exists(Config.val_dir), f"验证目录不存在: {Config.val_dir}"

    # 启动训练
    train_model(Config())