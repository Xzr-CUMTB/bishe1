import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== 中文显示配置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ========== 环境检查 ==========
print("=" * 50)
print(f"Python 版本: {sys.version.split()[0]}")
print(f"NumPy 版本: {np.__version__}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"Torchvision 版本: {models.__version__}")
print(f"Matplotlib 版本: {matplotlib.__version__}")
print("=" * 50)


# ========== 配置参数 ==========
class Config:
    data_root = r'D:\学习\毕设\参考\火灾训练数据集'
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    # 训练参数
    batch_size = 32
    num_epochs = 40  # 增加训练轮次
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = 'fire_detection_best.pth'
    early_stop_patience = 5  # 早停耐心值


# ========== 数据预处理（增强版） ==========
def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


# ========== 改进模型架构 ==========
class FireResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载预训练模型
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 冻结前3个stage的参数
        for param in list(self.base.parameters())[:-10]:
            param.requires_grad = False

        # 改进分类头
        in_features = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Dropout(0.7),  # 增强正则化
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),  # 添加批归一化
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.base(x))


# ========== 增强训练流程 ==========
def train_model(config):
    # 数据加载
    transforms_dict = get_transforms()
    train_dataset = datasets.ImageFolder(config.train_dir, transform=transforms_dict['train'])
    val_dataset = datasets.ImageFolder(config.val_dir, transform=transforms_dict['val'])

    # 处理类别不平衡（fires:300, nofires:356）
    class_counts = np.array([300, 356])
    weights = 1. / class_counts
    samples_weights = weights[train_dataset.targets]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,  # 使用加权采样
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # 增大验证批大小
        shuffle=False,
        num_workers=0
    )

    # 模型初始化
    model = FireResNet().to(config.device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)  # L2正则
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    scaler = torch.cuda.amp.GradScaler()  # 混合精度训练

    # 训练状态跟踪
    best_acc = 0.0
    no_improve_epochs = 0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0

        # 训练阶段（混合精度）
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}', unit='batch') as pbar:
            for images, labels in pbar:
                images = images.to(config.device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(config.device)

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

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

        # 早停机制
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_epochs = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'acc': val_acc
            }, config.save_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= config.early_stop_patience:
                print(f"\n早停触发：连续{config.early_stop_patience}轮未提升")
                break

        print(f"Epoch {epoch + 1:02d} | 训练损失: {epoch_loss:.4f} | 验证准确率: {val_acc:.2%}")

    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-', label='训练损失')
    plt.title('训练损失曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], 'r-', label='验证准确率')
    plt.title('验证准确率趋势')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


# ========== 主程序入口 ==========
if __name__ == '__main__':
    mp.set_start_method('spawn')

    # 环境检查
    assert os.path.exists(Config.train_dir), f"训练目录不存在: {Config.train_dir}"
    assert os.path.exists(Config.val_dir), f"验证目录不存在: {Config.val_dir}"

    # 启动训练
    train_model(Config())