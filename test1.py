import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


# 配置参数（修正变量定义）
class Config:
    data_root = r'D:\学习\毕设\参考\火灾训练数据集'  # 原始路径适配Windows系统
    train_dir = os.path.join(data_root, 'train')  # 修正变量名缺失问题
    val_dir = os.path.join(data_root, 'val')  # 验证集路径
    batch_size = 32
    num_epochs = 30
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = 'best_fire_model.pth'


# 数据增强配置
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动增强
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 数据集加载
train_dataset = datasets.ImageFolder(
    Config.train_dir,
    transform=train_transform
)
val_dataset = datasets.ImageFolder(
    Config.val_dir,
    transform=val_transform
)

# 数据加载器（优化num_workers）
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # 加速数据传输
)

val_loader = DataLoader(
    val_dataset,
    batch_size=Config.batch_size,
    shuffle=False,
    num_workers=4
)


# 改进的模型定义
class FireDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用预训练的ResNet18
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 冻结卷积层（前5层）
        for param in list(base_model.parameters())[:-5]:
            param.requires_grad = False

        # 修改全连接层
        num_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        self.model = base_model

    def forward(self, x):
        return torch.sigmoid(self.model(x))


model = FireDetector().to(Config.device)

# 训练配置
criterion = nn.BCELoss()
optimizer = optim.Adam([
    {'params': model.model.fc.parameters(), 'lr': Config.lr},
    {'params': model.model.layer4.parameters(), 'lr': Config.lr * 0.1}
], lr=Config.lr * 0.01)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# 训练循环（增加混合精度训练）
best_val_acc = 0.0
train_loss_history = []
val_loss_history = []
val_acc_history = []

for epoch in range(Config.num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{Config.num_epochs}') as pbar:
        for images, labels in pbar:
            images = images.to(Config.device)
            labels = labels.float().unsqueeze(1).to(Config.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # 混合精度训练
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(train_loader.dataset)
    train_loss_history.append(epoch_loss)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad(), tqdm(val_loader, desc='Validating') as pbar:
        for images, labels in pbar:
            images = images.to(Config.device)
            labels = labels.float().unsqueeze(1).to(Config.device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'acc': correct / total})

    val_epoch_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total

    val_loss_history.append(val_epoch_loss)
    val_acc_history.append(val_acc)

    # 学习率调整
    scheduler.step(val_epoch_loss)

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_epoch_loss,
        }, Config.save_path)

    print(f'Epoch {epoch + 1} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.2%}')

# 可视化训练过程
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.title('Training Metrics')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_acc_history, label='Val Acc')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()


# 预测函数（参考网页1、网页5）
def predict_image(image_path):
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    img_tensor = val_transform(img).unsqueeze(0).to(Config.device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = output.item()

    plt.imshow(img)
    plt.title(f'Fire Probability: {prob:.2%}')
    plt.axis('off')
    plt.show()
    return 'Fire' if prob > 0.5 else 'No Fire'


# 示例测试
test_image = os.path.join(Config.val_dir, 'fires', 'fire_001.jpg')
print(f'Prediction: {predict_image(test_image)}')