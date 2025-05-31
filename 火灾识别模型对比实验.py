import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import time
import pandas as pd
import seaborn as sns

# ========== 中文显示修复 ==========
try:
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
    rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"字体设置失败: {str(e)}")

# ========== 环境版本检查 ==========
print("=" * 50)
print(f"Python 版本: {sys.version.split()[0]}")
print(f"PyTorch 版本: {torch.__version__}")
print("=" * 50)


# ========== 配置参数 ==========
class Config:
    data_root = r'D:\学习\毕设\参考\火灾训练数据集'
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    result_dir = r'D:\学习\毕设\训练结果\对比实验'

    batch_size = 32
    num_epochs = 30
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 对比模型列表
    model_names = ['resnet18', 'vgg16', 'mobilenet_v3_small']
    model_paths = {
        'resnet18': os.path.join(result_dir, 'resnet18_best.pth'),
        'vgg16': os.path.join(result_dir, 'vgg16_best.pth'),
        'mobilenet_v3_small': os.path.join(result_dir, 'mobilenet_v3_small_best.pth')
    }


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
def create_model(model_name, pretrained=True):
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # 冻结前3层
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        # 冻结特征提取层
        for param in model.features.parameters():
            param.requires_grad = False
        # 修改分类头
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1)
        )

    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        # 冻结特征提取层
        for param in model.features.parameters():
            param.requires_grad = False
        # 修改分类头
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1)
        )

    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model


# ========== 训练单个模型 ==========
def train_model(model_name, config):
    model_dir = os.path.join(config.result_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # 数据加载
    transforms_dict = get_transforms()
    train_dataset = datasets.ImageFolder(config.train_dir, transform=transforms_dict['train'])
    val_dataset = datasets.ImageFolder(config.val_dir, transform=transforms_dict['val'])

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
    model = create_model(model_name).to(config.device)
    criterion = nn.BCELoss()

    # 仅优化需要梯度的参数
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    # 训练记录
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }

    print(f"\n{'=' * 50}")
    print(f"开始训练模型: {model_name}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"总参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"设备: {config.device}")
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")
    print('=' * 50)

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # 训练阶段
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}', unit='batch') as pbar:
            for images, labels in pbar:
                images = images.to(config.device)
                labels = labels.float().unsqueeze(1).to(config.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 统计训练准确率
                preds = (outputs > 0.5).float()
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)
                running_loss += loss.item() * images.size(0)

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{correct_train / total_train:.2%}"
                })

        # 验证阶段
        model.eval()
        correct_val = 0
        total_val = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config.device)
                labels = labels.float().unsqueeze(1).to(config.device)

                outputs = model(images)
                preds = (outputs > 0.5).float()

                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())

        # 计算验证指标
        val_acc = correct_val / total_val
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        val_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        val_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (
                                                                                                val_precision + val_recall) > 0 else 0

        # 记录指标
        epoch_loss = running_loss / len(train_dataset)
        train_acc = correct_train / total_train

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

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
                'acc': val_acc,
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1
            }, config.model_paths[model_name])
            print(f"保存最佳模型在Epoch {epoch + 1}，验证准确率 {val_acc:.2%}")

        print(f"Epoch {epoch + 1:02d} | 训练损失: {epoch_loss:.4f} | 训练准确率: {train_acc:.2%} | "
              f"验证准确率: {val_acc:.2%} | 精确率: {val_precision:.2%} | 召回率: {val_recall:.2%} | F1: {val_f1:.4f}")

    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(model_dir, f'{model_name}_history.csv'), index=False)

    # 可视化训练过程
    plot_training_history(history, model_name, model_dir)

    return history


# ========== 可视化训练历史 ==========
def plot_training_history(history, model_name, save_dir):
    plt.figure(figsize=(15, 10))

    # 准确率曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title(f'{model_name} - 准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 损失曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_loss'], label='训练损失')
    plt.title(f'{model_name} - 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 精确率-召回率曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['val_recall'], history['val_precision'], 'b-')
    plt.title(f'{model_name} - 精确率-召回率曲线')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.grid(True)

    # F1分数曲线
    plt.subplot(2, 2, 4)
    plt.plot(history['val_f1'], 'g-')
    plt.title(f'{model_name} - F1分数曲线')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_metrics.png'), dpi=300)
    plt.close()


# ========== 模型评估 ==========
def evaluate_model(model, data_loader, config):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    inference_times = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(config.device)
            start_time = time.time()
            outputs = model(images)
            inference_times.append(time.time() - start_time)

            probs = outputs.cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.numpy().flatten())
            all_probs.extend(probs)
            all_preds.extend(preds)

    # 计算指标
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # ROC曲线
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # PR曲线
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)

    # 推理时间
    avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒
    fps = 1 / np.mean(inference_times)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'ap': ap,
        'inference_time_ms': avg_inference_time,
        'fps': fps,
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve
    }

    return results


# ========== 可视化评估结果 ==========
def plot_evaluation_results(results, model_name, save_dir):
    # 混淆矩阵
    plt.figure(figsize=(6, 5))
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['非火灾', '火灾'],
                yticklabels=['非火灾', '火灾'])
    plt.title(f'{model_name} - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=300)
    plt.close()

    # ROC曲线
    plt.figure()
    plt.plot(results['fpr'], results['tpr'], color='darkorange', lw=2,
             label=f'ROC曲线 (AUC = {results["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC曲线')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, f'{model_name}_roc_curve.png'), dpi=300)
    plt.close()

    # PR曲线
    plt.figure()
    plt.plot(results['recall_curve'], results['precision_curve'], color='blue', lw=2,
             label=f'PR曲线 (AP = {results["ap"]:.2f})')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f'{model_name} - 精确率-召回率曲线')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(save_dir, f'{model_name}_pr_curve.png'), dpi=300)
    plt.close()


# ========== 对比实验主函数 ==========
def run_comparative_experiment(config):
    os.makedirs(config.result_dir, exist_ok=True)

    # 加载数据集
    transforms_dict = get_transforms()
    val_dataset = datasets.ImageFolder(config.val_dir, transform=transforms_dict['val'])
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # 存储所有结果
    all_results = {}
    comparison_data = []

    # 训练和评估每个模型
    for model_name in config.model_names:
        model_dir = os.path.join(config.result_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # 训练模型（如果尚未训练）
        if not os.path.exists(config.model_paths[model_name]):
            print(f"\n{'=' * 50}")
            print(f"训练模型: {model_name}")
            print('=' * 50)
            train_model(model_name, config)

        # 加载最佳模型
        model = create_model(model_name, pretrained=False).to(config.device)
        checkpoint = torch.load(config.model_paths[model_name])
        model.load_state_dict(checkpoint['model_state_dict'])

        # 计算可训练参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        # 评估模型
        print(f"\n{'=' * 50}")
        print(f"评估模型: {model_name}")
        print('=' * 50)
        results = evaluate_model(model, val_loader, config)
        all_results[model_name] = results

        # 可视化评估结果
        plot_evaluation_results(results, model_name, model_dir)

        # 添加到对比数据
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1 Score': results['f1'],
            'AUC': results['roc_auc'],
            'AP': results['ap'],
            'Inference Time (ms)': results['inference_time_ms'],
            'FPS': results['fps'],
            'Trainable Params': trainable_params,
            'Total Params': total_params
        })

    # 保存和显示对比结果
    df = pd.DataFrame(comparison_data)
    df.to_csv(os.path.join(config.result_dir, 'model_comparison.csv'), index=False)

    # 打印对比结果
    print("\n模型性能对比:")
    print(df)

    # 可视化对比结果
    plot_comparison_results(df, config.result_dir)

    return df


# ========== 可视化对比结果 ==========
def plot_comparison_results(df, save_dir):
    # 主要指标对比
    plt.figure(figsize=(12, 8))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'AP']

    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        sns.barplot(x='Model', y=metric, data=df, palette='viridis')
        plt.title(metric)
        plt.ylim(0.7, 1.0)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300)
    plt.close()


for epoch in range(config.num_epochs):
    # 训练阶段
    with tqdm(...) as pbar:
        for images, labels in pbar:
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad(set_to_none=True)  # 高效的内存优化
            loss.backward()
            optimizer.step()

    # 验证阶段
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            # 收集预测结果
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(outputs.cpu().numpy().flatten())


def plot_metrics(history, all_labels, all_probs, config):
    # 混淆矩阵
    y_pred = (np.array(all_probs) > 0.5).astype(int)
    cm = confusion_matrix(all_labels, y_pred)

    # ROC曲线
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)


if __name__ == '__main__':
    mp.set_start_method('spawn')  # Windows系统必需的多进程设置
    config = Config()

    # 路径验证
    assert os.path.exists(config.train_dir)
    assert os.path.exists(config.val_dir)