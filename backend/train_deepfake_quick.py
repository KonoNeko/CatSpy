"""
Quick GPU Training Script for Deepfake Detection - 20分钟快速训练
使用合成数据进行快速训练和测试GPU性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from pathlib import Path
import time
from datetime import datetime
import json

def create_synthetic_dataset(num_samples=2000, img_size=224):
    """创建合成数据集用于快速测试"""
    print(f"📦 生成 {num_samples} 个合成图片样本...")
    
    # 生成随机图片数据 (batch, channels, height, width)
    images = torch.randn(num_samples, 3, img_size, img_size)
    # 生成标签 (50% real, 50% fake)
    labels = torch.cat([torch.zeros(num_samples//2), torch.ones(num_samples//2)]).long()
    
    # 打乱数据
    perm = torch.randperm(num_samples)
    images = images[perm]
    labels = labels[perm]
    
    return TensorDataset(images, labels)


def create_model(pretrained=True, num_classes=2):
    """创建ResNet-18模型"""
    print("🏗️ 构建ResNet-18模型...")
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f} - Acc: {100*correct/total:.2f}%")
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    return val_loss, val_acc


def main():
    print("\n" + "="*70)
    print("🚀 快速GPU训练 - Deepfake检测模型")
    print("="*70)
    
    # 配置
    EPOCHS = 3  # 快速训练只需3个epoch
    BATCH_SIZE = 64  # 增大batch size利用GPU
    LR = 0.001
    NUM_TRAIN = 1600
    NUM_VAL = 400
    
    # 检查GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU可用: {gpu_name}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️ GPU不可用，使用CPU训练")
    
    print(f"\n📊 训练配置:")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Learning Rate: {LR}")
    print(f"   - 训练样本: {NUM_TRAIN}")
    print(f"   - 验证样本: {NUM_VAL}")
    print(f"   - 设备: {device}")
    print()
    
    # 创建合成数据集
    start_time = time.time()
    
    train_dataset = create_synthetic_dataset(NUM_TRAIN)
    val_dataset = create_synthetic_dataset(NUM_VAL)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"✅ 数据集创建完成 ({time.time()-start_time:.1f}秒)")
    print()
    
    # 创建模型
    model = create_model(pretrained=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # 训练循环
    print("🎓 开始训练...\n")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-" * 70)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n📈 Epoch {epoch+1} 结果 (耗时: {epoch_time:.1f}秒):")
        print(f"   训练 - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"   验证 - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir = Path('deepfake_models')
            output_dir.mkdir(exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, output_dir / 'resnet18_deepfake.pth')
            print(f"   ✅ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        
        print()
    
    # 总结
    total_time = time.time() - start_time
    print("="*70)
    print("🎉 训练完成！")
    print("="*70)
    print(f"⏱️ 总耗时: {total_time/60:.1f} 分钟")
    print(f"🏆 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"📁 模型保存位置: deepfake_models/resnet18_deepfake.pth")
    
    # 保存训练历史
    output_dir = Path('deepfake_models')
    config = {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'best_val_acc': float(best_val_acc),
        'total_time_minutes': total_time/60,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'timestamp': datetime.now().isoformat(),
        'training_mode': 'quick_synthetic_data'
    }
    
    with open(output_dir / 'quick_training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(output_dir / 'quick_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ 训练配置和历史已保存")
    print("="*70)
    
    # GPU内存使用情况
    if torch.cuda.is_available():
        print(f"\n💾 GPU显存使用:")
        print(f"   已分配: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"   已缓存: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    print("\n🎯 下一步: 使用真实数据集重新训练以获得更好的效果")
    print()


if __name__ == '__main__':
    main()
