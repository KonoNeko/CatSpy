"""
Quick GPU Training Script for Deepfake Detection - 20-minute quick training
Use synthetic data for fast training and GPU performance testing
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
    """Create synthetic dataset for quick testing"""
    print(f"📦 Generating {num_samples} synthetic image samples...")
    
    # Generate random image data (batch, channels, height, width)
    images = torch.randn(num_samples, 3, img_size, img_size)
    # Generate labels (50% real, 50% fake)
    labels = torch.cat([torch.zeros(num_samples//2), torch.ones(num_samples//2)]).long()
    
    # Shuffle data
    perm = torch.randperm(num_samples)
    images = images[perm]
    labels = labels[perm]
    
    return TensorDataset(images, labels)


def create_model(pretrained=True, num_classes=2):
    """Create ResNet-18 model"""
    print("🏗️ Building ResNet-18 model...")
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
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
    """Validate model"""
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
    print("🚀 Quick GPU Training - Deepfake Detection Model")
    print("="*70)
    
    # Configuration
    EPOCHS = 3  # Only 3 epochs for quick training
    BATCH_SIZE = 64  # Larger batch size to utilize GPU
    LR = 0.001
    NUM_TRAIN = 1600
    NUM_VAL = 400
    
    # Check GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU available: {gpu_name}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️ GPU not available, using CPU training")
    
    print(f"\n📊 Training configuration:")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Learning Rate: {LR}")
    print(f"   - Train samples: {NUM_TRAIN}")
    print(f"   - Val samples: {NUM_VAL}")
    print(f"   - Device: {device}")
    print()
    
    # Create synthetic dataset
    start_time = time.time()
    
    train_dataset = create_synthetic_dataset(NUM_TRAIN)
    val_dataset = create_synthetic_dataset(NUM_VAL)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"✅ Dataset created ({time.time()-start_time:.1f}s)")
    print()
    
    # Create model
    model = create_model(pretrained=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Training loop
    print("🎓 Starting training...\n")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Record
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n📈 Epoch {epoch+1} results (time: {epoch_time:.1f}s):")
        print(f"   Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"   Val - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # Save best model
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
            print(f"   ✅ Saved best model (val accuracy: {val_acc:.2f}%)")
        
        print()
    
    # Summary
    total_time = time.time() - start_time
    print("="*70)
    print("🎉 Training complete!")
    print("="*70)
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Best val accuracy: {best_val_acc:.2f}%")
    print(f"📁 Model saved at: deepfake_models/resnet18_deepfake.pth")
    
    # Save training history
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
    
    print(f"\n✅ Training config and history saved")
    print("="*70)
    
    # GPU memory usage
    if torch.cuda.is_available():
        print(f"\n💾 GPU VRAM usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"   Cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    print("\n🎯 Next step: Retrain with real data for better performance")
    print()


if __name__ == '__main__':
    main()
