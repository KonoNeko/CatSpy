

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import os


class DeepfakeDataset(Dataset):
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append(str(img_path))
                    self.labels.append(0)
        
        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append(str(img_path))
                    self.labels.append(1)
        
        print(f"Loaded {len(self.samples)} images from {data_dir}")
        print(f"  - Real: {self.labels.count(0)}")
        print(f"  - Fake: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))
            if self.transform:
                image = self.transform(image)
            return image, label


def prepare_dataset():
    """
    Prepare training and test datasets
    Organize images into proper folder structure
    """
    print("=" * 60)
    print("📁 Preparing dataset...")
    print("=" * 60)
    
    base_dir = Path(__file__).parent / 'data'
    source_dirs = {
        'pic-test': base_dir / 'pic-test',
        'pic0': base_dir / 'pic0',
        'pic1': base_dir / 'pic1'
    }
    
    dataset_dir = base_dir / 'deepfake_dataset'
    train_dir = dataset_dir / 'train'
    test_dir = dataset_dir / 'test'
    
    if dataset_dir.exists():
        print(f"Cleaning old dataset: {dataset_dir}")
        shutil.rmtree(dataset_dir)
    
    for split in ['train', 'test']:
        for category in ['real', 'fake']:
            dir_path = dataset_dir / split / category
            dir_path.mkdir(parents=True, exist_ok=True)
    
    print("\n📋 Data allocation rules:")
    print("  - pic-test/000001-000005.jpg → fake (AI-generated)")
    print("  - pic-test/000021-000025.jpg → real")
    print("  - pic0/* → real")
    print("  - pic1/* → fake (AI-generated)")
    
    # Stats tracking
    train_real_count = 0
    train_fake_count = 0
    test_real_count = 0
    test_fake_count = 0
    
    # Process pic-test folder
    pic_test_dir = source_dirs['pic-test']
    if pic_test_dir.exists():
        print(f"\nProcessing {pic_test_dir}...")
        for img_file in pic_test_dir.glob('*.jpg'):
            filename = img_file.name
            # Extract file number
            num = int(filename.replace('.jpg', ''))
            
            if 1 <= num <= 5:
                # Fake (AI-generated) - add to test set
                dest = test_dir / 'fake' / filename
                shutil.copy2(img_file, dest)
                test_fake_count += 1
                print(f"  ✓ {filename} → test/fake (AI-generated)")
            elif 21 <= num <= 25:
                # Real - add to test set
                dest = test_dir / 'real' / filename
                shutil.copy2(img_file, dest)
                test_real_count += 1
                print(f"  ✓ {filename} → test/real")
    
    # Process pic0 folder (all real)
    pic0_dir = source_dirs['pic0']
    if pic0_dir.exists():
        print(f"\nProcessing {pic0_dir} (all real images)...")
        for img_file in pic0_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Add to training set
                dest = train_dir / 'real' / img_file.name
                shutil.copy2(img_file, dest)
                train_real_count += 1
        print(f"  ✓ Copied {train_real_count} images to train/real")
    
    # Process pic1 folder (all fake/AI-generated)
    pic1_dir = source_dirs['pic1']
    if pic1_dir.exists():
        print(f"\nProcessing {pic1_dir} (all AI-generated images)...")
        for img_file in pic1_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Add to training set
                dest = train_dir / 'fake' / img_file.name
                shutil.copy2(img_file, dest)
                train_fake_count += 1
        print(f"  ✓ Copied {train_fake_count} images to train/fake")
    
    print("\n" + "=" * 60)
    print("📊 Dataset preparation complete!")
    print("=" * 60)
    print(f"Training set:")
    print(f"  - Real: {train_real_count} images")
    print(f"  - Fake: {train_fake_count} images")
    print(f"  - Total: {train_real_count + train_fake_count} images")
    print(f"\nTest set:")
    print(f"  - Real: {test_real_count} images")
    print(f"  - Fake: {test_fake_count} images")
    print(f"  - Total: {test_real_count + test_fake_count} images")
    print(f"\nDataset directory: {dataset_dir}")
    print("=" * 60 + "\n")
    
    return dataset_dir


def train_model(data_dir, epochs=20, batch_size=8, lr=0.001):
    """Train deepfake detection model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting training (device: {device})")
    print(f"Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Data directory: {data_dir}\n")
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  
    # Load datasets
    train_dir = Path(data_dir) / 'train'
    test_dir = Path(data_dir) / 'test'
    
    train_dataset = DeepfakeDataset(train_dir, transform=train_transform)
    test_dataset = DeepfakeDataset(test_dir, transform=test_transform)
    
    if len(train_dataset) == 0:
        print(f"❌ No images in training set: {train_dir}")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    
    # Initialize model
    print("📦 Initializing ResNet-18 model...")
    model = models.resnet18(pretrained=True)
    
    # Modify final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 2)
    )
    
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=3)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr
    }
    
    best_acc = 0.0
    best_epoch = 0
    
    # Training loop
    print(f"\n🎯 Starting training for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, labels in pbar:
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
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Test]')
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{test_loss/len(pbar):.4f}',
                    'acc': f'{100*correct/total:.2f}%'
                })
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Print results
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # Learning rate adjustment
        scheduler.step(test_loss)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            model_path = Path(__file__).parent / 'deepfake_models' / 'resnet18_deepfake_custom.pth'
            model_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ Saved new best model (test accuracy: {best_acc:.2f}%)")
    
    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print("=" * 60)
    print(f"Best test accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    
    # Save training history
    history_path = Path(__file__).parent / 'deepfake_models' / 'training_history_custom.json'
    history['best_accuracy'] = best_acc
    history['best_epoch'] = best_epoch
    history['timestamp'] = datetime.now().isoformat()
    
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"Training history saved to: {history_path}")
    print(f"Model saved to: {model_path}")
    
    # Plot training curves
    plot_training_history(history)
    
    return model, history


def plot_training_history(history):
    """Plot training history curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_title('Training and Test Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs_range, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs_range, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_title('Training and Test Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(__file__).parent / 'deepfake_models' / 'training_history_custom.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {plot_path}")
    
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 CatSpy Deepfake Detection Model Training")
    print("="*60 + "\n")
    
    # Step 1: Prepare dataset
    dataset_dir = prepare_dataset()
    
    # Step 2: Train model
    print("\nStarting training...")
    
    model, history = train_model(
        data_dir=dataset_dir,
        epochs=30,        # Adjust as needed
        batch_size=8,     # Reduce if out of memory
        lr=0.0001         # Learning rate
    )
    
    print("\n✅ All done!")
