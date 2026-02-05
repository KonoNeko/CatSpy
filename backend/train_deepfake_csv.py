"""
Optimized Training Script for ResNet-18 Deepfake Detection (CSV-based)

Usage:
    python train_deepfake_csv.py --csv data/deepfaketest.csv --epochs 20 --batch_size 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class DeepfakeCSVDataset(Dataset):
    """Efficient CSV-based dataset loader"""
    
    def __init__(self, csv_file, img_base_dir=None, transform=None):
        """
        Args:
            csv_file: Path to CSV with columns: 'path', 'label_str' (or 'label')
            img_base_dir: Base directory for images (if paths are relative)
            transform: Image transformations
        """
        self.df = pd.read_csv(csv_file)
        self.img_base_dir = Path(img_base_dir) if img_base_dir else Path(csv_file).parent.parent
        self.transform = transform
        
        # Map labels: real=0, fake=1
        if 'label_str' in self.df.columns:
            self.df['label'] = (self.df['label_str'] == 'fake').astype(int)
        elif 'label' not in self.df.columns:
            raise ValueError("CSV must have 'label' or 'label_str' column")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_base_dir / row['path']
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, row['label']
        except Exception as e:
            # Return a blank image if file not found
            print(f"⚠️ Error loading {img_path}: {e}")
            blank = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                blank = self.transform(blank)
            return blank, row['label']


def get_transforms(augment=True):
    """Get training and validation transforms"""
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_model(pretrained=True, num_classes=2):
    """Create ResNet-18 model"""
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with mixed precision"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    return val_loss, val_acc, all_preds, all_labels, all_probs


def plot_training_history(history, save_dir):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_probs, save_dir):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Deepfake Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train ResNet-18 Deepfake Detector')
    parser.add_argument('--csv', type=str, default='data/deepfaketest.csv',
                        help='Path to CSV file with image paths and labels')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='Base directory for images (default: infer from CSV location)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Training data split ratio (default: 0.8)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet pretrained weights')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation')
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                        help='Use mixed precision training (faster on GPU)')
    parser.add_argument('--early_stop', type=int, default=5,
                        help='Early stopping patience (default: 5)')
    parser.add_argument('--output_dir', type=str, default='deepfake_models',
                        help='Output directory for model and plots')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"🚀 Starting Deepfake Detection Training")
    print(f"{'='*60}")
    print(f"📊 CSV File: {args.csv}")
    print(f"💻 Device: {device}")
    print(f"🔢 Batch Size: {args.batch_size}")
    print(f"📈 Epochs: {args.epochs}")
    print(f"🎯 Learning Rate: {args.lr}")
    print(f"🔄 Pretrained: {args.pretrained}")
    print(f"🎨 Data Augmentation: {args.augment}")
    print(f"⚡ Mixed Precision: {args.mixed_precision}")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Prepare datasets
    print("📂 Loading dataset...")
    train_transform, val_transform = get_transforms(augment=args.augment)
    
    # Load full dataset
    full_dataset = DeepfakeCSVDataset(
        csv_file=args.csv,
        img_base_dir=args.img_dir,
        transform=train_transform
    )
    
    # Split into train/val
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update val dataset transform
    val_dataset.dataset.transform = val_transform
    
    print(f"✅ Total samples: {len(full_dataset)}")
    print(f"   - Training: {train_size}")
    print(f"   - Validation: {val_size}\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("🏗️ Building model...")
    model = create_model(pretrained=args.pretrained, num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Mixed precision
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    
    # Training loop
    print("\n🎓 Starting training...\n")
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, output_dir / 'resnet18_deepfake.pth')
            print(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{args.early_stop})")
        
        print()
        
        # Early stopping
        if patience_counter >= args.early_stop:
            print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("🎯 Final Evaluation")
    print(f"{'='*60}")
    
    # Load best model
    checkpoint = torch.load(output_dir / 'resnet18_deepfake.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    val_loss, val_acc, val_preds, val_labels, val_probs = validate(
        model, val_loader, criterion, device
    )
    
    # Metrics
    print(f"\n📊 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"\n📈 Classification Report:")
    print(classification_report(val_labels, val_preds, 
                                target_names=['Real', 'Fake'], 
                                digits=4))
    
    # ROC-AUC
    auc_score = roc_auc_score(val_labels, val_probs)
    print(f"🎯 ROC-AUC Score: {auc_score:.4f}")
    
    # Save plots
    print("\n📊 Generating visualizations...")
    plot_training_history(history, output_dir)
    plot_confusion_matrix(val_labels, val_preds, output_dir)
    plot_roc_curve(val_labels, val_probs, output_dir)
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config = {
        'csv_file': args.csv,
        'epochs': epoch + 1,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'best_val_acc': float(best_val_acc),
        'auc_score': float(auc_score),
        'pretrained': args.pretrained,
        'augmentation': args.augment,
        'train_samples': train_size,
        'val_samples': val_size,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"📁 Model saved to: {output_dir / 'resnet18_deepfake.pth'}")
    print(f"📊 Visualizations saved to: {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
