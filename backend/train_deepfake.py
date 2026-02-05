"""
Training script for ResNet-18 Deepfake Detection Model

This script trains a ResNet-18 model to detect deepfake images.
Dataset should be organized as:
    data/deepfake_dataset/
        train/
            real/
            fake/
        test/
            real/
            fake/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class DeepfakeDataset(Dataset):
    """Custom dataset for loading deepfake images"""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory with 'real' and 'fake' subdirectories
            transform: Transformations to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Load real images (label 0)
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append(str(img_path))
                    self.labels.append(0)
        
        # Load fake images (label 1)
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
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224))
            if self.transform:
                image = self.transform(image)
            return image, label


def create_sample_dataset(output_dir='data/deepfake_dataset'):
    """
    Create a sample dataset structure for demonstration
    Users should replace this with real deepfake datasets
    """
    output_dir = Path(output_dir)
    
    for split in ['train', 'test']:
        for category in ['real', 'fake']:
            dir_path = output_dir / split / category
            dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✅ Sample dataset structure created at: {output_dir}")
    print("\n📁 Directory structure:")
    print("data/deepfake_dataset/")
    print("├── train/")
    print("│   ├── real/  ← Place real face images here")
    print("│   └── fake/  ← Place deepfake images here")
    print("└── test/")
    print("    ├── real/  ← Place test real images here")
    print("    └── fake/  ← Place test deepfake images here")
    
    print("\n📌 Recommended datasets:")
    print("  - FaceForensics++: https://github.com/ondyari/FaceForensics")
    print("  - DFDC (Deepfake Detection Challenge): https://ai.facebook.com/datasets/dfdc/")
    print("  - Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics")
    
    return output_dir


def train_model(args):
    """Train the deepfake detection model"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Starting training on {device}")
    print(f"Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Data directory: {args.data_dir}")
    
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
    train_dir = Path(args.data_dir) / 'train'
    test_dir = Path(args.data_dir) / 'test'
    
    if not train_dir.exists():
        print(f"\n❌ Training directory not found: {train_dir}")
        print("Creating sample dataset structure...")
        create_sample_dataset(args.data_dir)
        print("\n⚠️ Please add images to the dataset folders and run again.")
        return
    
    train_dataset = DeepfakeDataset(train_dir, transform=train_transform)
    test_dataset = DeepfakeDataset(test_dir, transform=test_transform)
    
    if len(train_dataset) == 0:
        print(f"\n❌ No training images found in {train_dir}")
        print("Please add images to the dataset folders and run again.")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    
    # Initialize model
    print("\n📦 Initializing ResNet-18 model...")
    model = models.resnet18(pretrained=True)
    
    # Modify final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 2)
    )
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=3, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_acc = 0.0
    
    # Training loop
    print(f"\n🎯 Starting training for {args.epochs} epochs...\n")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
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
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Test]')
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{test_loss/len(pbar):.4f}',
                    'acc': f'{100*correct/total:.2f}%'
                })
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = output_dir / 'resnet18_deepfake.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, model_path)
            
            print(f'  ✅ Best model saved! Accuracy: {best_acc:.2f}%')
    
    # Final evaluation
    print(f"\n📊 Final Evaluation:")
    print(f"  Best Test Accuracy: {best_acc:.2f}%")
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=['Real', 'Fake'], 
                              digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = Path(args.output_dir) / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"  Confusion matrix saved to: {cm_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    history_path = Path(args.output_dir) / 'training_history.png'
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    print(f"  Training history saved to: {history_path}")
    
    # Save training report
    report = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'device': str(device)
        },
        'dataset': {
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset)
        },
        'results': {
            'best_test_accuracy': best_acc,
            'final_train_loss': history['train_loss'][-1],
            'final_test_loss': history['test_loss'][-1],
            'final_train_acc': history['train_acc'][-1],
            'final_test_acc': history['test_acc'][-1]
        },
        'history': history
    }
    
    report_path = Path(args.output_dir) / 'training_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"  Training report saved to: {report_path}")
    print(f"\n✅ Training completed successfully!")
    print(f"  Model saved to: {Path(args.output_dir) / 'resnet18_deepfake.pth'}")


def main():
    parser = argparse.ArgumentParser(description='Train ResNet-18 Deepfake Detection Model')
    parser.add_argument('--data_dir', type=str, default='data/deepfake_dataset',
                      help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='deepfake_models',
                      help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--create_structure', action='store_true',
                      help='Create sample dataset structure and exit')
    
    args = parser.parse_args()
    
    if args.create_structure:
        create_sample_dataset(args.data_dir)
        return
    
    train_model(args)


if __name__ == '__main__':
    main()
