"""
Test Deepfake Detection Model Accuracy

Load trained model and perform detailed evaluation on test set
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json


class DeepfakeDataset(Dataset):
    """Custom dataset for loading deepfake images"""
    
    def __init__(self, data_dir, transform=None):
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label, img_path
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))
            if self.transform:
                image = self.transform(image)
            return image, label, img_path


def load_model(model_path, device):
    """Load trained model"""
    print(f"📦 Loading model: {model_path}")
    
    # Initialize model structure
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 2)
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model


def test_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    print("\n🧪 Starting test...")
    
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_paths


def print_detailed_results(y_true, y_pred, y_probs, paths):
    """Print detailed test results"""
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("📊 Test Results")
    print("=" * 60)
    print(f"\nOverall accuracy: {accuracy * 100:.2f}%")
    print(f"Test samples: {len(y_true)}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("         Real    Fake")
    print(f"Real     {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"Fake     {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    target_names = ['Real', 'Fake (AI-generated)']
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    
    # Prediction results for each sample
    print("\n" + "=" * 60)
    print("📋 Prediction Results for Each Sample")
    print("=" * 60)
    
    correct_count = 0
    wrong_count = 0
    
    print("\n✅ Correct predictions:")
    for i, (true_label, pred_label, prob, path) in enumerate(zip(y_true, y_pred, y_probs, paths)):
        if true_label == pred_label:
            correct_count += 1
            true_class = "Real" if true_label == 0 else "Fake"
            confidence = prob[pred_label] * 100
            filename = Path(path).name
            print(f"  {correct_count}. {filename:20s} - {true_class} (confidence: {confidence:.2f}%)")
    
    print(f"\n❌ Wrong predictions:")
    for i, (true_label, pred_label, prob, path) in enumerate(zip(y_true, y_pred, y_probs, paths)):
        if true_label != pred_label:
            wrong_count += 1
            true_class = "Real" if true_label == 0 else "Fake"
            pred_class = "Real" if pred_label == 0 else "Fake"
            confidence = prob[pred_label] * 100
            filename = Path(path).name
            print(f"  {wrong_count}. {filename:20s} - True: {true_class}, Pred: {pred_class} (confidence: {confidence:.2f}%)")
    
    if wrong_count == 0:
        print("  (No wrong predictions)")
    
    # Statistics
    print("\n" + "=" * 60)
    print("📈 Statistics Summary")
    print("=" * 60)
    print(f"Correct predictions: {correct_count}/{len(y_true)} ({correct_count/len(y_true)*100:.2f}%)")
    print(f"Wrong predictions: {wrong_count}/{len(y_true)} ({wrong_count/len(y_true)*100:.2f}%)")
    
    # Per-class statistics
    real_indices = np.where(y_true == 0)[0]
    fake_indices = np.where(y_true == 1)[0]
    
    real_correct = np.sum(y_pred[real_indices] == 0)
    fake_correct = np.sum(y_pred[fake_indices] == 1)
    
    print(f"\nReal image accuracy: {real_correct}/{len(real_indices)} ({real_correct/len(real_indices)*100:.2f}%)")
    print(f"Fake image accuracy: {fake_correct}/{len(fake_indices)} ({fake_correct/len(fake_indices)*100:.2f}%)")


def main():
    print("\n" + "=" * 60)
    print("🧪 CatSpy Deepfake Model Testing")
    print("=" * 60 + "\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set paths
    base_dir = Path(__file__).parent
    model_path = base_dir / 'deepfake_models' / 'resnet18_deepfake_custom.pth'
    test_dir = base_dir / 'data' / 'deepfake_dataset' / 'test'
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please run training script prepare_and_train_deepfake.py first")
        return
    
    # Check if test data exists
    if not test_dir.exists():
        print(f"❌ Test data directory not found: {test_dir}")
        print("Please run training script to prepare dataset first")
        return
    
    # Data preprocessing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test data
    print(f"\n📂 Loading test data: {test_dir}")
    test_dataset = DeepfakeDataset(test_dir, transform=test_transform)
    
    if len(test_dataset) == 0:
        print(f"❌ No images in test set")
        return
    
    print(f"Test set size: {len(test_dataset)} images")
    print(f"  - Real: {test_dataset.labels.count(0)}")
    print(f"  - Fake: {test_dataset.labels.count(1)}")
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Load model
    model = load_model(model_path, device)
    
    # Test model
    y_pred, y_true, y_probs, paths = test_model(model, test_loader, device)
    
    # Print detailed results
    print_detailed_results(y_true, y_pred, y_probs, paths)
    
    # Save test results
    results = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'total_samples': len(y_true),
        'correct_predictions': int(np.sum(y_pred == y_true)),
        'wrong_predictions': int(np.sum(y_pred != y_true)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'predictions': [
            {
                'file': Path(path).name,
                'true_label': 'Real' if true == 0 else 'Fake',
                'predicted_label': 'Real' if pred == 0 else 'Fake',
                'confidence': float(prob[pred] * 100),
                'correct': bool(true == pred)
            }
            for path, true, pred, prob in zip(paths, y_true, y_pred, y_probs)
        ]
    }
    
    results_path = base_dir / 'deepfake_models' / 'test_results_custom.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Test results saved to: {results_path}")
    print("\n✅ Testing complete!")


if __name__ == '__main__':
    main()
