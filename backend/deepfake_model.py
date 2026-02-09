"""
Deepfake Detection Model using ResNet-18
Supports image and video deepfake detection
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
import tempfile
import os


class DeepfakeDetector:
    """ResNet-18 based deepfake detector for images and videos"""
    
    def __init__(self, model_path: str = 'deepfake_models/resnet18_deepfake_custom.pth', device: str = None):
        """
        Initialize the deepfake detector
        
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        # Set deterministic inference to avoid randomness
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize ResNet-18 model
        self.model = models.resnet18(pretrained=False)
        
        # Modify final layer for binary classification (real vs fake)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)
        )
        
        # Load trained weights if available
        model_path = Path(__file__).parent / model_path
        if model_path.exists():
            try:
                print(f"Loading trained model from {model_path}...")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                    
                print("✅ Trained model loaded successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not load trained model: {e}")
                print("Using untrained ResNet-18 model (train first for accurate results)")
        else:
            print(f"⚠️ Model not found at {model_path}")
            print("Using untrained ResNet-18 model (train first for accurate results)")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class labels
        self.classes = ['real', 'fake']
    
    def predict_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Predict if an image is real or deepfake
        
        Args:
            image_input: Image file path, PIL Image, or numpy array
            
        Returns:
            Dict with prediction results
        """
        # Convert input to PIL Image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
            image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        
        result = {
            'prediction': self.classes[predicted_class],
            'confidence': round(confidence, 2),
            'probabilities': {
                'real': round(probabilities[0][0].item() * 100, 2),
                'fake': round(probabilities[0][1].item() * 100, 2)
            },
            'is_deepfake': predicted_class == 1,
            'risk_level': self._get_risk_level(predicted_class, confidence)
        }
        
        return result
    
    def predict_video(self, video_path: str, sample_rate: int = 30) -> Dict:
        """
        Predict if a video contains deepfakes by analyzing frames
        
        Args:
            video_path: Path to video file
            sample_rate: Analyze every Nth frame (default: 30)
            
        Returns:
            Dict with video analysis results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        frame_results = []
        frame_count = 0
        analyzed_frames = 0
        
        print(f"Analyzing video: {total_frames} frames, {duration:.2f}s, sampling every {sample_rate} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_count % sample_rate == 0:
                result = self.predict_image(frame)
                frame_results.append({
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps if fps > 0 else 0,
                    'prediction': result['prediction'],
                    'confidence': result['confidence']
                })
                analyzed_frames += 1
            
            frame_count += 1
        
        cap.release()
        
        # Aggregate results
        if not frame_results:
            return {
                'error': 'No frames could be analyzed',
                'video_info': {
                    'total_frames': total_frames,
                    'duration': duration,
                    'fps': fps
                }
            }
        
        fake_count = sum(1 for r in frame_results if r['prediction'] == 'fake')
        real_count = len(frame_results) - fake_count
        fake_percentage = (fake_count / len(frame_results)) * 100
        
        # Average confidence for fake predictions
        fake_confidences = [r['confidence'] for r in frame_results if r['prediction'] == 'fake']
        avg_fake_confidence = np.mean(fake_confidences) if fake_confidences else 0
        
        # Overall verdict
        overall_prediction = 'fake' if fake_percentage > 30 else 'real'
        
        return {
            'video_prediction': overall_prediction,
            'is_deepfake': overall_prediction == 'fake',
            'fake_percentage': round(fake_percentage, 2),
            'average_fake_confidence': round(avg_fake_confidence, 2),
            'risk_level': self._get_video_risk_level(fake_percentage, avg_fake_confidence),
            'statistics': {
                'total_frames_analyzed': analyzed_frames,
                'frames_detected_fake': fake_count,
                'frames_detected_real': real_count,
                'sample_rate': sample_rate
            },
            'video_info': {
                'total_frames': total_frames,
                'duration': round(duration, 2),
                'fps': round(fps, 2)
            },
            'suspicious_frames': [
                {
                    'frame': r['frame_number'],
                    'time': round(r['timestamp'], 2),
                    'confidence': r['confidence']
                }
                for r in frame_results if r['prediction'] == 'fake'
            ][:10]  # Top 10 suspicious frames
        }
    
    def _get_risk_level(self, predicted_class: int, confidence: float) -> str:
        """Determine risk level based on prediction"""
        if predicted_class == 0:  # Real
            return 'safe'
        else:  # Fake
            if confidence >= 80:
                return 'high'
            elif confidence >= 60:
                return 'medium'
            else:
                return 'low'
    
    def _get_video_risk_level(self, fake_percentage: float, confidence: float) -> str:
        """Determine video risk level"""
        if fake_percentage >= 50:
            return 'high'
        elif fake_percentage >= 30:
            return 'medium'
        elif fake_percentage >= 10:
            return 'low'
        else:
            return 'safe'


def test_detector():
    """Test the deepfake detector with a sample image"""
    print("\n=== Testing Deepfake Detector ===\n")
    
    detector = DeepfakeDetector()
    
    # Create a test image
    test_image = Image.new('RGB', (224, 224), color='blue')
    
    print("Testing with sample image...")
    result = detector.predict_image(test_image)
    
    print(f"\nResult:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"  Is Deepfake: {result['is_deepfake']}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Probabilities: Real={result['probabilities']['real']}%, Fake={result['probabilities']['fake']}%")
    
    print("\n✅ Detector initialized successfully")
    print("⚠️ Note: Model is untrained. Train using train_deepfake.py for accurate results")


if __name__ == '__main__':
    test_detector()
