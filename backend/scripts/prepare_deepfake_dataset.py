"""
Data preparation script for deepfake detection
Downloads sample datasets and prepares them for training
"""

import os
import shutil
from pathlib import Path
import argparse


def create_dataset_structure(base_dir='data/deepfake_dataset'):
    """Create the dataset directory structure"""
    base_dir = Path(base_dir)
    
    directories = [
        base_dir / 'train' / 'real',
        base_dir / 'train' / 'fake',
        base_dir / 'test' / 'real',
        base_dir / 'test' / 'fake',
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Dataset structure created at: {base_dir}")
    return base_dir


def print_dataset_guide():
    """Print guide for obtaining deepfake datasets"""
    print("\n" + "="*70)
    print("📚 DEEPFAKE DATASET GUIDE")
    print("="*70)
    
    print("\n🎯 Recommended Public Datasets:\n")
    
    print("1. FaceForensics++ (Most Popular)")
    print("   - URL: https://github.com/ondyari/FaceForensics")
    print("   - Contains: Real faces and multiple deepfake methods")
    print("   - Size: ~500GB (full), ~30GB (compressed)")
    print("   - License: Academic use only")
    
    print("\n2. Deepfake Detection Challenge (DFDC)")
    print("   - URL: https://ai.facebook.com/datasets/dfdc/")
    print("   - Contains: 100,000+ videos")
    print("   - Size: ~470GB")
    print("   - License: Research use")
    
    print("\n3. Celeb-DF (Celeb DeepFake)")
    print("   - URL: https://github.com/yuezunli/celeb-deepfakeforensics")
    print("   - Contains: 590 real + 5,639 fake videos")
    print("   - Size: ~15GB")
    print("   - License: Research use")
    
    print("\n4. DeeperForensics-1.0")
    print("   - URL: https://github.com/EndlessSora/DeeperForensics-1.0")
    print("   - Contains: 60,000 videos")
    print("   - Size: ~200GB")
    print("   - License: Academic use")
    
    print("\n📝 Quick Start with Small Dataset:\n")
    
    print("Option A: Use Kaggle Datasets (Easier)")
    print("  1. Visit: https://www.kaggle.com/datasets")
    print("  2. Search: 'deepfake detection'")
    print("  3. Download smaller datasets (~1-5GB)")
    print("  4. Popular: '140k Real and Fake Faces'")
    
    print("\nOption B: Create Your Own Small Dataset")
    print("  1. Collect 500-1000 real face images from:")
    print("     - CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    print("     - FFHQ: https://github.com/NVlabs/ffhq-dataset")
    print("  2. Generate fake faces using:")
    print("     - ThisPersonDoesNotExist.com (StyleGAN2)")
    print("     - FaceSwap tools")
    print("  3. Split 80/20 for train/test")
    
    print("\n" + "="*70)
    print("📂 DATASET ORGANIZATION")
    print("="*70)
    
    print("\nYour dataset should be organized as:")
    print("""
data/deepfake_dataset/
├── train/
│   ├── real/          ← Training real images (e.g., 4000 images)
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── fake/          ← Training fake images (e.g., 4000 images)
│       ├── fake001.jpg
│       ├── fake002.jpg
│       └── ...
└── test/
    ├── real/          ← Test real images (e.g., 1000 images)
    │   ├── test001.jpg
    │   └── ...
    └── fake/          ← Test fake images (e.g., 1000 images)
        ├── test_fake001.jpg
        └── ...
    """)
    
    print("\n⚠️ Important Notes:")
    print("  - Images should be in JPG, PNG, or BMP format")
    print("  - Recommended image size: 224x224 or larger")
    print("  - Minimum dataset size: 1000 images (500 real + 500 fake)")
    print("  - Balanced dataset: Equal number of real and fake images")
    print("  - For better results: 10,000+ images recommended")
    
    print("\n" + "="*70)
    print("🚀 NEXT STEPS")
    print("="*70)
    
    print("\n1. Download or collect your dataset")
    print("2. Organize images into the structure above")
    print("3. Run training:")
    print("   python train_deepfake.py --epochs 20 --batch_size 32")
    
    print("\n4. For quick testing with small dataset:")
    print("   python train_deepfake.py --epochs 5 --batch_size 16")
    
    print("\n" + "="*70)


def create_sample_readme(base_dir='data/deepfake_dataset'):
    """Create a README file with instructions"""
    readme_content = """# Deepfake Detection Dataset

This directory should contain your deepfake detection training data.

## Directory Structure

```
deepfake_dataset/
├── train/
│   ├── real/     <- Place training real face images here
│   └── fake/     <- Place training deepfake images here
└── test/
    ├── real/     <- Place test real face images here
    └── fake/     <- Place test deepfake images here
```

## Dataset Requirements

- **Minimum**: 1,000 images (500 real + 500 fake)
- **Recommended**: 10,000+ images for better accuracy
- **Format**: JPG, PNG, or BMP
- **Size**: 224x224 pixels or larger
- **Balance**: Equal number of real and fake images

## Recommended Datasets

1. **FaceForensics++**: https://github.com/ondyari/FaceForensics
2. **DFDC**: https://ai.facebook.com/datasets/dfdc/
3. **Celeb-DF**: https://github.com/yuezunli/celeb-deepfakeforensics
4. **Kaggle Datasets**: Search "deepfake detection" on Kaggle

## Quick Start

For testing, you can use smaller datasets from Kaggle:
- "140k Real and Fake Faces" (good for quick testing)
- Extract and organize into train/test folders

## Training

Once you have your dataset ready:

```bash
# Train the model
python train_deepfake.py --epochs 20 --batch_size 32

# For quick testing with small dataset
python train_deepfake.py --epochs 5 --batch_size 16
```

## Notes

- All images will be automatically resized to 224x224 during training
- Data augmentation is applied automatically
- Training on GPU is highly recommended
- Expected training time: 30-60 minutes on GPU, 3-5 hours on CPU
"""
    
    readme_path = Path(base_dir) / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n✅ README created at: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for deepfake detection')
    parser.add_argument('--data_dir', type=str, default='data/deepfake_dataset',
                      help='Directory to create dataset structure')
    parser.add_argument('--guide', action='store_true',
                      help='Show detailed guide for obtaining datasets')
    
    args = parser.parse_args()
    
    # Create directory structure
    base_dir = create_dataset_structure(args.data_dir)
    
    # Create README
    create_sample_readme(base_dir)
    
    # Show guide
    if args.guide:
        print_dataset_guide()
    else:
        print("\n💡 For detailed dataset guide, run:")
        print(f"   python prepare_deepfake_dataset.py --guide")
    
    print(f"\n📁 Dataset directory ready: {base_dir}")
    print("📝 Next: Add images to the folders and run train_deepfake.py")


if __name__ == '__main__':
    main()
