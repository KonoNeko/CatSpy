"""
下载真实的Deepfake数据集

推荐数据集：
1. Kaggle 140k Real and Fake Faces (2GB) - 最简单快速
2. FFHQ Dataset - 高质量真实人脸
"""

import os
from pathlib import Path

print("""
╔═══════════════════════════════════════════════════════════════╗
║      Deepfake数据集下载指南                                      ║
╚═══════════════════════════════════════════════════════════════╝

您的CSV文件包含20,000个样本，但图片文件缺失。

推荐数据集选项：

【方案1 - 最简单】Kaggle 140k Real and Fake Faces (推荐!)
├─ 大小: ~2GB
├─ 样本: 70k真实 + 70k伪造
├─ 下载: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
└─ 步骤:
   1. 访问链接，点击 "Download"
   2. 解压到: backend/data/140k-real-and-fake-faces/
   3. 运行: python prepare_140k_dataset.py

【方案2】使用CSV中的原始数据（需要Kaggle API）
├─ 数据集: FFHQ (Flickr-Faces-HQ)
├─ 大小: ~70GB (完整) 或 ~2GB (子集)
├─ 需要: Kaggle API token
└─ 命令:
   pip install kaggle
   kaggle datasets download -d xhlulu/flickrfaceshq-dataset-nvidia

【方案3】快速测试 - 生成合成数据
├─ 大小: 0 (不需要下载)
├─ 用途: 快速验证训练流程
└─ 已完成: 使用 train_deepfake_quick.py

╔═══════════════════════════════════════════════════════════════╗
║  建议: 使用方案1 - 下载140k数据集到 data/ 目录                    ║
╚═══════════════════════════════════════════════════════════════╝
""")

# 检查数据目录
data_dir = Path("data")
expected_dirs = ["test/real", "test/fake", "train/real", "train/fake"]

print("\n📂 当前数据目录状态:")
for dir_path in expected_dirs:
    full_path = data_dir / dir_path
    if full_path.exists():
        file_count = len(list(full_path.glob("*.jpg"))) + len(list(full_path.glob("*.png")))
        print(f"   ✅ {dir_path}: {file_count} 个文件")
    else:
        print(f"   ❌ {dir_path}: 不存在")

print("\n" + "="*65)
print("下一步操作:")
print("="*65)
print("1. 访问: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
print("2. 点击 'Download' 下载数据集")
print("3. 解压到: backend/data/ 目录")
print("4. 运行: python train_deepfake_quick.py  (已有模型，可以测试)")
print("   或者: 等待真实数据下载完成后用真实数据训练")
print("="*65)
