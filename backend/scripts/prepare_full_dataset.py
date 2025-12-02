"""
准备完整的训练集和测试集
从完整数据集中划分 80% 训练，20% 测试
"""
import json
import random
from pathlib import Path

def prepare_dataset():
    # 读取完整数据集
    data_file = Path('data/dataset_phishing.jsonl')
    
    print('📂 读取完整数据集...')
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f'✅ 加载了 {len(data)} 个样本')
    
    # 统计标签分布
    label_counts = {}
    for item in data:
        label = item.get('label', 0)
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f'📊 标签分布: {label_counts}')
    
    # 按标签分组
    safe_samples = [item for item in data if item.get('label', 0) == 0]
    phishing_samples = [item for item in data if item.get('label', 0) == 1]
    
    print(f'   Safe: {len(safe_samples)} 样本')
    print(f'   Phishing: {len(phishing_samples)} 样本')
    
    # 打乱数据
    random.seed(42)
    random.shuffle(safe_samples)
    random.shuffle(phishing_samples)
    
    # 80/20 划分
    split_ratio = 0.8
    safe_split = int(len(safe_samples) * split_ratio)
    phishing_split = int(len(phishing_samples) * split_ratio)
    
    train_data = safe_samples[:safe_split] + phishing_samples[:phishing_split]
    test_data = safe_samples[safe_split:] + phishing_samples[phishing_split:]
    
    # 再次打乱
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    print(f'\n📝 数据划分:')
    print(f'   训练集: {len(train_data)} 样本')
    print(f'   测试集: {len(test_data)} 样本')
    
    # 保存训练集
    train_file = Path('data/train_full.jsonl')
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'✅ 训练集已保存到 {train_file}')
    
    # 保存测试集
    test_file = Path('data/test_full.jsonl')
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'✅ 测试集已保存到 {test_file}')
    
    # 统计新的标签分布
    train_labels = [item['label'] for item in train_data]
    test_labels = [item['label'] for item in test_data]
    
    print(f'\n📊 训练集标签分布:')
    print(f'   Safe: {train_labels.count(0)} ({train_labels.count(0)/len(train_labels)*100:.1f}%)')
    print(f'   Phishing: {train_labels.count(1)} ({train_labels.count(1)/len(train_labels)*100:.1f}%)')
    
    print(f'\n📊 测试集标签分布:')
    print(f'   Safe: {test_labels.count(0)} ({test_labels.count(0)/len(test_labels)*100:.1f}%)')
    print(f'   Phishing: {test_labels.count(1)} ({test_labels.count(1)/len(test_labels)*100:.1f}%)')

if __name__ == '__main__':
    prepare_dataset()
