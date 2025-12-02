"""
LoRA 模型评估脚本
用于评估经过 LoRA 微调后的分类模型在测试集上的性能

使用方法:
    python evaluate_lora_model.py \
        --base_model_name_or_path distilbert-base-uncased \
        --lora_adapter_path ./lora_out \
        --test_file data/val_samples.jsonl \
        --batch_size 16 \
        --max_length 256
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class TextClassificationDataset(Dataset):
    """文本分类数据集"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        """
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: HuggingFace tokenizer
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_model_and_tokenizer(base_model_path: str, lora_adapter_path: str, device: str) -> Tuple:
    """
    加载基座模型、LoRA 适配器和 tokenizer
    
    Args:
        base_model_path: 基座模型路径或名称
        lora_adapter_path: LoRA 适配器权重路径
        device: 运行设备 (cuda/cpu)
    
    Returns:
        (model, tokenizer)
    """
    print(f"📦 Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    print(f"📦 Loading base model from {base_model_path}...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=2  
    )
    
    
    lora_path = Path(lora_adapter_path)
    if lora_path.exists() and (lora_path / "adapter_model.safetensors").exists():
        print(f"🔧 Loading LoRA adapter from {lora_adapter_path}...")
        try:
            
            model = PeftModel.from_pretrained(base_model, lora_adapter_path)
            print("✅ LoRA adapter loaded successfully")
            
            
            # model = model.merge_and_unload()
            # print("✅ LoRA weights merged into base model")
            
        except Exception as e:
            print(f"⚠️  Failed to load LoRA adapter: {e}")
            print("⚠️  Using base model without LoRA")
            model = base_model
    else:
        print(f"⚠️  No LoRA adapter found at {lora_adapter_path}")
        print("⚠️  Using base model without LoRA")
        model = base_model
    
    model.to(device)
    model.eval()  
    
    return model, tokenizer


def load_dataset_from_file(file_path: str, text_column: str = "text", label_column: str = "label") -> Tuple[List[str], List[int]]:
    """
    从本地文件加载测试数据集
    
    Args:
        file_path: 数据文件路径
        text_column: 文本字段名
        label_column: 标签字段名
    
    Returns:
        (texts, labels)
    """
    print(f"📂 Loading test data from {file_path}...")
    
    texts = []
    labels = []
    
    file_path = Path(file_path)
    
    
    if file_path.suffix == '.jsonl':
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get(text_column) or obj.get('sentence') or obj.get('input') or ''
                    label = int(obj.get(label_column, 0))
                    texts.append(text)
                    labels.append(label)
                except json.JSONDecodeError:
                    continue
    
    elif file_path.suffix == '.json':
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for obj in data:
                text = obj.get(text_column, '')
                label = int(obj.get(label_column, 0))
                texts.append(text)
                labels.append(label)
    
    elif file_path.suffix == '.csv':
        
        import csv
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get(text_column, '')
                label = int(row.get(label_column, 0))
                texts.append(text)
                labels.append(label)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    print(f"✅ Loaded {len(texts)} samples")
    return texts, labels


def evaluate(
    model,
    dataloader: DataLoader,
    device: str,
    id_to_label: Dict[int, str] = None
) -> Tuple[List[int], List[int]]:
    """
    在测试集上运行推理并收集预测结果
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 运行设备
        id_to_label: ID 到标签的映射（可选）
    
    Returns:
        (y_true, y_pred) 真实标签和预测标签
    """
    print("\n🔍 Running inference on test set...")
    
    y_true = []
    y_pred = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            
            predictions = torch.argmax(logits, dim=-1)
            
            
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
    
    return y_true, y_pred


def compute_metrics(y_true: List[int], y_pred: List[int], id_to_label: Dict[int, str] = None):
    """
    计算并打印各项评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        id_to_label: ID 到标签名称的映射
    """
    print("\n" + "="*60)
    print("📊 EVALUATION METRICS")
    print("="*60)
    
    
    if id_to_label is None:
        id_to_label = {0: "Safe", 1: "Phishing"}
    
    target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    
    # 1. Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    print(f"\n📈 Binary Metrics (Phishing class):")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    print(f"\n📈 Macro Metrics (All classes):")
    print(f"   Precision: {precision_macro:.4f}")
    print(f"   Recall:    {recall_macro:.4f}")
    print(f"   F1-Score:  {f1_macro:.4f}")
    
    
    print(f"\n📋 Detailed Classification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print("-" * 60)
    print(f"                 Predicted")
    print(f"                 {target_names[0]:<12} {target_names[1]:<12}")
    print(f"Actual")
    for i, label in enumerate(target_names):
        print(f"{label:<12}  {cm[i][0]:<12}  {cm[i][1]:<12}")
    
    print("\n" + "="*60)
    
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate LoRA fine-tuned model on test set')
    
    
    parser.add_argument(
        '--base_model_name_or_path',
        type=str,
        default='distilbert-base-uncased',
        help='Base model name or path (e.g., distilbert-base-uncased)'
    )
    parser.add_argument(
        '--lora_adapter_path',
        type=str,
        default='./lora_out',
        help='Path to LoRA adapter weights'
    )
    
    
    parser.add_argument(
        '--test_file',
        type=str,
        default='data/val_samples.jsonl',
        help='Path to test dataset file (JSONL/JSON/CSV)'
    )
    parser.add_argument(
        '--text_column',
        type=str,
        default='text',
        help='Name of the text column in the dataset'
    )
    parser.add_argument(
        '--label_column',
        type=str,
        default='label',
        help='Name of the label column in the dataset'
    )
    
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for inference (default: 16)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=256,
        help='Maximum sequence length for tokenization (default: 256)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 LoRA Model Evaluation")
    print("="*60)
    print(f"Base Model:      {args.base_model_name_or_path}")
    print(f"LoRA Adapter:    {args.lora_adapter_path}")
    print(f"Test File:       {args.test_file}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Max Length:      {args.max_length}")
    print(f"Device:          {args.device}")
    print("="*60 + "\n")
    
    
    id_to_label = {0: "Safe", 1: "Phishing"}
    
    
    model, tokenizer = load_model_and_tokenizer(
        args.base_model_name_or_path,
        args.lora_adapter_path,
        args.device
    )
    
    
    texts, labels = load_dataset_from_file(
        args.test_file,
        args.text_column,
        args.label_column
    )
    
    
    test_dataset = TextClassificationDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  
        num_workers=0   
    )
    
    
    y_true, y_pred = evaluate(model, test_dataloader, args.device, id_to_label)
    
    
    metrics = compute_metrics(y_true, y_pred, id_to_label)
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
