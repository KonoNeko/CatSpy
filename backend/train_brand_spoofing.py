"""
Train DistilBERT with LoRA on Brand Spoofing Detection Dataset

This script trains the phishing detection model using the brand_spoofing_training_dataset_1000.csv

Usage:
    python train_brand_spoofing.py --csv_file data/brand_spoofing_training_dataset_1000.csv
    python train_brand_spoofing.py --csv_file data/brand_spoofing_training_dataset_1000.csv --epochs 5 --batch_size 16
"""
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, TaskType, get_peft_model
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
import json


def load_brand_spoofing_csv(csv_path: Path) -> Tuple[List[str], List[int]]:
    """
    Load brand spoofing dataset from CSV.
    
    CSV format:
    - url: The URL to analyze
    - brand: Brand name (Microsoft, Google, Amazon, etc.)
    - label: 0=legitimate, 1=phishing
    - spoof_type: Type of spoofing technique
    - reason: Explanation of the classification
    
    Returns:
        texts: List of text samples (URL + context)
        labels: List of labels (0=safe, 1=phishing)
    """
    df = pd.read_csv(csv_path)
    
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        url = row['url']
        brand = row['brand']
        label = int(row['label'])
        
        # Create text input combining URL and brand context
        # This helps the model learn brand-URL associations
        text = f"Visit {url} for {brand} services"
        
        texts.append(text)
        labels.append(label)
    
    return texts, labels


class BrandSpoofingDataset(torch.utils.data.Dataset):
    """Dataset for brand spoofing detection."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: DistilBertTokenizerFast, max_length: int = 256):
        self.tokenizer = tokenizer
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(pred):
    """Compute evaluation metrics."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def main():
    parser = argparse.ArgumentParser(description='Train Brand Spoofing Detection Model')
    parser.add_argument('--csv_file', type=str, default='data/brand_spoofing_training_dataset_1000.csv',
                        help='Path to brand spoofing CSV file')
    parser.add_argument('--output_dir', type=str, default='./lora_brand_spoofing',
                        help='Where to save the PEFT adapter')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print("=" * 70)
    print("BRAND SPOOFING DETECTION MODEL TRAINING")
    print("=" * 70)
    print(f"\n📊 Loading dataset: {csv_path}")
    
    # Load data
    texts, labels = load_brand_spoofing_csv(csv_path)
    
    print(f"✅ Loaded {len(texts)} samples")
    print(f"   - Legitimate (0): {labels.count(0)}")
    print(f"   - Phishing (1): {labels.count(1)}")
    
    # Split train/test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    
    print(f"\n📦 Data split:")
    print(f"   - Training: {len(train_texts)} samples")
    print(f"   - Testing: {len(test_texts)} samples")
    
    # Load tokenizer and model
    print(f"\n🔧 Loading model: {args.model_name}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    base_model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Configure LoRA
    print(f"\n⚙️  Configuring LoRA:")
    print(f"   - Rank (r): {args.r}")
    print(f"   - Alpha: {args.lora_alpha}")
    print(f"   - Dropout: {args.lora_dropout}")
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=['q_lin', 'v_lin']
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    # Create datasets
    print(f"\n📝 Tokenizing datasets (max_length={args.max_length})...")
    train_dataset = BrandSpoofingDataset(train_texts, train_labels, tokenizer, args.max_length)
    test_dataset = BrandSpoofingDataset(test_texts, test_labels, tokenizer, args.max_length)
    
    # Training arguments
    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        seed=42,
    )
    
    # Create trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    
    train_result = trainer.train()
    
    # Evaluate
    print("\n" + "=" * 70)
    print("📊 FINAL EVALUATION")
    print("=" * 70)
    
    eval_results = trainer.evaluate()
    
    print(f"\n✅ Training completed!")
    print(f"\n📈 Results:")
    print(f"   - Accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"   - Precision: {eval_results['eval_precision']:.4f}")
    print(f"   - Recall:    {eval_results['eval_recall']:.4f}")
    print(f"   - F1 Score:  {eval_results['eval_f1']:.4f}")
    
    # Save model
    print(f"\n💾 Saving model to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training report
    report_path = output_dir / "training_report.json"
    report = {
        "dataset": str(csv_path),
        "total_samples": len(texts),
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "model": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lora_config": {
            "r": args.r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout
        },
        "results": {
            "accuracy": eval_results['eval_accuracy'],
            "precision": eval_results['eval_precision'],
            "recall": eval_results['eval_recall'],
            "f1": eval_results['eval_f1']
        }
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Training report saved: {report_path}")
    print("\n" + "=" * 70)
    print("✨ ALL DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
