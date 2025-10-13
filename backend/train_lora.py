"""
Fine-tune DistilBERT with LoRA (PEFT) on a small JSONL dataset.

Usage example:
  python train_lora.py --train_file ../data/train_samples.jsonl --output_dir ./lora_out --epochs 3 --batch_size 8

Input JSONL format: each line is a JSON object with fields {"text": "...", "label": 0}

This script is intentionally simple and suitable for small experiments on CPU/GPU.
"""
import argparse
import json
from pathlib import Path
from typing import List

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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def read_jsonl(path: Path):
    texts = []
    labels = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            texts.append(j.get('text') or j.get('sentence') or j.get('input') or '')
            labels.append(int(j.get('label', 0)))
    return texts, labels


class TextDataset(torch.utils.data.Dataset):
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


def find_default_target_modules(model):
    # Heuristic: pick module *names* that correspond to supported module types.
    # PEFT/LoRA currently supports Linear/Embedding/Conv/Conv1D/MultiheadAttention-like modules.
    # We will iterate model.named_modules(), inspect the class name, and choose the last
    # attribute name for modules whose class is supported. Avoid activation modules (GELU, Relu, etc.).
    supported = ('Linear', 'Embedding', 'Conv1d', 'Conv1D', 'MultiheadAttention', 'Conv2d', 'Conv3d', 'Conv1d')
    candidate = []
    seen = set()
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if any(s in cls_name for s in supported):
            parts = name.split('.')
            last = parts[-1]
            if last and last not in seen:
                candidate.append(last)
                seen.add(last)

    # If detection produced nothing reasonable, fall back to common DistilBERT projection names
    if not candidate:
        return ['q_lin', 'v_lin']
    return candidate


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def main():
    parser = argparse.ArgumentParser(description='Train DistilBERT with LoRA (PEFT)')
    parser.add_argument('--train_file', type=str, required=True, help='Path to train JSONL file')
    parser.add_argument('--output_dir', type=str, default='./lora_out', help='Where to save the PEFT adapter')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--target_modules', type=str, default='', help='Comma-separated target module names for LoRA (optional)')
    parser.add_argument('--eval_file', type=str, default='', help='Optional JSONL file for evaluation')
    parser.add_argument('--eval_split', type=float, default=0.1, help='If --eval_file not provided, split train file by this fraction for evaluation')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    args = parser.parse_args()

    train_path = Path(args.train_file)
    assert train_path.exists(), f"Train file not found: {train_path}"

    print('Loading data...')
    texts, labels = read_jsonl(train_path)
    print(f'Loaded {len(texts)} samples')

    print('Preparing tokenizer and model...')
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    base_model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Determine target modules for LoRA
    if args.target_modules:
        target_modules = [t.strip() for t in args.target_modules.split(',') if t.strip()]
    else:
        target_modules = find_default_target_modules(base_model)

    print('Using LoRA target modules:', target_modules)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    print('Applying PEFT/LoRA...')
    try:
        model = get_peft_model(base_model, lora_config)
    except ValueError as e:
        # Common failure: detected target module types include unsupported classes like GELUActivation
        msg = str(e)
        print('PEFT/LoRA injection failed:', msg)
        print('Retrying with a conservative default target_modules: ["q_lin","v_lin","lin"]')
        lora_config.target_modules = ['q_lin', 'v_lin', 'lin']
        try:
            model = get_peft_model(base_model, lora_config)
        except Exception as e2:
            print('Retry also failed:', e2)
            raise ValueError(
                'PEFT/LoRA adapter injection failed. Please specify a valid --target_modules (comma-separated) ' 
                'that map to Linear/Embedding/MultiheadAttention module names in the model. Example: --target_modules q_lin,v_lin'
            )

    # Print trainable params summary
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f'Trainable params: {trainable} / {total} ({100*trainable/total:.2f}%)')

    # Prepare datasets (train / eval)
    if args.eval_file:
        eval_path = Path(args.eval_file)
        assert eval_path.exists(), f"Eval file not found: {eval_path}"
        eval_texts, eval_labels = read_jsonl(eval_path)
        train_texts, train_labels = texts, labels
    else:
        # split the provided train file into train/val
        if args.eval_split and 0.0 < args.eval_split < 0.5:
            train_texts, eval_texts, train_labels, eval_labels = train_test_split(
                texts, labels, test_size=args.eval_split, stratify=labels if len(set(labels))>1 else None, random_state=42
            )
        else:
            train_texts, train_labels = texts, labels
            eval_texts, eval_labels = [], []

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
    eval_dataset = TextDataset(eval_texts, eval_labels, tokenizer, max_length=args.max_length) if eval_texts else None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Enable fp16 if a CUDA device is available
    use_fp16 = torch.cuda.is_available()
    print(f'CUDA available: {torch.cuda.is_available()}, using fp16={use_fp16}')

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.lr,
        logging_steps=50,
        evaluation_strategy='epoch' if eval_dataset is not None else 'no',
        save_strategy='epoch' if eval_dataset is not None else 'no',
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=3,
        fp16=use_fp16,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print('Starting training...')
    trainer.train()

    if eval_dataset is not None:
        print('Evaluating on validation set...')
        res = trainer.evaluate(eval_dataset=eval_dataset)
        print('Eval results:', res)

    print('Saving PEFT adapter to', args.output_dir)
    model.save_pretrained(args.output_dir)

    print('Done.')


if __name__ == '__main__':
    main()
