import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

class PhishingDetector:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1)
        self.model = get_peft_model(self.model, lora_config)

    def train(self, texts, labels, epochs=2, lr=2e-5):
        # 简化训练流程，实际可扩展为完整训练
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor(labels)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        return loss.item()

    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
        return pred
