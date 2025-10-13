import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Make PEFT optional to avoid import-time crashes when versions are incompatible
try:
    from peft import get_peft_model, LoraConfig, TaskType  # type: ignore
    _PEFT_AVAILABLE = True
except Exception as _peft_err:
    print(f"Warning: PEFT not available or incompatible: {_peft_err}. Proceeding without PEFT/LoRA.")
    get_peft_model = None  # type: ignore
    LoraConfig = None      # type: ignore
    TaskType = None        # type: ignore
    _PEFT_AVAILABLE = False

class PhishingDetector:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        # Try to apply LoRA/PEFT only if available. If it fails, continue without PEFT so the API can run.
        if _PEFT_AVAILABLE and LoraConfig is not None and get_peft_model is not None and TaskType is not None:
            try:
                lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1)
                self.model = get_peft_model(self.model, lora_config)
            except ValueError as e:
                # Common failure: missing/invalid target_modules; continue without PEFT
                print(f"Warning: PEFT/LoRA not applied: {e}. Running without PEFT.")
            except Exception as e:
                # Other PEFT-related issues shouldn't block running the API
                print(f"Warning: unexpected error applying PEFT/LoRA: {e}. Running without PEFT.")

    def train(self, texts, labels, epochs=2, lr=2e-5):
    
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
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        # Return both predicted label and probability scores (list)
        return {"label": int(pred), "scores": probs[0].tolist()}
