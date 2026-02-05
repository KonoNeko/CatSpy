import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

try:
    from peft import PeftModel  # type: ignore
    _PEFT_AVAILABLE = True
except Exception as _peft_err:
    print(f"Warning: PEFT not available: {_peft_err}")
    PeftModel = None       # type: ignore
    _PEFT_AVAILABLE = False

class PhishingDetector:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2, lora_adapter_path='lora_out_full'):
        print(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Loading base model from {model_name}...")
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
        adapter_path = Path(__file__).parent / lora_adapter_path
        adapter_file = adapter_path / "adapter_model.safetensors"
        
        if _PEFT_AVAILABLE and PeftModel is not None and adapter_path.exists() and adapter_file.exists():
            try:
                print(f"Loading LoRA adapter from {adapter_path}...")
                self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
                print("LoRA adapter loaded successfully")
            except Exception as e:
                print(f"Failed to load LoRA adapter: {e}")
                print("Using base model without LoRA")
                self.model = base_model
        elif not adapter_path.exists() or not adapter_file.exists():
            print(f"LoRA adapter not found at {adapter_path}")
            print("Using base model without LoRA")
            self.model = base_model
        else:
            print("PEFT not available. Using base model")
            self.model = base_model
        
        self.model.eval()

    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        return {"label": int(pred), "scores": probs[0].tolist()}
