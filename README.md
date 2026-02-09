# CatSpy

Phishing Email Detection + Deepfake Recognition System

## Features

- **Phishing Detection**: DistilBERT + LoRA (96.41% accuracy)
- **Deepfake Detection**: ResNet-18 (100% test accuracy)
- **Email Scanning**: Gmail API integration, automated risk assessment
- **Brand Spoofing**: Multi-layer domain and content analysis

## Quick Start

**Requirements**: Python 3.8+, 4GB RAM

```bash
git clone https://github.com/KonoNeko/CatSpy.git
cd CatSpy/backend
.\start.ps1  # Auto-create virtual environment and start
```

Open browser: `http://localhost:8000`

Test: Click "Test Data (60)" in Email Security tab to load test data

## Model Performance

**Phishing Detection** (11,430 samples full training)
- Accuracy: 96.41% | Precision: 97.54% | Recall: 95.22% | F1: 96.37%
- Error rate: 3.59% (FP 2.4%, FN 4.8%)
- Model: DistilBERT + LoRA (739K trainable params, only 1.09% of total)
- Model size: 2.8 MB (LoRA adapter only)

**Deepfake Detection**
- Test accuracy: 100%
- Model: ResNet-18

## Model Training

### Data Preparation
JSONL format (`data/train.jsonl`, `data/test.jsonl`):
```json
{"text": "Urgent! Verify account: http://phish.com", "label": 1}
{"text": "Meeting at 3 PM tomorrow", "label": 0}
```

### Training
```bash
cd backend
.\.venv\Scripts\python.exe train_lora.py --train_file data/train_full.jsonl --eval_file data/test_full.jsonl --output_dir lora_out_full --epochs 5
```

Parameters: `--batch_size 16` `--lr 2e-4` `--max_length 256`

### Evaluation
```bash
.\.venv\Scripts\python.exe evaluate_lora_model.py --lora_adapter_path lora_out_full --test_file data/test_full.jsonl
```

Detailed report: [TRAINING_REPORT.md](./backend/TRAINING_REPORT.md)

## API

**Base**: `http://localhost:8000`

### Detect Text/URL
```bash
POST /predict
{"text": "Click here to verify your PayPal account!"}
```
Returns: `result` (SAFE/SUSPICIOUS/PHISHING), `score` (0-100), `confidence`, `risk_indicators`

### Email Scanning
```bash
POST /api/scan              # Scan Gmail (requires Gmail API config)
POST /api/load-test-data    # Load 60 test emails
GET  /api/risk-summary      # Risk statistics
GET  /api/suspicious-emails # Suspicious email list (?limit=50&min_risk_score=0.5)
DELETE /api/clear-database  # Clear database
```

### System
```bash
GET /health      # Health check
GET /api/status  # System status (model type, Gmail connection)
```

## Project Structure

```
backend/
  ├── api.py                 # FastAPI main
  ├── model.py               # DistilBERT + LoRA model
  ├── deepfake_model.py      # ResNet-18 Deepfake detection
  ├── utils.py               # Risk scoring & heuristic analysis
  ├── train_lora.py          # Training script
  ├── evaluate_lora_model.py # Evaluation script
  ├── start.ps1              # Startup script
  ├── lora_out_full/         # Trained LoRA model
  └── data/                  # Train/test datasets
frontend/
  └── index.html             # Web interface
```

## Tech Stack

- **Backend**: FastAPI + Uvicorn + SQLAlchemy + SQLite
- **ML**: PyTorch 2.0+ + Transformers 4.57+ + PEFT 0.17+
- **Model**: DistilBERT-base-uncased + LoRA
- **Frontend**: Vanilla JavaScript + HTML5 + CSS3

## Configuration

### Gmail API (Optional)

1. Enable Gmail API in Google Cloud Console
2. Download OAuth 2.0 credentials to `backend/scripts/credentials.json`
3. Modify `backend/api.py`: `ENABLE_GMAIL_API = True`
4. Restart server, first run will open browser for authorization

### Environment Variables

`backend/.env`:
```env
ENABLE_GMAIL_API=false
MODEL_PATH=lora_out_full
MAX_SEQUENCE_LENGTH=256
HOST=0.0.0.0
PORT=8000
```

### Database

Auto-creates `backend/email_security.db` (SQLite)

## Testing

Test data: 60 emails (47 phishing, 8 spam, 5 malware)

Single prediction test:
```bash
cd backend
.\.venv\Scripts\python.exe -c "from model import PhishingDetector; d = PhishingDetector('lora_out_full'); print(d.predict('Click to verify account'))"
```

## Disclaimer

Research prototype. Do not use in production. Model has false positives/negatives, use with other security measures.

---

**CatSpy Team** | v1.0 | 2025-12
