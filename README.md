# CatSpy 🐱 - AI Security Detection Platform

> **AI-powered phishing detection system with 96.41% accuracy**

Local phishing & URL detection system powered by DistilBERT + LoRA fine-tuning, featuring real-time email security monitoring and hybrid ML+heuristic risk assessment.

## 🎯 Features

- **🎣 Phishing Detection**: Advanced AI-powered text and URL analysis using fine-tuned LoRA model
- **📧 Email Security Dashboard**: Comprehensive risk assessment with real-time threat statistics and categorization
- **🧪 Test Data Support**: 60 diverse test emails covering phishing, spam, and malware scenarios
- **📊 Intelligent Risk Scoring**: Hybrid ML + heuristic-based scoring system with gibberish detection
- **🔍 Multi-layer Analysis**: Domain risk, urgency tactics, credential requests, URL obfuscation detection

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (with pip)
- PowerShell (Windows) or Bash (Linux/Mac)
- 4GB+ RAM recommended for model inference

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/KonoNeko/CatSpy.git
cd CatSpy
```

2. **Install dependencies**
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. **Start the backend server**
```powershell
.\start.ps1
```

Or manually:
```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

4. **Access the web interface**

Open your browser and navigate to: **http://localhost:8000**

The frontend (`frontend/index.html`) is automatically served by the backend.

### Quick Test

Once running, click **"Test Data (60)"** in the Email Security tab to load 60 sample emails and see the system in action.

## 📊 Model Performance

### Current Model Accuracy

| Metric | Test Set (2,286 samples) | Full Dataset (11,430 samples) |
|--------|---------------------------|-------------------------------|
| **Accuracy** | **95.28%** | **96.41%** |
| Precision | 96.33% | 97.54% |
| Recall | 94.14% | 95.22% |
| F1-Score | 95.22% | 96.37% |

### Error Analysis

- **False Positives**: 137 safe URLs misclassified as phishing (2.4%)
- **False Negatives**: 273 phishing URLs misclassified as safe (4.8%)
- **Total Errors**: 410 out of 11,430 samples (3.59%)

### Model Architecture

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Total Parameters**: 67,694,596
- **Trainable Parameters**: 739,586 (1.09% of total)
- **Model Size**: ~2.8 MB (LoRA adapter only)

## 🔧 Model Training

### Prepare Dataset

1. **Format training data** (JSONL format):
```json
{"text": "Urgent! Verify your account now: http://phishing-site.com", "label": 1}
{"text": "Meeting scheduled for tomorrow at 3 PM", "label": 0}
```
- `text`: Email content or URL
- `label`: 0 (Safe), 1 (Phishing)

2. **Split into train/test sets** (80/20 split recommended):
```powershell
cd backend
python scripts/prepare_full_dataset.py
```

### Train Model

**Basic training**:
```powershell
cd backend
.\.venv\Scripts\python.exe train_lora.py `
  --train_file data/train_full.jsonl `
  --eval_file data/test_full.jsonl `
  --output_dir lora_out_full `
  --epochs 5 `
  --batch_size 16 `
  --lr 2e-4 `
  --max_length 256
```

**Training parameters**:
- `--train_file`: Path to training data (JSONL)
- `--eval_file`: Path to evaluation/test data (JSONL)
- `--output_dir`: Directory to save trained model
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size for training (default: 16)
- `--lr`: Learning rate (default: 2e-4)
- `--max_length`: Maximum sequence length (default: 256)

**Training output**:
- LoRA adapter weights: `lora_out_full/adapter_model.safetensors`
- Configuration: `lora_out_full/adapter_config.json`
- Training logs with loss and accuracy metrics

### Evaluate Model

**Run evaluation**:
```powershell
cd backend
.\.venv\Scripts\python.exe evaluate_lora_model.py `
  --lora_adapter_path lora_out_full `
  --test_file data/test_full.jsonl `
  --batch_size 32
```

**Evaluation output**:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix
- Classification report
- Detailed error analysis

### Model Training Timeline

- **Dataset**: 11,430 samples (50% Safe, 50% Phishing)
- **Training Time**: ~30-45 minutes on GPU, 2-3 hours on CPU
- **Best Epoch**: Model selection based on F1 score on validation set
- **Hardware Requirements**: 
  - GPU: 4GB+ VRAM (CUDA-enabled)
  - CPU: 8GB+ RAM (slower fallback)

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Authentication
No authentication required (local deployment)

---

### 🎯 Phishing Detection

#### `POST /predict`
Analyze text or URL for phishing indicators.

**Request Body**:
```json
{
  "text": "Click here to verify your PayPal account immediately!"
}
```

**Response** (200 OK):
```json
{
  "result": "PHISHING",
  "score": 78,
  "confidence": 87.3,
  "model": {
    "label": 1,
    "confidence": {
      "safe": 12.7,
      "phishing": 87.3
    }
  },
  "heuristics": {
    "domain_risk": "high",
    "urgency_language": "detected",
    "creds_request": "detected",
    "shortener_obfuscation": "none",
    "brand_spoof": "suspected"
  },
  "risk_indicators": [
    {
      "type": "urgency",
      "text": "immediately",
      "severity": "high"
    }
  ],
  "cues": [
    {
      "type": "keyword",
      "text": "verify, account"
    }
  ]
}
```

**Response Fields**:
- `result`: Risk level (SAFE, SUSPICIOUS, PHISHING)
- `score`: Risk score 0-100
- `confidence`: Model confidence percentage
- `heuristics`: Rule-based detection results
- `risk_indicators`: Detected threats with severity
- `cues`: Identified suspicious elements

---

### 📧 Email Security

#### `POST /api/scan`
Scan emails from Gmail (requires Gmail API setup).

**Request Body**:
```json
{
  "limit": 100
}
```

**Response** (200 OK):
```json
{
  "message": "Scan completed",
  "scanned": 100,
  "inserted": 95,
  "duplicates": 5,
  "high_risk": 12,
  "suspicious": 28
}
```

---

#### `POST /api/load-test-data`
Load 60 test emails for demonstration.

**Request Body**: None

**Response** (200 OK):
```json
{
  "message": "Test data loaded successfully",
  "scanned": 60,
  "high_risk": 6,
  "suspicious": 21
}
```

---

#### `GET /api/risk-summary`
Get email risk statistics.

**Response** (200 OK):
```json
{
  "total_emails": 60,
  "suspicious_emails": 27,
  "high_risk_emails": 6,
  "by_risk_level": {
    "low": 33,
    "medium": 21,
    "high": 6
  },
  "by_category": {
    "phishing": 39,
    "spam": 8,
    "malware": 5,
    "normal": 8
  }
}
```

---

#### `GET /api/suspicious-emails`
Retrieve list of suspicious emails.

**Query Parameters**:
- `limit` (optional): Number of results (default: 50)
- `min_risk_score` (optional): Minimum risk score 0.0-1.0 (default: 0.5)

**Example**:
```
GET /api/suspicious-emails?limit=10&min_risk_score=0.6
```

**Response** (200 OK):
```json
{
  "items": [
    {
      "id": 1,
      "subject": "Urgent: Account Verification Required",
      "sender": "security@paypal-verify.com",
      "date": "2025-12-02T10:30:00",
      "risk_score": 0.85,
      "risk_level": "high",
      "category": "phishing",
      "reasons": [
        "High urgency language detected",
        "Credential request identified",
        "Domain mismatch with brand"
      ]
    }
  ],
  "total": 10
}
```

---

#### `DELETE /api/clear-database`
Clear all email records from database.

**Response** (200 OK):
```json
{
  "message": "Database cleared successfully",
  "deleted": 60
}
```

---

#### `GET /api/status`
Get system status and configuration.

**Response** (200 OK):
```json
{
  "status": "running",
  "model_type": "DistilBERT + LoRA",
  "gmail_connected": false,
  "gmail_account": null,
  "data_source": "test_data"
}
```

---

### 🏥 Health Check

#### `GET /health`
Check if the API is running.

**Response** (200 OK):
```json
{
  "status": "ok"
}
```

---

### Error Responses

All endpoints may return error responses:

**400 Bad Request**:
```json
{
  "detail": "Invalid request parameters"
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Model prediction failed"
}
```

## 📁 Project Structure

```
CatSpy/
├── backend/
│   ├── api.py                    # FastAPI application & REST endpoints
│   ├── model.py                  # DistilBERT + LoRA model wrapper
│   ├── utils.py                  # Risk scoring & heuristic analysis
│   ├── train_lora.py             # Model training script
│   ├── evaluate_lora_model.py    # Model evaluation script
│   ├── test_email_data.py        # Test email generator (60 samples)
│   ├── gmail_api.py              # Gmail API integration (optional)
│   ├── start.ps1                 # Windows startup script
│   ├── start.sh                  # Linux/Mac startup script
│   ├── requirements.txt          # Python dependencies
│   ├── lora_out_full/            # Trained LoRA adapter
│   │   ├── adapter_model.safetensors
│   │   └── adapter_config.json
│   ├── data/                     # Training datasets
│   │   ├── train_full.jsonl      # Training data (9,144 samples)
│   │   ├── test_full.jsonl       # Test data (2,286 samples)
│   │   └── dataset_phishing.csv  # Raw phishing dataset
│   └── scripts/
│       └── prepare_full_dataset.py
├── frontend/
│   ├── index.html                # Single-page web UI
│   └── catspy-logo.png           # Application logo
├── README.md                      # This file
└── TRAINING_REPORT.md            # Detailed training report
```

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI 0.104+
- **Server**: Uvicorn (ASGI)
- **Database**: SQLite 3
- **ORM**: SQLAlchemy

### Machine Learning
- **ML Framework**: PyTorch 2.0+
- **Transformer Library**: Hugging Face Transformers 4.57+
- **Fine-tuning**: PEFT (Parameter-Efficient Fine-Tuning) 0.17+
- **Base Model**: DistilBERT-base-uncased
- **Tokenizer**: DistilBERT tokenizer (max length: 256)

### Frontend
- **Framework**: Vanilla JavaScript (no dependencies)
- **UI**: HTML5 + CSS3 (custom styling)
- **Charts**: Canvas-based visualizations

### Development
- **Language**: Python 3.8+
- **Package Manager**: pip
- **Version Control**: Git

## ⚙️ Configuration

### Environment Variables

Create `.env` file in `backend/` directory:

```env
# Gmail API (Optional)
ENABLE_GMAIL_API=false
GMAIL_CREDENTIALS_PATH=scripts/credentials.json

# Model Configuration
MODEL_PATH=lora_out_full
MAX_SEQUENCE_LENGTH=256

# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true
```

### Gmail API Setup (Optional)

1. Enable Gmail API in Google Cloud Console
2. Download OAuth 2.0 credentials
3. Save as `backend/scripts/credentials.json`
4. Set `ENABLE_GMAIL_API=True` in `backend/api.py`
5. Restart backend server
6. First run will prompt browser OAuth flow

### Database Configuration

SQLite database is created automatically at `backend/email_security.db`.

**Schema**:
```sql
CREATE TABLE email_scans (
    id INTEGER PRIMARY KEY,
    message_id TEXT UNIQUE,
    subject TEXT,
    sender TEXT,
    body TEXT,
    date TEXT,
    risk_score REAL,
    risk_level TEXT,
    category TEXT,
    reasons TEXT,
    scanned_at TEXT
);
```

## 🧪 Testing

### Test Email Categories

The system includes 60 diverse test emails:
- **47 Phishing emails**: PayPal, Amazon, Banking, Netflix, etc.
- **8 Spam emails**: Supplements, credit cards, dating, etc.
- **5 Malware emails**: Fake invoices, security updates, etc.

### Manual Testing

Test individual predictions:
```powershell
cd backend
.\.venv\Scripts\python.exe -c "from model import PhishingDetector; import json; d = PhishingDetector(lora_adapter_path='lora_out_full'); print(json.dumps(d.predict('Click here to verify your account')))"
```

### Scoring Validation

Run scoring tests:
```powershell
cd backend
.\.venv\Scripts\python.exe test_scoring.py
```

## 📖 Training Report

For detailed model training results, evaluation metrics, and error analysis, see:
[TRAINING_REPORT.md](./backend/TRAINING_REPORT.md)

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for research and educational purposes.

## ⚠️ Disclaimer

**Research prototype - Not for production use**

This system is a proof-of-concept and should not be relied upon as the sole security control. Always:
- Use in combination with other security measures
- Verify suspicious emails through official channels
- Keep your security software updated
- Exercise caution with any unsolicited communications

The model may produce false positives and false negatives. Human judgment is essential.

## 📧 Support

For issues and questions:
- GitHub Issues: [https://github.com/KonoNeko/CatSpy/issues](https://github.com/KonoNeko/CatSpy/issues)

---

**Made with 🐱 by the CatSpy Team** | **Model Version**: 1.0 | **Last Updated**: December 2025
