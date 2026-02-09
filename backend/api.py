from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import json
import random

try:
    from .model import PhishingDetector
    from .utils import map_model_to_frontend
    from .deepfake_model import DeepfakeDetector
except ImportError:
    from model import PhishingDetector
    from utils import map_model_to_frontend
    from deepfake_model import DeepfakeDetector

ENABLE_GMAIL_API = True

GMAIL_API_AVAILABLE = False
if ENABLE_GMAIL_API:
    try:
        from gmail_api import get_gmail_client
        GMAIL_API_AVAILABLE = True
        print("Gmail API enabled")
    except Exception as e:
        print(f"Gmail API not available: {e}")
        GMAIL_API_AVAILABLE = False
else:
    print("Gmail API disabled")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
detector = PhishingDetector()

# Initialize deepfake detector
try:
    # Use custom trained model
    deepfake_detector = DeepfakeDetector(model_path='deepfake_models/resnet18_deepfake_custom.pth')
    print("✅ Deepfake detector initialized with custom trained model")
except Exception as e:
    print(f"⚠️ Deepfake detector initialization failed: {e}")
    deepfake_detector = None

DATABASE_URL = "sqlite:///./email_security.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class EmailScan(Base):
    __tablename__ = "emails"
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(String, unique=True, index=True, nullable=False)
    subject = Column(String, nullable=False)
    sender = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    risk_level = Column(String, nullable=False)
    risk_score = Column(Float, nullable=False)
    category = Column(String, nullable=False)
    reasons = Column(Text, nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

gmail_client = None
if GMAIL_API_AVAILABLE and ENABLE_GMAIL_API:
    try:
        gmail_client = get_gmail_client()
        print("Gmail client initialized")
    except Exception as e:
        print(f"Gmail client init failed: {e}")
        gmail_client = None

import os
FRONTEND_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend")))
FRONTEND_INDEX = FRONTEND_DIR / "index.html"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_index():
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))
    return {"detail": "index.html not found"}


@app.get("/index.html", include_in_schema=False)
def serve_index_alias():
    return serve_index()


class PredictRequest(BaseModel):
    text: str = Field(..., example="Congratulations! You've won a prize.")


class PredictResponse(BaseModel):
    label: int
    scores: List[float]
    model: str = Field(default="local")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        model_out = detector.predict(req.text)
        frontend = map_model_to_frontend(model_out, req.text, model_name="distilbert-base-uncased")
        return frontend
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.options("/predict", include_in_schema=False)
def predict_options():
    return Response(status_code=200)


class TrainRequest(BaseModel):
    texts: List[str]
    labels: List[int]


class TrainResponse(BaseModel):
    loss: float


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    try:
        loss = detector.train(req.texts, req.labels)
        return {"loss": float(loss)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ScanEmailRequest(BaseModel):
    limit: int = Field(default=50, ge=1, le=500)

class RiskSummary(BaseModel):
    total_emails: int
    suspicious_emails: int
    high_risk_emails: int
    by_risk_level: Dict[str, int]
    by_category: Dict[str, int]
    trend_by_date: List[Dict]

class SuspiciousEmail(BaseModel):
    subject: str
    sender: str
    date: datetime
    risk_level: str
    risk_score: float
    category: str
    reasons: List[str]

class SuspiciousEmailsResponse(BaseModel):
    items: List[SuspiciousEmail]


def generate_mock_emails(count: int = 10) -> List[Dict]:
    """Generate mock email data for testing"""
    mock_subjects = [
        "URGENT: Verify your account now",
        "Your package is waiting",
        "Meeting notes from yesterday",
        "Re: Project update",
        "Click here to claim your prize!",
        "Invoice #12345",
        "Password reset request",
        "Weekly newsletter",
        "Congratulations! You won!",
        "Security alert for your account"
    ]
    mock_senders = [
        "noreply@secure-bank.com",
        "support@shipping.com",
        "john@company.com",
        "alerts@suspicious-domain.xyz",
        "team@newsletter.com"
    ]
    
    emails = []
    for i in range(min(count, len(mock_subjects))):
        emails.append({
            "message_id": f"mock-{i}-{hash(mock_subjects[i])}",
            "subject": mock_subjects[i],
            "sender": random.choice(mock_senders),
            "date": datetime.now() - timedelta(days=random.randint(0, 30)),
            "body": f"This is a test email body for: {mock_subjects[i]}"
        })
    
    return emails


@app.post("/api/scan")
async def scan_emails(req: ScanEmailRequest, db: Session = Depends(get_db)):
    """Scan emails from Gmail or use mock data"""
    emails_to_scan = []
    
    # Try Gmail API first
    if gmail_client:
        try:
            emails_to_scan = gmail_client.fetch_latest_emails(limit=req.limit)
            print(f"✅ Fetched {len(emails_to_scan)} emails from Gmail")
        except Exception as e:
            print(f"⚠️ Gmail fetch failed: {e}")
    
    # Fallback to mock data
    if not emails_to_scan:
        print("📧 Using mock email data")
        emails_to_scan = generate_mock_emails(req.limit)
    
    # Scan and save to database
    inserted = 0
    for email in emails_to_scan:
        # Check if already scanned
        existing = db.query(EmailScan).filter(
            EmailScan.message_id == email.get('message_id', f"mock-{hash(email['subject'])}")
        ).first()
        if existing:
            continue
        
        # Predict risk
        text = f"{email['subject']} {email.get('body', '')}"
        prediction = detector.predict(text)
        
        # Determine risk level and category
        risk_score = prediction.get('confidence', {}).get('phishing', 0) / 100
        if risk_score >= 0.7:
            risk_level = 'high'
            category = 'phishing'
        elif risk_score >= 0.4:
            risk_level = 'medium'
            category = 'spam'
        else:
            risk_level = 'low'
            category = 'normal'
        
        # Extract reasons
        reasons = []
        for indicator in prediction.get('risk_indicators', []):
            reasons.append(indicator['text'])
        
        # Save to database
        # Convert date string to datetime object if needed
        from dateutil import parser as date_parser
        email_date = date_parser.parse(email['date']) if isinstance(email['date'], str) else email['date']
        
        scan = EmailScan(
            message_id=email.get('message_id', f"mock-{hash(email['subject'])}"),
            subject=email['subject'],
            sender=email['sender'],
            date=email_date,
            risk_level=risk_level,
            risk_score=risk_score,
            category=category,
            reasons=json.dumps(reasons)
        )
        db.add(scan)
        inserted += 1
    
    db.commit()
    return {"scanned": len(emails_to_scan), "inserted": inserted}


@app.get("/api/risk-summary", response_model=RiskSummary)
async def get_risk_summary(db: Session = Depends(get_db)):
    """Get risk statistics summary"""
    total = db.query(EmailScan).count()
    suspicious = db.query(EmailScan).filter(EmailScan.risk_score >= 0.3).count()
    high_risk = db.query(EmailScan).filter(EmailScan.risk_level == 'high').count()
    
    # By risk level
    by_level = {}
    for level in ['low', 'medium', 'high']:
        by_level[level] = db.query(EmailScan).filter(EmailScan.risk_level == level).count()
    
    # By category
    by_cat = {}
    for cat in ['normal', 'spam', 'phishing', 'malware']:
        by_cat[cat] = db.query(EmailScan).filter(EmailScan.category == cat).count()
    
    # Trend by date (last 30 days)
    trend = []
    for i in range(30, -1, -1):
        date = datetime.now() - timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        suspicious_count = db.query(EmailScan).filter(
            func.date(EmailScan.date) == date.date(),
            EmailScan.risk_score >= 0.3
        ).count()
        high_count = db.query(EmailScan).filter(
            func.date(EmailScan.date) == date.date(),
            EmailScan.risk_level == 'high'
        ).count()
        trend.append({"date": date_str, "suspicious": suspicious_count, "high": high_count})
    
    return {
        "total_emails": total,
        "suspicious_emails": suspicious,
        "high_risk_emails": high_risk,
        "by_risk_level": by_level,
        "by_category": by_cat,
        "trend_by_date": trend
    }


@app.get("/api/suspicious-emails", response_model=SuspiciousEmailsResponse)
async def get_suspicious_emails(
    limit: int = 50,
    min_risk_score: float = 0.3,
    db: Session = Depends(get_db)
):
    """Get list of suspicious emails"""
    emails = db.query(EmailScan).filter(
        EmailScan.risk_score >= min_risk_score
    ).order_by(EmailScan.risk_score.desc()).limit(limit).all()
    
    items = []
    for email in emails:
        try:
            cues_list = json.loads(email.reasons) if email.reasons else []
            # Convert cues (list of dicts) to simple reason strings
            reasons = [f"{cue.get('type', 'unknown')}: {cue.get('text', 'N/A')}" for cue in cues_list]
        except:
            reasons = ["Parse error"]
        
        items.append(SuspiciousEmail(
            subject=email.subject,
            sender=email.sender,
            date=email.date,
            risk_level=email.risk_level,
            risk_score=email.risk_score,
            category=email.category,
            reasons=reasons
        ))
    
    return SuspiciousEmailsResponse(items=items)


@app.post("/api/load-test-data")
async def load_test_data(db: Session = Depends(get_db)):
    """Load 50 high-risk test emails into the database"""
    try:
        from test_email_data import generate_test_emails
        from utils import check_heuristics
        
        print("🧪 Generating test emails...")
        test_emails = generate_test_emails()
        print(f"✅ Generated {len(test_emails)} test emails")
        
        # Clear existing test data to avoid UNIQUE constraint errors
        print("🗑️ Clearing old test data...")
        deleted_count = db.query(EmailScan).filter(EmailScan.message_id.like('test-email-%')).delete(synchronize_session=False)
        db.commit()
        print(f"✅ Cleared {deleted_count} old test emails")
        
        scanned_count = 0
        high_risk_count = 0
        suspicious_count = 0
        
        for email_data in test_emails:
            # Analyze with ML model
            text = f"{email_data['subject']} {email_data['body']}"
            prediction = detector.predict(text)
            
            # Get heuristic analysis (returns tuple: heuristics_dict, cues_list)
            heuristics, cues = check_heuristics(text)
            
            # Calculate risk score (same logic as scan_test_data)
            # Model returns {"label": int, "scores": [normal_prob, phishing_prob]}
            phishing_prob = prediction['scores'][1]  # Index 1 is phishing probability
            high_risk_keywords = ['urgent', 'immediately', 'verify', 'suspended', 'limited', 'unusual']
            medium_risk_keywords = ['click', 'confirm', 'update', 'secure']
            spam_keywords = ['buy now', 'limited offer', 'act now', 'special offer', 'discount', 'free', 'order now', 'click here']
            malware_keywords = ['.exe', 'download', 'install', 'attached', 'attachment', 'open file', 'run this']
            
            high_risk_count_kw = sum(1 for kw in high_risk_keywords if kw.lower() in text.lower())
            medium_risk_count_kw = sum(1 for kw in medium_risk_keywords if kw.lower() in text.lower())
            spam_count_kw = sum(1 for kw in spam_keywords if kw.lower() in text.lower())
            malware_count_kw = sum(1 for kw in malware_keywords if kw.lower() in text.lower())
            indicator_count = len(heuristics.get('risk_indicators', []))
            
            risk_score = (
                phishing_prob * 0.4 +
                min(high_risk_count_kw / 3, 1.0) * 0.3 +
                min(indicator_count / 4, 1.0) * 0.2 +
                min(medium_risk_count_kw / 3, 1.0) * 0.1
            )
            
            # Determine category based on email type or content analysis
            email_type = email_data.get('type', None)
            if email_type == 'malware' or malware_count_kw >= 2:
                category = 'malware'
                risk_level = 'high'
                risk_score = max(risk_score, 0.85)
                high_risk_count += 1
            elif email_type == 'spam' or spam_count_kw >= 3:
                category = 'spam'
                if risk_score >= 0.7:
                    risk_level = 'high'
                    high_risk_count += 1
                elif risk_score >= 0.4:
                    risk_level = 'medium'
                    suspicious_count += 1
                else:
                    risk_level = 'low'
            elif email_type == 'phishing' or prediction['label'] == 1:
                category = 'phishing'
                if risk_score >= 0.7:
                    risk_level = 'high'
                    high_risk_count += 1
                elif risk_score >= 0.4:
                    risk_level = 'medium'
                    suspicious_count += 1
                else:
                    risk_level = 'low'
            else:
                category = 'normal'
                risk_level = 'low'
            
            # Save to database
            # Convert datetime object to datetime if needed (test data uses datetime objects)
            email_date = email_data['date']
            if isinstance(email_date, str):
                from dateutil import parser as date_parser
                email_date = date_parser.parse(email_date)
            
            email_scan = EmailScan(
                message_id=email_data['message_id'],
                sender=email_data['sender'],
                subject=email_data['subject'],
                date=email_date,
                risk_level=risk_level,
                risk_score=risk_score,
                category=category,
                reasons=json.dumps(cues)
            )
            db.add(email_scan)
            scanned_count += 1
        
        db.commit()
        
        return {
            "success": True,
            "scanned": scanned_count,
            "high_risk": high_risk_count,
            "suspicious": suspicious_count,
            "message": f"Successfully loaded {scanned_count} test emails"
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error loading test data: {error_details}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to load test data: {str(e)}")


@app.get("/api/status")
async def get_status():
    """Get system status"""
    gmail_info = None
    if gmail_client:
        profile = gmail_client.get_user_profile()
        if profile:
            gmail_info = {
                "email": profile['email'],
                "messages_total": profile['messages_total'],
                "threads_total": profile['threads_total']
            }
    
    return {
        "gmail_connected": gmail_client is not None,
        "gmail_account": gmail_info,
        "model_loaded": detector is not None,
        "data_source": "Gmail API" if gmail_client else "Mock Data",
        "model_type": "DistilBERT + LoRA"
    }


@app.delete("/api/clear-database")
async def clear_database(db: Session = Depends(get_db)):
    """Clear all email scans from database"""
    try:
        count = db.query(EmailScan).count()
        db.query(EmailScan).delete()
        db.commit()
        return {
            "success": True,
            "deleted": count,
            "message": f"Successfully cleared {count} email records"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")


# ==================== DEEPFAKE DETECTION ENDPOINTS ====================

from fastapi import File, UploadFile
import tempfile
import shutil

# File size limits
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB

class DeepfakeImageResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    is_deepfake: bool
    risk_level: str

class DeepfakeVideoResponse(BaseModel):
    video_prediction: str
    is_deepfake: bool
    fake_percentage: float
    average_fake_confidence: float
    risk_level: str
    statistics: Dict
    video_info: Dict
    suspicious_frames: List[Dict]


@app.post("/api/deepfake/detect-image", response_model=DeepfakeImageResponse)
async def detect_deepfake_image(file: UploadFile = File(...)):
    """
    Detect if an uploaded image is a deepfake
    
    - **file**: Image file (JPG, PNG, BMP) - Max size: 10MB
    """
    if deepfake_detector is None:
        raise HTTPException(status_code=503, detail="Deepfake detector not available")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size
    file_content = await file.read()
    if len(file_content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail=f"Image file too large. Maximum size is {MAX_IMAGE_SIZE // 1024 // 1024}MB")
    await file.seek(0)  # Reset file pointer
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Run detection
        result = deepfake_detector.predict_image(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return result
    
    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/api/deepfake/detect-video", response_model=DeepfakeVideoResponse)
async def detect_deepfake_video(
    file: UploadFile = File(...),
    sample_rate: int = 30
):
    """
    Detect if an uploaded video contains deepfakes
    
    - **file**: Video file (MP4, AVI, MOV) - Max size: 100MB
    - **sample_rate**: Analyze every Nth frame (default: 30)
    """
    if deepfake_detector is None:
        raise HTTPException(status_code=503, detail="Deepfake detector not available")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Check file size
    file_content = await file.read()
    if len(file_content) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=400, detail=f"Video file too large. Maximum size is {MAX_VIDEO_SIZE // 1024 // 1024}MB")
    await file.seek(0)  # Reset file pointer
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Run detection (this may take a while for large videos)
        result = deepfake_detector.predict_video(tmp_path, sample_rate=sample_rate)
        
        # Clean up
        os.unlink(tmp_path)
        
        return result
    
    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/api/deepfake/status")
async def deepfake_status():
    """Get deepfake detector status"""
    return {
        "available": deepfake_detector is not None,
        "model": "ResNet-18" if deepfake_detector else None,
        "supports_image": deepfake_detector is not None,
        "supports_video": deepfake_detector is not None,
        "trained": Path("deepfake_models/resnet18_deepfake.pth").exists() if deepfake_detector else False
    }
