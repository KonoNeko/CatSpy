import re
from urllib.parse import urlparse

URGENCY_KEYWORDS = ["urgent", "immediately", "asap", "verify", "verify your", "action required", "limited time", "secure your account"]
CREDENTIAL_KEYWORDS = ["password", "passcode", "verify your account", "login", "credit card", "social security", "ssn", "pin"]
SHORTENER_DOMAINS = {"bit.ly","tinyurl.com","t.co","goo.gl","ow.ly","is.gd","buff.ly"}

def extract_urls(text):
    url_re = re.compile(r'(https?://[^\s]+)')
    return url_re.findall(text)

def domain_from_url(url):
    try:
        p = urlparse(url)
        return p.netloc.lower()
    except:
        return ""

def check_heuristics(text):
    txt = text.lower()
    cues = []
    heuristics = {
        "domain_risk":"ok",
        "urgency_language":"none",
        "creds_request":"none",
        "shortener_obfuscation":"none",
        "brand_spoof":"none"
    }

    urls = extract_urls(text)
    for u in urls:
        dom = domain_from_url(u)
        if dom:
            if dom in SHORTENER_DOMAINS:
                heuristics["shortener_obfuscation"] = "present"
                cues.append({"type":"url","text":u,"details":{"domain":dom}})
            if re.search(r'\d{3,}', dom) or re.match(r'^\d+\.\d+\.\d+\.\d+$', dom):
                heuristics["domain_risk"] = "suspicious"
                cues.append({"type":"url","text":u,"details":{"domain":dom}})

    urgency_hits = [k for k in URGENCY_KEYWORDS if k in txt]
    if urgency_hits:
        heuristics["urgency_language"] = "high" if any(k in txt for k in ["urgent","immediately","action required","asap"]) else "low"
        cues.append({"type":"keyword","text":", ".join(set(urgency_hits)),"details":{}})

    cred_hits = [k for k in CREDENTIAL_KEYWORDS if k in txt]
    if cred_hits:
        heuristics["creds_request"] = "confirmed" if any(k in txt for k in ["password","login","verify your account"]) else "maybe"
        cues.append({"type":"credential_request","text":", ".join(set(cred_hits)),"details":{}})

    BRANDS = ["paypal","amazon","google","microsoft","apple"]
    for b in BRANDS:
        if b in txt:
            heuristics["brand_spoof"] = "possible"
            cues.append({"type":"brand","text":b,"details":{}})
            break

    return heuristics, cues

def map_model_to_frontend(model_result, text, model_name="distilbert-base-uncased"):
    probs = model_result.get("scores", [])
    safe_prob = float(probs[0]) if len(probs) > 0 else 0.5
    phishing_prob = float(probs[1]) if len(probs) > 1 else 0.5
    
    pred_label = int(model_result.get("label", 0))
    heuristics, cues = check_heuristics(text)
    
    urls = extract_urls(text)
    has_url = len(urls) > 0
    has_http = 'http://' in text.lower() or 'https://' in text.lower()
    
    threat_keywords = ['password', 'login', 'account', 'verify', 'bank', 'paypal', 'amazon', 
                      'suspended', 'locked', 'confirm', 'update', 'click', 'urgent', 'security']
    has_threat_keywords = any(word in text.lower() for word in threat_keywords)
    
    has_meaningful_words = len([w for w in text.split() if len(w) > 3 and w.isalpha()]) >= 3
    
    is_gibberish = (not has_url and not has_http and 
                    not has_threat_keywords and
                    (not has_meaningful_words or len(text) < 20))
    
    has_strong_heuristics = (
        heuristics["creds_request"] in ["confirmed", "maybe"] or
        heuristics["urgency_language"] == "high" or
        heuristics["domain_risk"] in ["suspicious", "high"] or
        heuristics["shortener_obfuscation"] == "present"
    )

    if is_gibberish:
        base = min(25, phishing_prob * 30)
        weight_map = {
            "domain_risk": {"ok":0, "suspicious":0, "high":0},
            "urgency_language": {"none":0, "low":0, "high":0},
            "creds_request": {"none":0, "maybe":0, "confirmed":0},
            "shortener_obfuscation": {"none":0, "present":0},
            "brand_spoof": {"none":0, "possible":0, "likely":0}
        }
    elif has_strong_heuristics:
        if has_url or has_http:
            base = 50 + (phishing_prob - 0.3) * 80
        else:
            base = 45 + max(0, (phishing_prob - 0.3) * 60)
        
        weight_map = {
            "domain_risk": {"ok":0, "suspicious":12, "high":18},
            "urgency_language": {"none":0, "low":8, "high":15},
            "creds_request": {"none":0, "maybe":12, "confirmed":18},
            "shortener_obfuscation": {"none":0, "present":12},
            "brand_spoof": {"none":0, "possible":10, "likely":15}
        }
    elif pred_label == 0:
        base = max(0, (phishing_prob - 0.5) * 100)
        
        weight_map = {
            "domain_risk": {"ok":0, "suspicious":5, "high":12},
            "urgency_language": {"none":0, "low":3, "high":8},
            "creds_request": {"none":0, "maybe":5, "confirmed":10},
            "shortener_obfuscation": {"none":0, "present":5},
            "brand_spoof": {"none":0, "possible":3, "likely":8}
        }
    else:
        if not has_url and not has_http and not has_threat_keywords:
            base = min(40, 25 + (phishing_prob - 0.5) * 30)
        else:
            base = 50 + (phishing_prob - 0.5) * 100
        
        if has_url or has_http:
            weight_map = {
                "domain_risk": {"ok":0, "suspicious":10, "high":15},
                "urgency_language": {"none":0, "low":8, "high":12},
                "creds_request": {"none":0, "maybe":10, "confirmed":15},
                "shortener_obfuscation": {"none":0, "present":10},
                "brand_spoof": {"none":0, "possible":8, "likely":12}
            }
        else:
            weight_map = {
                "domain_risk": {"ok":0, "suspicious":3, "high":8},
                "urgency_language": {"none":0, "low":5, "high":10},
                "creds_request": {"none":0, "maybe":8, "confirmed":12},
                "shortener_obfuscation": {"none":0, "present":5},
                "brand_spoof": {"none":0, "possible":5, "likely":10}
            }

    extra = 0
    for k,v in heuristics.items():
        extra += weight_map.get(k, {}).get(v, 0)

    score = int(round(min(max(base + extra, 0), 100)))

    if score >= 70:
        result = "PHISHING"
    elif score >= 35:
        result = "SUSPICIOUS"
    else:
        result = "SAFE"

    summary_parts = []
    risk_indicators = []
    
    if heuristics["creds_request"] != "none":
        summary_parts.append("Credential request detected")
        risk_indicators.append({"type": "credential", "severity": "high", "text": "Requests sensitive information"})
    if heuristics["urgency_language"] != "none":
        summary_parts.append("Urgent language detected")
        risk_indicators.append({"type": "urgency", "severity": "medium", "text": "Uses pressure tactics"})
    if heuristics["domain_risk"] != "ok":
        summary_parts.append("Suspicious domain")
        risk_indicators.append({"type": "domain", "severity": "high", "text": "Untrusted or suspicious domain"})
    if heuristics["shortener_obfuscation"] != "none":
        risk_indicators.append({"type": "obfuscation", "severity": "medium", "text": "Uses URL shortener"})
    if heuristics["brand_spoof"] != "none":
        risk_indicators.append({"type": "spoofing", "severity": "high", "text": "Possible brand impersonation"})
    
    if not summary_parts:
        summary = "No significant threats detected. Content appears safe."
    else:
        summary = "; ".join(summary_parts) + "."
    
    confidence = int(round(max(safe_prob, phishing_prob) * 100))
    
    return {
        "result": result,
        "score": score,
        "confidence": confidence,
        "summary": summary,
        "risk_indicators": risk_indicators,
        "heuristics": heuristics,
        "cues": cues,
        "details": {
            "has_url": has_url,
            "has_http": has_http,
            "is_gibberish": is_gibberish,
            "url_count": len(urls)
        },
        "model": {
            "name": model_name,
            "prediction": "safe" if pred_label == 0 else "phishing",
            "confidence": {
                "safe": round(safe_prob * 100, 2),
                "phishing": round(phishing_prob * 100, 2)
            }
        }
    }
