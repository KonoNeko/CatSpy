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
    phishing_prob = float(probs[1]) if len(probs) > 1 else float(probs[0] if probs else 0.0)
    base = phishing_prob * 100.0

    heuristics, cues = check_heuristics(text)

    weight_map = {
        "domain_risk": {"ok":0, "suspicious":8, "high":20},
        "urgency_language": {"none":0, "low":6, "high":12},
        "creds_request": {"none":0, "maybe":10, "confirmed":25},
        "shortener_obfuscation": {"none":0, "present":10},
        "brand_spoof": {"none":0, "possible":8, "likely":16}
    }

    extra = 0
    for k,v in heuristics.items():
        extra += weight_map.get(k, {}).get(v, 0)

    score = int(round(min(max(base + extra, 0), 100)))

    if score >= 65:
        result = "PHISHING"
    elif score >= 30:
        result = "SUSPICIOUS"
    else:
        result = "SAFE"

    summary_parts = []
    if heuristics["creds_request"] != "none":
        summary_parts.append("Credential request detected")
    if heuristics["urgency_language"] != "none":
        summary_parts.append("Urgent language detected")
    if heuristics["domain_risk"] != "ok":
        summary_parts.append("Suspicious domain")
    if not summary_parts:
        summary = "No risky cues found."
    else:
        summary = "; ".join(summary_parts) + "."

    return {
        "result": result,
        "score": score,
        "summary": summary,
        "heuristics": heuristics,
        "cues": cues,
        "model": {"name": model_name, "label": int(model_result.get("label",0)), "scores": model_result.get("scores",[])}
    }
