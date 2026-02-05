import re
from urllib.parse import urlparse
from difflib import SequenceMatcher

URGENCY_KEYWORDS = ["urgent", "immediately", "asap", "verify", "verify your", "action required", "limited time", "secure your account"]
CREDENTIAL_KEYWORDS = ["password", "passcode", "verify your account", "login", "credit card", "social security", "ssn", "pin"]
SHORTENER_DOMAINS = {"bit.ly","tinyurl.com","t.co","goo.gl","ow.ly","is.gd","buff.ly"}

# 品牌及其合法域名
BRAND_DOMAINS = {
    "paypal": ["paypal.com", "paypal.co.uk", "paypal.ca", "paypal.de", "paypal.fr"],
    "amazon": ["amazon.com", "amazon.co.uk", "amazon.ca", "amazon.de", "amazon.fr", "amazon.in"],
    "google": ["google.com", "gmail.com", "youtube.com", "google.co.uk", "google.ca"],
    "microsoft": ["microsoft.com", "outlook.com", "hotmail.com", "live.com", "office.com", "xbox.com"],
    "apple": ["apple.com", "icloud.com", "itunes.com", "me.com"],
    "netflix": ["netflix.com"],
    "facebook": ["facebook.com", "fb.com", "messenger.com"],
    "instagram": ["instagram.com"],
    "linkedin": ["linkedin.com"],
    "twitter": ["twitter.com", "x.com"],
    "whatsapp": ["whatsapp.com"],
    "bank": ["bankofamerica.com", "chase.com", "wellsfargo.com", "citibank.com"]
}

# 白名单：常见合法网站（包括各种域名形式）
WHITELIST_DOMAINS = {
    # 搜索引擎
    "google.com", "bing.com", "yahoo.com", "duckduckgo.com", "baidu.com",
    # 社交媒体
    "facebook.com", "fb.com", "instagram.com", "twitter.com", "x.com", "linkedin.com",
    "reddit.com", "pinterest.com", "tumblr.com", "snapchat.com", "tiktok.com",
    "whatsapp.com", "telegram.org", "discord.com", "slack.com",
    # 电商
    "amazon.com", "ebay.com", "alibaba.com", "aliexpress.com", "etsy.com",
    "walmart.com", "target.com", "bestbuy.com", "costco.com",
    # 科技公司
    "microsoft.com", "apple.com", "google.com", "ibm.com", "oracle.com",
    "adobe.com", "salesforce.com", "zoom.us", "dropbox.com",
    # 邮箱服务
    "gmail.com", "outlook.com", "hotmail.com", "yahoo.com", "protonmail.com",
    "icloud.com", "mail.ru", "aol.com", "live.com",
    # 流媒体
    "netflix.com", "youtube.com", "hulu.com", "disneyplus.com", "spotify.com",
    "twitch.tv", "vimeo.com", "soundcloud.com",
    # 新闻媒体
    "cnn.com", "bbc.com", "nytimes.com", "wsj.com", "theguardian.com",
    "forbes.com", "reuters.com", "bloomberg.com",
    # 银行金融
    "paypal.com", "stripe.com", "square.com", "venmo.com",
    "bankofamerica.com", "chase.com", "wellsfargo.com", "citibank.com",
    # 开发者工具
    "github.com", "gitlab.com", "stackoverflow.com", "npmjs.com",
    "pypi.org", "docker.com", "heroku.com", "vercel.com",
    # 云服务
    "aws.amazon.com", "azure.microsoft.com", "cloud.google.com",
    "digitalocean.com", "cloudflare.com", "linode.com",
    # 学习教育
    "wikipedia.org", "coursera.org", "udemy.com", "khanacademy.org",
    "edx.org", "linkedin.com", "medium.com",
    # 其他常见网站
    "wordpress.com", "godaddy.com", "wix.com", "squarespace.com",
    "shopify.com", "mailchimp.com", "canva.com", "notion.so",
    # 中国常见网站
    "taobao.com", "tmall.com", "jd.com", "weibo.com", "wechat.com",
    "qq.com", "163.com", "126.com", "sina.com", "sohu.com",
    # 各国域名变体
    "amazon.co.uk", "amazon.de", "amazon.fr", "amazon.in", "amazon.jp",
    "google.co.uk", "google.de", "google.fr", "google.ca", "google.com.au",
    "ebay.co.uk", "ebay.de", "ebay.fr", "ebay.ca", "ebay.com.au",
    "paypal.co.uk", "paypal.de", "paypal.fr", "paypal.ca",
    "netflix.co.uk", "netflix.ca", "netflix.com.au",
    "bbc.co.uk", "bbc.com",
}

def similarity_ratio(a, b):
    """计算两个字符串的相似度 (0.0-1.0)"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def is_whitelisted(domain):
    """
    检查域名是否在白名单中
    支持多种格式：
    - 完整匹配: example.com
    - 子域名: api.example.com, www.example.com
    """
    if not domain:
        return False
    
    # 移除www.前缀和端口号，转换为小写
    clean_domain = domain.replace("www.", "").split(":")[0].lower()
    
    # 直接匹配
    if clean_domain in WHITELIST_DOMAINS:
        return True
    
    # 检查是否是白名单域名的子域名
    # 例如：api.github.com 应该匹配 github.com
    for whitelist_domain in WHITELIST_DOMAINS:
        if clean_domain.endswith("." + whitelist_domain) or clean_domain == whitelist_domain:
            return True
    
    # 检查BRAND_DOMAINS中的合法域名
    for brand_domains in BRAND_DOMAINS.values():
        for legit_domain in brand_domains:
            if clean_domain.endswith(legit_domain) or clean_domain == legit_domain:
                return True
    
    return False

def check_brand_spoofing(domain, text_lower):
    """
    检测品牌伪造，包括：
    1. 域名与知名品牌高度相似（拼写错误）
    2. 文本中提到品牌但域名不匹配
    3. 包含品牌名作为子串（例如：mmmmmicroft 包含 microft）
    4. 通过重复字符混淆（例如：gooogle, micccrosoft）
    
    重要：白名单域名不会被标记为伪造
    """
    if not domain:
        return "none", None
    
    # 移除www.前缀和端口号
    clean_domain = domain.replace("www.", "").split(":")[0].lower()
    
    # 🔥 白名单检查 - 如果在白名单中，直接返回安全
    if is_whitelisted(clean_domain):
        return "none", None
    
    for brand, legitimate_domains in BRAND_DOMAINS.items():
        # 检查文本中是否提到该品牌
        brand_mentioned = brand in text_lower
        
        # 检查域名是否是合法域名
        is_legitimate = any(clean_domain.endswith(legit) for legit in legitimate_domains)
        
        if brand_mentioned and not is_legitimate:
            # 品牌在文本中但域名不是官方的 - 可能伪造
            return "suspected", brand
        
        # 检查域名与品牌名的相似度（拼写错误检测）
        # 例如: mmicroft.com vs microsoft
        domain_parts = clean_domain.split('.')
        for part in domain_parts:
            if len(part) >= 4:  # 只检查足够长的部分
                # 🔥 新增：检查是否包含品牌名作为子串
                # 例如：mmmmmicroft 包含类似 microsoft 的部分
                if brand in part and not is_legitimate:
                    return "likely", f"{brand} (embedded in: {part})"
                
                # 🔥 新增：去除重复字符后再检查
                # 例如：gooogle -> gogle, micccrosoft -> microsoft
                dedupe_part = re.sub(r'(.)\1+', r'\1', part)  # 将连续重复字符替换为单个
                if dedupe_part != part:  # 如果有重复字符
                    dedupe_sim = similarity_ratio(dedupe_part, brand)
                    if dedupe_sim >= 0.7:
                        return "likely", f"{brand} (char-stuffing: {part})"
                
                # 🔥 新增：检查品牌名是否作为子串存在（模糊匹配）
                # 例如：mmmmmicroft 与 microsoft 的最长公共子序列
                for i in range(len(part) - len(brand) + 1):
                    substring = part[i:i+len(brand)]
                    if similarity_ratio(substring, brand) >= 0.75:
                        return "likely", f"{brand} (substring match: {part})"
                
                # 原有的相似度检测
                sim = similarity_ratio(part, brand)
                # 相似度在70%-95%之间 - 很可能是拼写错误的品牌伪造
                if 0.7 <= sim < 0.95:
                    return "likely", f"{brand} (typo: {part})"
                # 完全匹配但不在合法域名列表 - 确认伪造
                elif sim >= 0.95 and not is_legitimate:
                    return "confirmed", brand
    
    return "none", None

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
    spoofed_brand = None
    
    for u in urls:
        dom = domain_from_url(u)
        if dom:
            # 检查品牌伪造
            spoof_level, brand_info = check_brand_spoofing(dom, txt)
            if spoof_level != "none":
                heuristics["brand_spoof"] = spoof_level
                spoofed_brand = brand_info
                cues.append({"type":"brand_spoof","text":f"{dom} (mimics {brand_info})","details":{"domain":dom, "brand":brand_info}})
            
            # 检查短链接
            if dom in SHORTENER_DOMAINS:
                heuristics["shortener_obfuscation"] = "present"
                cues.append({"type":"url","text":u,"details":{"domain":dom}})
            
            # 检查可疑域名（IP地址或包含大量数字）
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

    # 检查文本中提到的品牌（如果域名没有检测到品牌伪造）
    if heuristics["brand_spoof"] == "none":
        for brand in BRAND_DOMAINS.keys():
            if brand in txt:
                heuristics["brand_spoof"] = "possible"
                cues.append({"type":"brand","text":brand,"details":{}})
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
