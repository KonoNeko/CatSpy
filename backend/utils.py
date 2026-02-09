import re
import unicodedata
from urllib.parse import urlparse
from difflib import SequenceMatcher

URGENCY_KEYWORDS = ["urgent", "immediately", "asap", "verify", "verify your", "action required", "limited time", "secure your account"]
CREDENTIAL_KEYWORDS = ["password", "passcode", "verify your account", "login", "credit card", "social security", "ssn", "pin"]
SHORTENER_DOMAINS = {"bit.ly","tinyurl.com","t.co","goo.gl","ow.ly","is.gd","buff.ly"}

# Major brands and their legitimate domains
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

WHITELIST_DOMAINS = {
    # Search engines
    "google.com", "bing.com", "yahoo.com", "duckduckgo.com", "baidu.com",
    # Social media
    "facebook.com", "fb.com", "instagram.com", "twitter.com", "x.com", "linkedin.com",
    "reddit.com", "pinterest.com", "tumblr.com", "snapchat.com", "tiktok.com",
    "whatsapp.com", "telegram.org", "discord.com", "slack.com",
    # E-commerce
    "amazon.com", "ebay.com", "alibaba.com", "aliexpress.com", "etsy.com",
    "walmart.com", "target.com", "bestbuy.com", "costco.com",
    # Tech companies
    "microsoft.com", "apple.com", "google.com", "ibm.com", "oracle.com",
    "adobe.com", "salesforce.com", "zoom.us", "dropbox.com",
    # Email services
    "gmail.com", "outlook.com", "hotmail.com", "yahoo.com", "protonmail.com",
    "icloud.com", "mail.ru", "aol.com", "live.com",
    # Streaming
    "netflix.com", "youtube.com", "hulu.com", "disneyplus.com", "spotify.com",
    "twitch.tv", "vimeo.com", "soundcloud.com",
    # News media
    "cnn.com", "bbc.com", "nytimes.com", "wsj.com", "theguardian.com",
    "forbes.com", "reuters.com", "bloomberg.com",
    # Banking & finance
    "paypal.com", "stripe.com", "square.com", "venmo.com",
    "bankofamerica.com", "chase.com", "wellsfargo.com", "citibank.com",
    # Developer tools
    "github.com", "gitlab.com", "stackoverflow.com", "npmjs.com",
    "pypi.org", "docker.com", "heroku.com", "vercel.com",
    # Cloud services
    "aws.amazon.com", "azure.microsoft.com", "cloud.google.com",
    "digitalocean.com", "cloudflare.com", "linode.com",
    # Education
    "wikipedia.org", "coursera.org", "udemy.com", "khanacademy.org",
    "edx.org", "linkedin.com", "medium.com",
    # Other popular sites
    "wordpress.com", "godaddy.com", "wix.com", "squarespace.com",
    "shopify.com", "mailchimp.com", "canva.com", "notion.so",
    # Chinese sites
    "taobao.com", "tmall.com", "jd.com", "weibo.com", "wechat.com",
    "qq.com", "163.com", "126.com", "sina.com", "sohu.com",
    # International domain variants
    "amazon.co.uk", "amazon.de", "amazon.fr", "amazon.in", "amazon.jp",
    "google.co.uk", "google.de", "google.fr", "google.ca", "google.com.au",
    "ebay.co.uk", "ebay.de", "ebay.fr", "ebay.ca", "ebay.com.au",
    "paypal.co.uk", "paypal.de", "paypal.fr", "paypal.ca",
    "netflix.co.uk", "netflix.ca", "netflix.com.au",
    "bbc.co.uk", "bbc.com",
}

def similarity_ratio(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def is_whitelisted(domain):

    if not domain:
        return False
    
    # Remove www prefix, port, and lowercase
    clean_domain = domain.replace("www.", "").split(":")[0].lower()
    
    # Direct match
    if clean_domain in WHITELIST_DOMAINS:
        return True
    
    # Check if it's a subdomain of a whitelisted domain
    # e.g., api.github.com should match github.com
    for whitelist_domain in WHITELIST_DOMAINS:
        if clean_domain.endswith("." + whitelist_domain) or clean_domain == whitelist_domain:
            return True
    
    # Check legitimate domains in BRAND_DOMAINS
    for brand_domains in BRAND_DOMAINS.values():
        for legit_domain in brand_domains:
            # Only match exact domain or proper subdomains (with dot separator)
            # This prevents mmicrosoft.com from matching microsoft.com
            if clean_domain == legit_domain or clean_domain.endswith("." + legit_domain):
                return True
    
    return False

def check_brand_spoofing(domain, text_lower):
    """
    Detect brand spoofing using multiple techniques:
    1. Domain highly similar to known brands (typos)
    2. Brand mentioned in text but domain doesn't match
    3. Brand name as substring (e.g., mmmmmicroft contains microft)
    4. Character repetition obfuscation (e.g., gooogle, micccrosoft)
    
    Important: Whitelisted domains are never flagged as spoofed
    """
    if not domain:
        return "none", None
    
    # Remove www prefix and port
    clean_domain = domain.replace("www.", "").split(":")[0].lower()
    
    # Whitelist check - return safe if whitelisted
    if is_whitelisted(clean_domain):
        return "none", None
    
    for brand, legitimate_domains in BRAND_DOMAINS.items():
        # Check if brand is mentioned in text
        brand_mentioned = brand in text_lower
        
        # Check if domain is legitimate
        is_legitimate = any(clean_domain.endswith(legit) for legit in legitimate_domains)
        
        if brand_mentioned and not is_legitimate:
            # Brand in text but domain isn't official - possible spoof
            return "suspected", brand
        
        # Check domain similarity to brand name (typo detection)
        # e.g., mmicroft.com vs microsoft
        domain_parts = clean_domain.split('.')
        for part in domain_parts:
            if len(part) >= 4:  # Only check sufficiently long parts
                # Check if brand name is embedded as substring
                # e.g., mmmmmicroft contains microsoft-like part
                if brand in part and not is_legitimate:
                    return "likely", f"{brand} (embedded in: {part})"
                
                # Check after removing repeated characters
                # e.g., gooogle -> gogle, micccrosoft -> microsoft
                dedupe_part = re.sub(r'(.)\1+', r'\1', part)  # Replace consecutive repeats with single char
                if dedupe_part != part:  # If there are repeated chars
                    dedupe_sim = similarity_ratio(dedupe_part, brand)
                    if dedupe_sim >= 0.7:
                        return "likely", f"{brand} (char-stuffing: {part})"
                
                # Check if brand name exists as fuzzy substring
                # e.g., mmmmmicroft vs microsoft longest common subsequence
                for i in range(len(part) - len(brand) + 1):
                    substring = part[i:i+len(brand)]
                    if similarity_ratio(substring, brand) >= 0.75:
                        return "likely", f"{brand} (substring match: {part})"
                
                # Original similarity detection
                sim = similarity_ratio(part, brand)
                # Similarity 70%-95% - likely typo-based brand spoof
                if 0.7 <= sim < 0.95:
                    return "likely", f"{brand} (typo: {part})"
                # Perfect match but not in legitimate domain list - confirmed spoof
                elif sim >= 0.95 and not is_legitimate:
                    return "confirmed", brand
    
    return "none", None

def extract_urls(text):
    """Extract URLs and domain-like patterns from text.
    Matches:
    - Full URLs: https://example.com, http://example.com
    - Domain patterns: www.example.com, example.com
    """
    urls = []
    
    # Match full URLs with protocol
    url_re = re.compile(r'https?://[^\s]+')
    urls.extend(url_re.findall(text))
    
    # Match domain patterns (www.domain.com or domain.com)
    # Look for patterns like: www.something.com or something.something.com
    domain_re = re.compile(r'\b(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*\.)+[a-zA-Z]{2,}\b')
    domain_matches = domain_re.findall(text)
    
    # Add domains that aren't already captured as full URLs
    for match in domain_matches:
        # Reconstruct the full match (domain_re captures groups, we need the full string)
        full_match = re.search(r'\b((?:www\.)?' + re.escape(match) + r'[a-zA-Z]{2,})', text)
        if full_match:
            domain = full_match.group(0)
            # Only add if not already in urls as part of a full URL
            if not any(domain in url for url in urls):
                urls.append(domain)
    
    return urls

def domain_from_url(url):
    """Extract domain from URL or domain string.
    Handles:
    - Full URLs: https://www.example.com/path -> www.example.com
    - Domain only: www.example.com -> www.example.com
    """
    try:
        # If URL has protocol, use urlparse
        if url.startswith(('http://', 'https://')):
            p = urlparse(url)
            return p.netloc.lower()
        else:
            # Treat as domain directly, remove any path
            domain = url.split('/')[0].split('?')[0]
            return domain.lower()
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
            # Check brand spoofing
            spoof_level, brand_info = check_brand_spoofing(dom, txt)
            if spoof_level != "none":
                heuristics["brand_spoof"] = spoof_level
                spoofed_brand = brand_info
                cues.append({"type":"brand_spoof","text":f"{dom} (mimics {brand_info})","details":{"domain":dom, "brand":brand_info}})
            
            # Check URL shorteners
            if dom in SHORTENER_DOMAINS:
                heuristics["shortener_obfuscation"] = "present"
                cues.append({"type":"url","text":u,"details":{"domain":dom}})
            
            # Check suspicious domains (IP address or many digits)
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

    # Check brands mentioned in text (if domain check found no spoofing)
    # Only flag as possible spoof if brand mentioned but NO legitimate domain found in text
    if heuristics["brand_spoof"] == "none":
        has_legitimate_domain = False
        # First check if any URL in text contains legitimate domain
        for u in urls:
            dom = domain_from_url(u)
            if dom and is_whitelisted(dom):
                has_legitimate_domain = True
                break
        
        # Only flag brand mention if no legitimate domain present
        if not has_legitimate_domain:
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
        heuristics["shortener_obfuscation"] == "present" or
        heuristics["brand_spoof"] in ["likely", "confirmed", "suspected"]
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
            "brand_spoof": {"none":0, "possible":15, "suspected":25, "likely":45, "confirmed":50}
        }
    elif pred_label == 0:
        base = max(0, (phishing_prob - 0.5) * 100)
        
        weight_map = {
            "domain_risk": {"ok":0, "suspicious":5, "high":12},
            "urgency_language": {"none":0, "low":3, "high":8},
            "creds_request": {"none":0, "maybe":5, "confirmed":10},
            "shortener_obfuscation": {"none":0, "present":5},
            "brand_spoof": {"none":0, "possible":15, "suspected":30, "likely":70, "confirmed":80}
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
                "brand_spoof": {"none":0, "possible":8, "suspected":18, "likely":30, "confirmed":35}
            }
        else:
            weight_map = {
                "domain_risk": {"ok":0, "suspicious":3, "high":8},
                "urgency_language": {"none":0, "low":5, "high":10},
                "creds_request": {"none":0, "maybe":8, "confirmed":12},
                "shortener_obfuscation": {"none":0, "present":5},
                "brand_spoof": {"none":0, "possible":5, "suspected":15, "likely":25, "confirmed":30}
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
        severity = "critical" if heuristics["brand_spoof"] in ["likely", "confirmed"] else "high"
        spoof_text = "Brand impersonation detected" if heuristics["brand_spoof"] in ["likely", "confirmed"] else "Possible brand impersonation"
        risk_indicators.append({"type": "spoofing", "severity": severity, "text": spoof_text})
    
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


# ============================================================================
# ADVANCED BRAND SPOOFING DETECTION SYSTEM
# ============================================================================

# Top-level domains list for eTLD+1 extraction
COMMON_TLDS = [
    '.com', '.org', '.net', '.edu', '.gov', '.mil', '.int',
    '.co.uk', '.co.jp', '.co.in', '.co.za', '.com.au', '.com.br',
    '.ac.uk', '.gov.uk', '.org.uk',
    '.de', '.fr', '.it', '.es', '.nl', '.be', '.ch', '.at',
    '.ca', '.mx', '.ar', '.cl', '.co', '.pe',
    '.cn', '.jp', '.kr', '.in', '.sg', '.my', '.th', '.vn',
    '.au', '.nz',
    '.ru', '.ua', '.pl', '.cz', '.sk',
    '.io', '.ai', '.app', '.dev', '.tech', '.online', '.site'
]

# Legitimate brand domains (comprehensive list)
LEGITIMATE_BRANDS = {
    'microsoft': ['microsoft.com', 'outlook.com', 'hotmail.com', 'live.com', 'office.com', 'azure.com', 'xbox.com', 'msn.com'],
    'google': ['google.com', 'gmail.com', 'youtube.com', 'google.co.uk', 'google.ca', 'google.com.au', 'goo.gl'],
    'amazon': ['amazon.com', 'amazon.co.uk', 'amazon.ca', 'amazon.de', 'amazon.fr', 'amazon.in', 'amazon.jp', 'aws.amazon.com'],
    'apple': ['apple.com', 'icloud.com', 'itunes.com', 'me.com', 'mac.com'],
    'facebook': ['facebook.com', 'fb.com', 'messenger.com', 'fbcdn.net'],
    'paypal': ['paypal.com', 'paypal.co.uk', 'paypal.ca', 'paypal.de', 'paypal.fr'],
    'netflix': ['netflix.com'],
    'linkedin': ['linkedin.com'],
    'twitter': ['twitter.com', 'x.com', 't.co'],
    'instagram': ['instagram.com'],
    'whatsapp': ['whatsapp.com'],
    'adobe': ['adobe.com'],
    'dropbox': ['dropbox.com'],
    'zoom': ['zoom.us'],
    'spotify': ['spotify.com'],
}

# Security-related keywords that attackers combine with brand names
SECURITY_KEYWORDS = [
    'login', 'signin', 'sign-in', 'account', 'verify', 'verification',
    'secure', 'security', 'update', 'confirm', 'authenticate', 'auth',
    'password', 'reset', 'recovery', 'support', 'help', 'service',
    'notification', 'alert', 'warning', 'suspended', 'locked'
]

# Common homoglyphs (similar-looking characters)
HOMOGLYPHS = {
    'a': ['а', 'ạ', 'ă', 'ā', 'ą'],  # Latin a vs Cyrillic а
    'e': ['е', 'ė', 'ę', 'ē'],       # Latin e vs Cyrillic е
    'i': ['і', 'ï', 'ī', 'į'],       # Latin i vs Cyrillic і
    'o': ['о', 'ο', 'ọ', 'ō', '0'],  # Latin o vs Cyrillic о, Greek ο, digit 0
    'p': ['р', 'ρ'],                 # Latin p vs Cyrillic р, Greek ρ
    'c': ['с', 'ϲ'],                 # Latin c vs Cyrillic с, Greek ϲ
    's': ['ѕ'],                      # Latin s vs Cyrillic ѕ
    'y': ['у', 'ү'],                 # Latin y vs Cyrillic у
    'x': ['х', 'ⅹ'],                 # Latin x vs Cyrillic х
    'h': ['һ'],                      # Latin h vs Cyrillic һ
    'n': ['ո'],                      # Latin n vs Armenian ո
    'l': ['ӏ', '1', 'і'],            # Latin l vs digit 1, Cyrillic і
    'd': ['ԁ'],                      # Latin d vs Cyrillic ԁ
    'g': ['ց'],                      # Latin g vs Armenian ց
    'j': ['ј'],                      # Latin j vs Cyrillic ј
}


def levenshtein_distance(s1, s2):
    """
    Calculate Levenshtein (edit) distance between two strings.
    Returns the minimum number of single-character edits needed.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def jaro_winkler_similarity(s1, s2, scaling=0.1):
    """
    Calculate Jaro-Winkler similarity (0.0 to 1.0).
    Higher score means more similar.
    """
    if s1 == s2:
        return 1.0
    
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Jaro distance calculation
    match_distance = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0
    
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
    
    # Winkler modification
    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    
    return jaro + prefix * scaling * (1 - jaro)


def extract_registrable_domain(url_or_domain):
    """
    Extract the registrable domain (eTLD+1) from a URL or domain.
    
    Examples:
        microsoft.com -> microsoft.com
        login.microsoft.com -> microsoft.com
        http://login.microsoft.com.phishing.net -> phishing.net
        sub.domain.co.uk -> domain.co.uk
        www.example.com -> example.com
    """
    # Parse URL if it's a full URL
    if url_or_domain.startswith('http://') or url_or_domain.startswith('https://'):
        parsed = urlparse(url_or_domain)
        domain = parsed.netloc
    else:
        domain = url_or_domain
    
    # Remove port
    domain = domain.split(':')[0].lower().strip()
    
    # Remove www prefix
    if domain.startswith('www.'):
        domain = domain[4:]
    
    if not domain:
        return None
    
    # Split into parts
    parts = domain.split('.')
    if len(parts) < 2:
        return None
    
    # Check for multi-part TLDs (e.g., .co.uk, .com.au)
    # These require checking last 2-3 parts
    for i in range(len(parts) - 1, 0, -1):
        potential_tld = '.' + '.'.join(parts[i:])
        if potential_tld in COMMON_TLDS:
            # Found a matching TLD
            tld_parts = len(parts) - i
            if len(parts) > tld_parts:
                # Return domain + TLD (the registrable domain)
                # For example.co.uk, this returns domain.co.uk
                return '.'.join(parts[-(tld_parts + 1):])
            else:
                # The entire string is just the TLD (shouldn't happen)
                return domain
    
    # No multi-part TLD found, assume standard .com/.org/etc.
    # Return last two parts (domain.tld)
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    
    return domain


def detect_homoglyphs(domain):
    """
    Detect non-ASCII characters and potential homoglyph attacks.
    Returns (has_non_ascii, suspicious_chars, ascii_equivalent)
    """
    has_non_ascii = False
    suspicious_chars = []
    ascii_equivalent = []
    
    for char in domain:
        # Check if character is ASCII
        if ord(char) > 127:
            has_non_ascii = True
            suspicious_chars.append(char)
            
            # Try to find ASCII equivalent
            found_match = False
            for ascii_char, homoglyph_list in HOMOGLYPHS.items():
                if char in homoglyph_list:
                    ascii_equivalent.append(ascii_char)
                    found_match = True
                    break
            
            if not found_match:
                # Try Unicode normalization
                normalized = unicodedata.normalize('NFKD', char)
                ascii_approx = ''.join([c for c in normalized if ord(c) < 128])
                ascii_equivalent.append(ascii_approx if ascii_approx else '?')
        else:
            ascii_equivalent.append(char)
    
    return has_non_ascii, suspicious_chars, ''.join(ascii_equivalent)


def detect_typosquatting(domain, brand):
    """
    Detect typosquatting techniques:
    - Character repetition (microsoft -> miccrosoft)
    - Missing characters (microsoft -> microft)
    - Character swaps (microsoft -> microsotf)
    - Digit substitution (google -> g00gle, microsoft -> micr0s0ft)
    - Adjacent key typos
    """
    domain_part = extract_registrable_domain(domain)
    if not domain_part:
        return None
    
    # Remove TLD to get the main part
    main_part = domain_part.split('.')[0]
    
    issues = []
    
    # Check for digit substitutions (leet speak)
    digit_map = {'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b'}
    normalized = main_part
    for digit, letter in digit_map.items():
        normalized = normalized.replace(digit, letter)
    
    if normalized != main_part and normalized == brand:
        substitutions = [f"'{d}' → '{digit_map[d]}'" for d in main_part if d in digit_map]
        issues.append(f"Digit substitution detected: {', '.join(substitutions)}")
    
    # Check for repeated characters
    dedupe = re.sub(r'(.)\1+', r'\1', main_part)
    if dedupe != main_part and dedupe == brand:
        repeated_chars = [m.group(0) for m in re.finditer(r'(.)\1+', main_part)]
        issues.append(f"Repeated characters: {', '.join(repeated_chars)}")
    
    # Calculate edit distance
    edit_dist = levenshtein_distance(main_part, brand)
    max_len = max(len(main_part), len(brand))
    similarity = 1 - (edit_dist / max_len)
    
    # Also check normalized version similarity
    normalized_edit_dist = levenshtein_distance(normalized, brand)
    normalized_similarity = 1 - (normalized_edit_dist / max_len)
    
    if 0.7 <= similarity < 1.0 or 0.7 <= normalized_similarity < 1.0:
        best_similarity = max(similarity, normalized_similarity)
        issues.append(f"Edit distance: {min(edit_dist, normalized_edit_dist)} (similarity: {best_similarity:.2%})")
    
    # Check Jaro-Winkler similarity
    jw_sim = jaro_winkler_similarity(main_part, brand)
    jw_sim_normalized = jaro_winkler_similarity(normalized, brand)
    best_jw = max(jw_sim, jw_sim_normalized)
    
    if 0.8 <= best_jw < 1.0:
        issues.append(f"Jaro-Winkler similarity: {best_jw:.2%}")
    
    # Check if brand is substring
    if brand in main_part and main_part != brand:
        issues.append(f"Contains brand as substring: '{brand}' in '{main_part}'")
    
    return {
        'main_part': main_part,
        'edit_distance': min(edit_dist, normalized_edit_dist),
        'similarity': max(similarity, normalized_similarity),
        'jaro_winkler': best_jw,
        'issues': issues
    } if issues else None


def detect_subdomain_trick(full_domain, brand, legitimate_domains):
    """
    Detect subdomain-based brand impersonation.
    Example: microsoft.com.phishing.net (brand in subdomain, but registrable domain is different)
    
    Important: Legitimate subdomains (e.g., login.microsoft.com) should NOT be flagged.
    """
    registrable = extract_registrable_domain(full_domain)
    
    # Check if registrable domain is legitimate
    is_legitimate = registrable in legitimate_domains
    
    # If the registrable domain itself is legitimate, it's safe (even with subdomains)
    if is_legitimate:
        return None
    
    # Check if brand appears in the full domain
    if brand in full_domain or any(legit.split('.')[0] in full_domain for legit in legitimate_domains):
        # Brand appears in domain but registrable domain is not legitimate
        return {
            'registrable_domain': registrable,
            'full_domain': full_domain,
            'issue': f"Brand '{brand}' appears as subdomain of untrusted domain '{registrable}'"
        }
    
    return None


def detect_brand_keyword_abuse(domain):
    """
    Detect combinations of brand-like names with security keywords.
    Example: microsoft-login-secure.com, paypal-verify.com
    """
    domain_lower = domain.lower()
    detected = []
    
    for brand in LEGITIMATE_BRANDS.keys():
        if brand in domain_lower:
            for keyword in SECURITY_KEYWORDS:
                if keyword in domain_lower:
                    detected.append({
                        'brand': brand,
                        'keyword': keyword,
                        'pattern': f"{brand}...{keyword}"
                    })
    
    return detected if detected else None


def analyze_brand_spoofing(domain_or_url):
    """
    Comprehensive brand spoofing analysis.
    
    Args:
        domain_or_url: Domain name or full URL to analyze
    
    Returns:
        Dictionary with structured analysis results including:
        - detected_brand: Brand that may be targeted (or None)
        - risk_level: 'Legitimate', 'Suspicious', or 'High Risk'
        - detection_reasons: List of specific issues found
        - explanation: Human-readable summary for ESL users
        - technical_details: Detailed technical analysis
    """
    # Extract registrable domain
    registrable_domain = extract_registrable_domain(domain_or_url)
    
    if not registrable_domain:
        return {
            'detected_brand': None,
            'risk_level': 'Unknown',
            'detection_reasons': ['Invalid or unparseable domain'],
            'explanation': 'The provided domain could not be analyzed.',
            'technical_details': {}
        }
    
    # Extract full domain (including subdomains)
    if domain_or_url.startswith('http://') or domain_or_url.startswith('https://'):
        parsed = urlparse(domain_or_url)
        full_domain = parsed.netloc.split(':')[0].lower()
    else:
        full_domain = domain_or_url.split(':')[0].lower()
    
    # Remove www
    if full_domain.startswith('www.'):
        full_domain = full_domain[4:]
    
    detection_reasons = []
    detected_brand = None
    risk_level = 'Legitimate'
    technical_details = {}
    
    # 1. Check if domain is legitimate
    for brand, legit_domains in LEGITIMATE_BRANDS.items():
        if registrable_domain in legit_domains:
            return {
                'detected_brand': brand,
                'risk_level': 'Legitimate',
                'detection_reasons': [f'Domain is official {brand.title()} domain'],
                'explanation': f'This is a legitimate {brand.title()} domain. Safe to use.',
                'technical_details': {
                    'registrable_domain': registrable_domain,
                    'brand': brand,
                    'verified_legitimate': True
                }
            }
    
    # 2. Check for homoglyphs / IDN attacks
    has_non_ascii, suspicious_chars, ascii_equiv = detect_homoglyphs(registrable_domain)
    if has_non_ascii:
        risk_level = 'High Risk'
        detection_reasons.append(f'Homoglyph attack detected: Non-ASCII characters {suspicious_chars}')
        detection_reasons.append(f'ASCII equivalent would be: {ascii_equiv}')
        technical_details['homoglyph_attack'] = {
            'original': registrable_domain,
            'ascii_equivalent': ascii_equiv,
            'suspicious_chars': suspicious_chars
        }
        
        # Check if ASCII equivalent matches a known brand
        for brand, legit_domains in LEGITIMATE_BRANDS.items():
            for legit in legit_domains:
                if ascii_equiv == legit or ascii_equiv.startswith(brand):
                    detected_brand = brand
                    detection_reasons.append(f'Impersonates {brand.title()} using look-alike characters')
                    break
    
    # 3. Check for typosquatting against all known brands
    for brand, legit_domains in LEGITIMATE_BRANDS.items():
        typo_result = detect_typosquatting(registrable_domain, brand)
        if typo_result and typo_result['issues']:
            detected_brand = brand
            risk_level = 'High Risk' if typo_result['similarity'] >= 0.85 else 'Suspicious'
            detection_reasons.append(f'Typosquatting detected (targeting {brand.title()})')
            detection_reasons.extend(typo_result['issues'])
            technical_details['typosquatting'] = typo_result
            break
    
    # 4. Check for subdomain tricks
    for brand, legit_domains in LEGITIMATE_BRANDS.items():
        subdomain_trick = detect_subdomain_trick(full_domain, brand, legit_domains)
        if subdomain_trick:
            detected_brand = brand
            risk_level = 'High Risk'
            detection_reasons.append('Subdomain-based impersonation')
            detection_reasons.append(subdomain_trick['issue'])
            technical_details['subdomain_trick'] = subdomain_trick
            break
    
    # 5. Check for brand + security keyword abuse
    keyword_abuse = detect_brand_keyword_abuse(registrable_domain)
    if keyword_abuse:
        for abuse in keyword_abuse:
            detected_brand = abuse['brand']
            risk_level = 'Suspicious' if risk_level == 'Legitimate' else risk_level
            detection_reasons.append(f"Suspicious pattern: {abuse['pattern']}")
            detection_reasons.append(f"Combines brand '{abuse['brand']}' with security keyword '{abuse['keyword']}'")
        technical_details['keyword_abuse'] = keyword_abuse
    
    # Generate explanation
    if risk_level == 'High Risk':
        explanation = f"⚠️ HIGH RISK: This domain appears to impersonate {detected_brand.title() if detected_brand else 'a legitimate brand'}. "
        explanation += "It uses deceptive techniques to look like a trusted website. DO NOT enter passwords or personal information."
    elif risk_level == 'Suspicious':
        explanation = f"⚠️ SUSPICIOUS: This domain shows signs of impersonating {detected_brand.title() if detected_brand else 'a known brand'}. "
        explanation += "Exercise caution and verify the URL carefully before proceeding."
    else:
        explanation = "No brand spoofing detected. The domain does not appear to impersonate known brands."
    
    return {
        'detected_brand': detected_brand,
        'risk_level': risk_level,
        'detection_reasons': detection_reasons if detection_reasons else ['No suspicious patterns detected'],
        'explanation': explanation,
        'technical_details': {
            'registrable_domain': registrable_domain,
            'full_domain': full_domain,
            **technical_details
        }
    }


# Example usage and test cases
if __name__ == "__main__":
    test_domains = [
        "microsoft.com",
        "mmicrosoft.com",
        "mmicrosoft.comm",
        "microsoft-login-secure.com",
        "google.com",
        "g00gle.com",
        "login.microsoft.com",
        "microsoft.com.phishing.net",
        "gοοgle.com",  # with Greek omicrons
        "аpple.com",    # with Cyrillic 'а'
    ]
    
    print("=== BRAND SPOOFING DETECTION ANALYSIS ===\n")
    
    for domain in test_domains:
        print(f"\n{'='*70}")
        print(f"Analyzing: {domain}")
        print('='*70)
        
        result = analyze_brand_spoofing(domain)
        
        print(f"\nDetected Brand: {result['detected_brand'] or 'None'}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"\nExplanation:")
        print(f"   {result['explanation']}")
        print(f"\nDetection Reasons:")
        for reason in result['detection_reasons']:
            print(f"   * {reason}")
        
        if result['technical_details']:
            print(f"\nTechnical Details:")
            for key, value in result['technical_details'].items():
                print(f"   {key}: {value}")
