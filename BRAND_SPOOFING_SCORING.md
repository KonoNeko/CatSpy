# CatSpy Phishing Detection - Scoring System

## Scoring Formula

```
Final Score = AI Base Score + Heuristic Weights
           = min(max(base + extra, 0), 100)
```

**Classification Thresholds:**
- 0-33 points → SAFE
- 34-66 points → SUSPICIOUS
- 67-100 points → PHISHING

---

## 5 Detection Dimensions

| Dimension | Detection Target | Levels |
|-----------|-----------------|--------|
| **brand_spoof** | Brand impersonation (Microsoft/Google/Apple/etc) | none/possible/suspected/likely/confirmed |
| **creds_request** | Credential requests (password/login) | none/maybe/confirmed |
| **urgency_language** | Urgency pressure (urgent/expire) | none/low/high |
| **domain_risk** | Domain risks (IP/excessive digits) | ok/suspicious/high |
| **shortener_obfuscation** | URL shorteners (bit.ly/etc) | none/present |

---

## 4 Scoring Branches

System automatically selects scoring branch based on content characteristics:

### Branch 1: Gibberish Content
**Trigger:** No URLs + No HTTP + No threat keywords + (< 3 meaningful words OR length < 20 chars)  
**Base Formula:** `min(25, phishing_prob * 30)`  
**All Heuristic Weights:** 0 (disabled)

---

### Branch 2: Strong Heuristics ⚡
**Trigger:** Credential request OR High urgency OR Domain risk OR Shortener OR Brand spoofing  
**Base Formula:** 
- With URLs: `50 + (phishing_prob - 0.3) * 80`
- No URLs: `45 + max(0, (phishing_prob - 0.3) * 60)`

**Weight Table:**
| Dimension | none | low/possible | high/maybe/suspected | confirmed/present |
|-----------|------|-------------|---------------------|------------------|
| domain_risk | 0 | - | suspicious: 12 | high: 18 |
| urgency_language | 0 | 8 | high: 15 | - |
| creds_request | 0 | - | maybe: 12 | confirmed: 18 |
| shortener | 0 | - | present: 12 | - |
| **brand_spoof** | **0** | **15** | **25** | **likely: 45, confirmed: 50** |

---

### Branch 3: AI Predicts SAFE
**Trigger:** AI predicts SAFE (label=0) + No strong heuristics  
**Base Formula:** `max(0, (phishing_prob - 0.5) * 100)`

**Weight Table:**
| Dimension | none | low/possible | high/maybe/suspected | confirmed/present |
|-----------|------|-------------|---------------------|------------------|
| domain_risk | 0 | - | suspicious: 5 | high: 12 |
| urgency_language | 0 | 3 | high: 8 | - |
| creds_request | 0 | - | maybe: 5 | confirmed: 10 |
| shortener | 0 | - | present: 5 | - |
| **brand_spoof** | **0** | **15** | **30** | **likely: 70, confirmed: 80** |

---

### Branch 4a: AI Predicts PHISHING + Has URL
**Trigger:** AI predicts PHISHING (label=1) + Has URL/HTTP + No strong heuristics  
**Base Formula:** `50 + (phishing_prob - 0.5) * 100`

**Weight Table:**
| Dimension | none | low/possible | high/maybe/suspected | confirmed/present |
|-----------|------|-------------|---------------------|------------------|
| domain_risk | 0 | - | suspicious: 10 | high: 15 |
| urgency_language | 0 | 8 | high: 12 | - |
| creds_request | 0 | - | maybe: 10 | confirmed: 15 |
| shortener | 0 | - | present: 10 | - |
| **brand_spoof** | **0** | **8** | **18** | **likely: 30, confirmed: 35** |

---

### Branch 4b: AI Predicts PHISHING + No URL
**Trigger:** AI predicts PHISHING (label=1) + No URL/HTTP + No strong heuristics  
**Base Formula:** `min(40, 25 + (phishing_prob - 0.5) * 30)` (conservative cap)

**Weight Table:**
| Dimension | none | low/possible | high/maybe/suspected | confirmed/present |
|-----------|------|-------------|---------------------|------------------|
| domain_risk | 0 | - | suspicious: 3 | high: 8 |
| urgency_language | 0 | 5 | high: 10 | - |
| creds_request | 0 | - | maybe: 8 | confirmed: 12 |
| shortener | 0 | - | present: 5 | - |
| **brand_spoof** | **0** | **5** | **15** | **likely: 25, confirmed: 30** |

---

## Real-World Examples

**Example 1: `maicrosoft.com` (Typosquatting)**
```
Input: "Please visit https://www.maicrosoft.com/en-ca"
Detection: brand_spoof="likely" (90% similarity)
Route: Branch 2 (Strong Heuristics)
Calculation: Base=84 + Brand weight=45 = 129 → Capped at 100
Result: PHISHING (100/100)
```

**Example 2: `mmicrosoft.com` (Character Stuffing)**
```
Input: "Visit www.mmicrosoft.com"
Detection: brand_spoof="likely" (char repetition)
Route: Branch 2 (Strong Heuristics)
Calculation: Base=75 + Brand weight=45 = 120 → Capped at 100
Result: PHISHING (100/100)
```

**Example 3: `microsoft.com` (Legitimate)**
```
Input: "Visit microsoft.com"
Detection: Whitelisted, all checks=none
Route: Branch 3 (AI Predicts SAFE)
Calculation: Base=0 + Extra=0 = 0
Result: SAFE (0/100)
```

**Example 4: Multi-Attack**
```
Input: "URGENT! Your PayPal account suspended. Verify password at https://bit.ly/paypai-verify"
Detection: brand_spoof="likely" + creds="confirmed" + urgency="high" + shortener="present"
Route: Branch 2 (Strong Heuristics)
Calculation: Base=100 + (45+18+15+12)=90 → Capped at 100
Result: PHISHING (100/100)
```

---

## Brand Spoofing Detection

**Detection Methods:**
- Levenshtein similarity (maicrosoft vs microsoft)
- Character stuffing deduplication (mmicrosoft → microsoft)
- Fuzzy substring matching (microsoft-secure-login.com)
- Homoglyphs (micr0soft using digit 0)

**Whitelist Protection:**
```python
# Exact match + subdomain match
if domain == "microsoft.com" or domain.endswith(".microsoft.com"):
    return True  # Skip brand detection
```

---

## Test Results

| Domain | Score | Classification |
|--------|-------|----------------|
| `maicrosoft.com` | 89-100 | PHISHING |
| `mmicrosoft.com` | 71-100 | PHISHING |
| `g00gle.com` | 100 | PHISHING |
| `microsoft-login-secure.com` | 100 | PHISHING |
| `microsoft.com` | 0 | SAFE |
| `login.microsoft.com` | 0 | SAFE |

---

## Code Location

**File:** `backend/utils.py`

| Function | Lines | Purpose |
|----------|-------|---------|
| `check_brand_spoofing()` | ~107-172 | Brand impersonation detection |
| `check_heuristics()` | ~220-280 | Run 5 detection dimensions |
| `map_model_to_frontend()` | ~284-420 | Scoring algorithm (4-branch routing) |
| `is_whitelisted()` | ~78-105 | Whitelist checking |
| `extract_urls()` | ~173-203 | URL extraction |

**Key Data:**
```python
# Monitored brand list
BRAND_DOMAINS = {"microsoft": [...], "google": [...], "apple": [...]}

# Whitelisted domains
WHITELIST_DOMAINS = {"microsoft.com", "google.com", ...}

# Keyword libraries
URGENCY_KEYWORDS = ["urgent", "immediately", "expire", ...]
CREDENTIAL_KEYWORDS = ["password", "login", "verify", ...]
SHORTENER_DOMAINS = {"bit.ly", "tinyurl.com", ...}
```

---

## Design Principles

1. **Adaptive Weighting**: 4 branches allocate different weights based on content features
2. **AI + Rules Synergy**: Deep learning + expert rules complement each other
3. **False Positive Protection**: Whitelist + legitimate domain detection
4. **Aggressive Override**: Branches 2 & 3 can override conservative AI predictions
5. **Context Awareness**: URL presence affects both base and weights

**Test Status:** 6/6 scenarios passing  
**Last Updated:** 2026-02-08
