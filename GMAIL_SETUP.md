# Gmail API Setup Guide

Follow these steps to enable Gmail API and test with your own inbox.

## Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Create Project" (or select existing project)
3. Enter project name: `CatSpy-Local-Test`
4. Click "Create"

## Step 2: Enable Gmail API

1. In Google Cloud Console, go to **APIs & Services > Library**
2. Search for "Gmail API"
3. Click "Gmail API" and click "Enable"

## Step 3: Configure OAuth Consent Screen

1. Go to **APIs & Services > OAuth consent screen**
2. Select **External** user type (for testing)
3. Click "Create"
4. Fill in required fields:
   - App name: `CatSpy Local Test`
   - User support email: Your email
   - Developer contact: Your email
5. Click "Save and Continue"
6. **Scopes**: Click "Add or Remove Scopes"
   - Search and add: `https://www.googleapis.com/auth/gmail.readonly`
   - Click "Update" then "Save and Continue"
7. **Test users**: Click "Add Users"
   - Add your Gmail address (the one you want to scan)
   - Click "Add" then "Save and Continue"
8. Click "Back to Dashboard"

## Step 4: Create OAuth Credentials

1. Go to **APIs & Services > Credentials**
2. Click "+ Create Credentials" > "OAuth client ID"
3. Application type: **Desktop app**
4. Name: `CatSpy Desktop Client`
5. Click "Create"
6. **Download JSON**: Click the download button (⬇️)
7. Save as `credentials.json`

## Step 5: Install credentials.json

```powershell
# Copy the downloaded credentials.json to backend/scripts/
Copy-Item "C:\Users\YourName\Downloads\credentials.json" "C:\Users\dhc20\OneDrive\桌面\Catspy\CatSpy\backend\scripts\credentials.json"
```

Or manually copy the file to:
```
C:\Users\dhc20\OneDrive\桌面\Catspy\CatSpy\backend\scripts\credentials.json
```

## Step 6: Enable Gmail API in Code

The API is already enabled in the code. Just verify the setting:

File: `backend/api.py` (line 25)
```python
ENABLE_GMAIL_API = True  # ✅ Set to True
```

## Step 7: Start Application

```powershell
cd C:\Users\dhc20\OneDrive\桌面\Catspy\CatSpy\backend
.\.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Step 8: First-time Authorization

When you first scan emails:

1. Application will print an authorization URL in terminal
2. **Copy the URL** and open in your browser
3. Sign in with your Gmail account
4. Click "Continue" (you may see a warning - click "Continue" again)
5. Allow permissions
6. Browser redirects to `http://localhost/?code=...`
7. **Copy the ENTIRE URL** from browser address bar
8. Paste it back in the terminal

Example URL to copy:
```
http://localhost/?code=4/0AY...long_code...&scope=https://www.googleapis.com/auth/gmail.readonly
```

Token will be saved to `backend/scripts/token.pickle` for future use.

## Testing

1. Open frontend: http://127.0.0.1:8000/index.html
2. Go to "Email Security" tab
3. Click "🔍 Scan Emails"
4. Your real Gmail emails will be analyzed!

## Troubleshooting

### "Access blocked: This app's request is invalid"
- Make sure you added your email as a test user in OAuth consent screen

### "credentials.json not found"
- Check file path: `backend/scripts/credentials.json`
- Make sure you downloaded the correct OAuth client credentials (Desktop app type)

### "Token expired"
- Delete `backend/scripts/token.pickle`
- Restart application to re-authorize

### "Gmail API disabled"
- Verify `ENABLE_GMAIL_API = True` in `backend/api.py`

## Security Notes

⚠️ **Important**:
- `credentials.json` contains sensitive OAuth client info - DO NOT commit to git
- `token.pickle` contains your access token - DO NOT share
- For production, use service accounts or proper OAuth flows
- This setup is for LOCAL TESTING ONLY

## Privacy

- App only reads emails (readonly scope)
- No emails are stored permanently
- Only metadata and analysis results saved to local SQLite database
- You can revoke access anytime: https://myaccount.google.com/permissions
