"""
Gmail API Integration Module
For fetching and parsing Gmail messages
"""
import os
import base64
import pickle
from typing import List, Dict, Optional
from datetime import datetime
from email.utils import parsedate_to_datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re

# Gmail API permission scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class GmailClient:
    """Gmail API Client"""
    
    def __init__(self, credentials_path: str = 'scripts/credentials.json', token_path: str = 'scripts/token.pickle'):
        """
        Initialize Gmail client
        
        Args:
            credentials_path: Path to OAuth credentials file
            token_path: Path to access token cache
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate and create Gmail API service"""
        creds = None
        
        # Try to load saved token
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, user login required
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # Disable HTTPS check (local dev only)
                import os as os_module
                os_module.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
                
                # Detect credential type (web or installed)
                import json
                with open(self.credentials_path, 'r') as f:
                    creds_data = json.load(f)
                
                is_web_app = 'web' in creds_data
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                
                print("\n" + "="*60)
                print(f"📧 Gmail Authorization ({'Web App' if is_web_app else 'Desktop App'})")
                print("="*60)
                
                if is_web_app:
                    # Web application: ALWAYS use manual method for reliability
                    print("\n📝 Manual Authorization Required\n")
                    
                    # Force specific redirect_uri to match credentials
                    flow.redirect_uri = "http://localhost:8080/"
                    
                    auth_url, _ = flow.authorization_url(
                        prompt='consent',
                        access_type='offline',
                        include_granted_scopes='true'
                    )
                    
                    print(f"1️⃣  Open this URL in your browser:\n")
                    print(f"   {auth_url}\n")
                    print(f"2️⃣  Sign in with your Gmail account")
                    print(f"3️⃣  Click 'Allow' to authorize CatSpy\n")
                    print(f"4️⃣  After authorization, browser will show error page")
                    print(f"    (This is normal! We need the URL from address bar)\n")
                    print(f"5️⃣  Copy the COMPLETE URL from browser address bar")
                    print(f"    Example: http://localhost:8080/?code=4/0AY0...&scope=...\n")
                    
                    redirect_url = input("📋 Paste the complete URL here: ").strip()
                    
                    if not redirect_url.startswith('http://localhost:8080/'):
                        print("❌ Error: URL must start with http://localhost:8080/")
                        print("Please restart and try again.")
                        exit(1)
                    
                    flow.fetch_token(authorization_response=redirect_url)
                    creds = flow.credentials
                    print("\n✅ Authorization successful!")
                else:
                    # Desktop application: simpler flow
                    print("\n✅ Browser will open automatically for authorization")
                    print("="*60 + "\n")
                    
                    try:
                        creds = flow.run_local_server(
                            port=8080,
                            authorization_prompt_message='Authorization successful!',
                            success_message='Authorization completed!',
                            open_browser=True
                        )
                    except Exception as e:
                        print(f"\n⚠️ Local server failed: {e}\n")
                        auth_url, _ = flow.authorization_url(
                            prompt='consent',
                            access_type='offline'
                        )
                        print(f"Open this URL:\n{auth_url}\n")
                        code = input("Enter the authorization code: ").strip()
                        flow.fetch_token(code=code)
                        creds = flow.credentials
            
            # Save credentials for next time
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('gmail', 'v1', credentials=creds)
        print("✅ Gmail API authenticated successfully")
    
    def get_user_profile(self) -> Optional[Dict]:
        """
        Get Gmail user profile information
        
        Returns:
            Dict with emailAddress, messagesTotal, threadsTotal
        """
        try:
            profile = self.service.users().getProfile(userId='me').execute()
            return {
                'email': profile.get('emailAddress', 'Unknown'),
                'messages_total': profile.get('messagesTotal', 0),
                'threads_total': profile.get('threadsTotal', 0)
            }
        except Exception as e:
            print(f"❌ Failed to get user profile: {e}")
            return None
    
    def fetch_latest_emails(self, limit: int = 50, query: str = '') -> List[Dict]:
        """
        Fetch latest emails list
        
        Args:
            limit: Maximum number of emails to fetch
            query: Gmail search query (e.g. 'is:unread' or 'after:2025/11/01')
        
        Returns:
            Email list with message_id, subject, sender, date, body
        """
        try:
            # Get email list
            results = self.service.users().messages().list(
                userId='me',
                maxResults=limit,
                q=query
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                print("📭 No emails found")
                return []
            
            emails = []
            print(f"📬 Fetching {len(messages)} emails...")
            
            for msg in messages:
                email_data = self._get_email_details(msg['id'])
                if email_data:
                    emails.append(email_data)
            
            print(f"✅ Successfully fetched {len(emails)} emails")
            return emails
        
        except HttpError as error:
            print(f"❌ Gmail API error: {error}")
            return []
    
    def _get_email_details(self, msg_id: str) -> Optional[Dict]:
        """
        Get single email details
        
        Args:
            msg_id: Gmail message ID
        
        Returns:
            Email details dictionary
        """
        try:
            message = self.service.users().messages().get(
                userId='me',
                id=msg_id,
                format='full'
            ).execute()
            
            headers = message['payload']['headers']
            
            # Extract email header info
            subject = self._get_header(headers, 'Subject') or '(No Subject)'
            sender = self._get_header(headers, 'From') or '(Unknown Sender)'
            date_str = self._get_header(headers, 'Date')
            
            # Parse date
            try:
                email_date = parsedate_to_datetime(date_str) if date_str else datetime.now()
            except:
                email_date = datetime.now()
            
            # Extract email body
            body = self._get_email_body(message)
            
            return {
                'message_id': msg_id,
                'subject': subject,
                'sender': sender,
                'date': email_date.isoformat(),
                'body': body
            }
        
        except HttpError as error:
            print(f"❌ Failed to fetch email {msg_id}: {error}")
            return None
    
    def _get_header(self, headers: List[Dict], name: str) -> Optional[str]:
        """Extract specified field from email headers"""
        for header in headers:
            if header['name'].lower() == name.lower():
                return header['value']
        return None
    
    def _get_email_body(self, message: Dict) -> str:
        """
        Extract email body (supports plain text and HTML)
        
        Args:
            message: Message object returned by Gmail API
        
        Returns:
            Email body text
        """
        body = ""
        
        if 'parts' in message['payload']:
            # Multi-part email
            for part in message['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                        break
                elif part['mimeType'] == 'text/html' and not body:
                    if 'data' in part['body']:
                        html_body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                        # Simple HTML cleanup (should use BeautifulSoup)
                        body = self._clean_html(html_body)
        else:
            # Single-part email
            if 'data' in message['payload']['body']:
                body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8', errors='ignore')
        
        return body[:5000]  # Limit length to avoid excessive size
    
    def _clean_html(self, html: str) -> str:
        """Simple HTML tag cleanup"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Clean extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# Global Gmail client instance (lazy init)
_gmail_client = None

def get_gmail_client() -> GmailClient:
    """Get Gmail client singleton"""
    global _gmail_client
    if _gmail_client is None:
        _gmail_client = GmailClient()
    return _gmail_client
