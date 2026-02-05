from datetime import datetime, timedelta
import random

def generate_test_emails():
    
    emails = [
        {"subject": "🚨 URGENT: Your PayPal Account Has Been Suspended!", "sender": "security@paypal-verify.tk", "body": "Your account shows suspicious activity. Click here IMMEDIATELY to verify: http://paypal-secure.tk/verify?token=x7k2j9"},
        {"subject": "Action Required: Verify Your PayPal Payment", "sender": "noreply@paypal-security.ml", "body": "We detected unusual payment. Confirm your identity now: https://paypal-confirm.ga/login"},
        {"subject": "PayPal: Your account will be limited", "sender": "service@paypal-alert.cf", "body": "Limited access in 24h unless you verify: http://paypal-restore.tk/account"},
        {"subject": "Important: PayPal Security Update Required", "sender": "updates@paypal-secure.ml", "body": "Update your security settings immediately: https://paypal-update.ga/secure"},
        {"subject": "PayPal Receipt: You sent $500.00", "sender": "receipts@paypal-tx.tk", "body": "If you didn't make this payment, click here: http://paypal-dispute.cf/cancel"},
        
        # Amazon Phishing (5)
        {"subject": "Amazon: Your account has been locked", "sender": "security@amazon-verify.ml", "body": "Unusual login attempts detected. Verify within 24 hours: https://amazon-restore.tk/signin"},
        {"subject": "Your Amazon order $899.99 has shipped", "sender": "orders@amazon-shipping.ga", "body": "If you didn't place this order, cancel here: http://amazon-cancel.cf/order?id=12345"},
        {"subject": "Amazon Prime: Payment method declined", "sender": "billing@amazon-prime.tk", "body": "Update payment to continue Prime benefits: https://amazon-payment.ml/update"},
        {"subject": "Congratulations! You won $500 Amazon Gift Card", "sender": "prizes@amazon-rewards.ga", "body": "Claim your prize before it expires: http://amazon-prize.tk/claim?code=WIN500"},
        {"subject": "Amazon Security Alert: New device login", "sender": "alerts@amazon-security.cf", "body": "Login from unknown device. Secure account: https://amazon-secure.ml/verify"},
        
        # Banking Phishing (5)
        {"subject": "Chase: Suspicious activity on your account", "sender": "fraud@chase-alert.tk", "body": "We detected fraudulent charges. Verify now: http://chase-verify.ml/fraud?urgent=yes"},
        {"subject": "Bank of America: Confirm large transaction", "sender": "security@bankofamerica-tx.ga", "body": "Transaction of $2,500 pending approval: https://boa-confirm.tk/transaction"},
        {"subject": "Wells Fargo: Your account requires verification", "sender": "alerts@wellsfargo-secure.cf", "body": "Unusual activity detected. Verify identity: http://wellsfargo-verify.ml/account"},
        {"subject": "Citibank: Account temporarily locked", "sender": "support@citibank-security.tk", "body": "Unlock your account immediately: https://citibank-unlock.ga/restore"},
        {"subject": "Your credit card has been charged $1,299", "sender": "billing@card-services.ml", "body": "If you didn't authorize, dispute here: http://dispute-charge.tk/cancel"},
        
        # Tech Company Phishing (5)
        {"subject": "Microsoft: Unusual sign-in activity", "sender": "account@microsoft-security.tk", "body": "Someone tried to access your account. Secure it: https://microsoft-verify.ml/security"},
        {"subject": "Apple ID has been locked", "sender": "noreply@appleid-support.ga", "body": "Your Apple ID locked due to suspicious activity: http://appleid-unlock.tk/restore"},
        {"subject": "Google Security Alert: New device", "sender": "security@google-alert.cf", "body": "Sign-in from Russia blocked. Review: https://google-security.ml/review"},
        {"subject": "Facebook: Your account will be disabled", "sender": "support@facebook-appeals.tk", "body": "Violation detected. Appeal within 24h: http://facebook-appeal.ga/submit"},
        {"subject": "Instagram: Someone logged into your account", "sender": "security@instagram-verify.ml", "body": "Unrecognized login detected. Secure account: https://instagram-secure.tk/login"},
        
        # Lottery/Prize Scams (5)
        {"subject": "YOU WON $5,000,000 in the International Lottery!", "sender": "winner@intl-lottery.tk", "body": "Congratulations! Claim your prize now: http://lottery-claim.ml/winner?id=5M"},
        {"subject": "IRS Tax Refund: $2,847 Approved", "sender": "refunds@irs-portal.ga", "body": "You qualify for tax refund. Submit bank details: https://irs-refund.tk/claim"},
        {"subject": "Walmart: You're Selected for $1000 Gift Card", "sender": "giveaway@walmart-promo.cf", "body": "Complete survey to claim: http://walmart-winner.ml/survey?prize=1000"},
        {"subject": "Publishers Clearing House: Final Notice", "sender": "prizes@pch-winner.tk", "body": "You won $10M! Call now to claim: 1-800-SCAM-NUM"},
        {"subject": "Google Rewards: Claim Your $500", "sender": "rewards@google-cash.ga", "body": "You've been selected for $500 reward: https://google-reward.tk/claim"},
        
        # Netflix/Streaming Scams (3)
        {"subject": "Netflix: Payment failed - Update now", "sender": "billing@netflix-payment.tk", "body": "Subscription will be cancelled. Update billing: https://netflix-billing.ml/update"},
        {"subject": "Spotify: Premium account suspended", "sender": "support@spotify-account.ga", "body": "Payment declined. Update method: http://spotify-payment.tk/renew"},
        {"subject": "Disney+: Your subscription is expiring", "sender": "billing@disneyplus-renew.cf", "body": "Renew now to continue watching: https://disney-renew.ml/payment"},
        
        # Delivery/Shipping Scams (5)
        {"subject": "DHL: Package delivery failed", "sender": "delivery@dhl-shipping.tk", "body": "Pay $4.99 customs fee to reschedule: http://dhl-customs.ml/pay?id=DHL9472"},
        {"subject": "USPS: Parcel held - customs fee required", "sender": "tracking@usps-delivery.ga", "body": "Package on hold. Pay fee: https://usps-customs.tk/payment"},
        {"subject": "FedEx: Delivery exception for your package", "sender": "alerts@fedex-track.cf", "body": "Address verification needed: http://fedex-verify.ml/address"},
        {"subject": "UPS: Failed delivery attempt", "sender": "delivery@ups-redelivery.tk", "body": "Reschedule delivery with $2.99 fee: https://ups-reschedule.ga/pay"},
        {"subject": "Amazon Delivery: Package requires signature", "sender": "logistics@amazon-deliver.ml", "body": "Confirm delivery details: http://amazon-delivery.tk/confirm"},
        
        # Job/Money Scams (4)
        {"subject": "LinkedIn: $95/hour remote job - No experience!", "sender": "jobs@linkedin-careers.tk", "body": "Start immediately! Provide bank details: https://linkedin-job.ml/apply"},
        {"subject": "Work From Home - Earn $5000/week easily", "sender": "hr@homejobs-247.ga", "body": "No experience needed! Sign up now: http://work-home.tk/register"},
        {"subject": "Mystery Shopper: Get Paid $500/assignment", "sender": "hiring@mystery-shop.cf", "body": "Deposit check and send $400 back: https://mystery-shopper.ml/signup"},
        {"subject": "Government Grant: You're approved for $50K", "sender": "grants@gov-funding.tk", "body": "Pay $95 processing fee to receive grant: http://gov-grants.ga/apply"},
        
        # Tech Support Scams (3)
        {"subject": "VIRUS ALERT: Your computer is infected!", "sender": "support@windows-security.ml", "body": "Call 1-800-VIRUS-FIX immediately! Critical threat detected!"},
        {"subject": "McAfee: 5 Threats Detected - Act Now", "sender": "alerts@mcafee-urgent.tk", "body": "Download security patch immediately: https://mcafee-fix.ga/download"},
        {"subject": "Norton: Subscription auto-renewed $349.99", "sender": "billing@norton-renew.cf", "body": "Cancel within 24h for refund: http://norton-cancel.ml/refund"},
        
        # Cryptocurrency Scams (3)
        {"subject": "Coinbase: Unusual withdrawal detected", "sender": "security@coinbase-alert.tk", "body": "2.5 BTC withdrawal pending. Cancel if not you: https://coinbase-freeze.ml/wallet"},
        {"subject": "Elon Musk Bitcoin Giveaway - 5 BTC Free!", "sender": "giveaway@tesla-crypto.ga", "body": "Send 0.5 BTC, get 5 BTC back! Limited time: http://tesla-btc.tk/giveaway"},
        {"subject": "Binance: Account verification required", "sender": "verify@binance-security.cf", "body": "Complete KYC to withdraw funds: https://binance-verify.ml/kyc"},
        
        # Social Media Scams (2)
        {"subject": "WhatsApp: Verification code 847392", "sender": "verify@whatsapp-code.tk", "body": "Someone is trying to register your number: http://whatsapp-verify.ml/security"},
        {"subject": "TikTok: Your account has been reported", "sender": "compliance@tiktok-review.ga", "body": "Verify within 48h to avoid ban: https://tiktok-verify.tk/appeal"},
        
        # Additional Phishing (2)
        {"subject": "Microsoft: Your OneDrive storage is full", "sender": "storage@microsoft-onedrive.tk", "body": "Upgrade now to avoid data loss: https://onedrive-upgrade.ml/storage", "type": "phishing"},
        {"subject": "IRS: You have unclaimed tax refund of $1,247", "sender": "refunds@irs-treasury.ga", "body": "Claim your refund before deadline: http://irs-refund-claim.tk/form", "type": "phishing"},
        
        # Spam Emails (8)
        {"subject": "Get 50% OFF on all supplements - Limited offer!", "sender": "deals@health-supplements.com", "body": "Buy now and save big on protein powders, vitamins, and more. Click here for special discount.", "type": "spam"},
        {"subject": "Congratulations! You're pre-approved for a credit card", "sender": "offers@creditcard-deals.net", "body": "Special offer just for you! Apply now for 0% APR for 12 months.", "type": "spam"},
        {"subject": "Hot singles in your area want to meet you!", "sender": "dating@meet-singles.xyz", "body": "Join now for free and start chatting with attractive people near you.", "type": "spam"},
        {"subject": "Lose 30 pounds in 30 days - Doctor recommended!", "sender": "health@weight-loss-miracle.com", "body": "Revolutionary new diet pill that actually works! Order now and get 2 bottles free.", "type": "spam"},
        {"subject": "Make $10,000 per month from home - No experience!", "sender": "opportunity@easy-money.biz", "body": "Join thousands who are earning passive income. Click here to learn the secret.", "type": "spam"},
        {"subject": "Enlarge your size naturally - Guaranteed results!", "sender": "health@male-enhancement.org", "body": "Clinically proven formula. See results in just 2 weeks. Discreet shipping.", "type": "spam"},
        {"subject": "Get your diploma online in 2 weeks!", "sender": "admissions@online-degrees.net", "body": "Accredited degrees from $299. No classes required. Order now!", "type": "spam"},
        {"subject": "Re: Re: Fw: Fw: You won't believe this video!", "sender": "viral@forward-this.com", "body": "OMG this is so funny! Watch before it gets deleted!!! Click here now!!!", "type": "spam"},
        
        # Malware Emails (5)
        {"subject": "Your invoice INV-98472 is attached", "sender": "billing@invoice-system.ru", "body": "Please review the attached invoice.exe file for payment details. Download attachment to view.", "type": "malware"},
        {"subject": "Security Update Required - Install Now", "sender": "updates@windows-update.tk", "body": "Critical security patch available. Download SecurityUpdate.exe immediately to protect your system.", "type": "malware"},
        {"subject": "Your document has been scanned - View PDF", "sender": "scanner@office-docs.ml", "body": "Scanned document ready. Open attached Document_Scan.pdf.exe to view.", "type": "malware"},
        {"subject": "Adobe Flash Player Update Available", "sender": "updates@adobe-flash.tk", "body": "Your Flash Player is outdated. Download and install FlashPlayer_Update.exe now.", "type": "malware"},
        {"subject": "Resume - John Smith.doc", "sender": "john.smith@jobapply.ru", "body": "Thank you for considering my application. Please review my attached resume.doc.exe file.", "type": "malware"},
    ]
    
    # Generate with dates
    for i, email in enumerate(emails):
        email['message_id'] = f'test-email-{i+1}'
        email['date'] = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
        if 'body' not in email:
            email['body'] = email['subject']
    
    return emails
