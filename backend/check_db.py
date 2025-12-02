import sqlite3

conn = sqlite3.connect('email_security.db')
cursor = conn.cursor()

print('Total emails:', cursor.execute('SELECT COUNT(*) FROM email_scans').fetchone()[0])

print('\nBy category:')
for row in cursor.execute('SELECT category, COUNT(*) FROM email_scans GROUP BY category').fetchall():
    print(f'  {row[0]}: {row[1]}')

print('\nBy risk level:')
for row in cursor.execute('SELECT risk_level, COUNT(*) FROM email_scans GROUP BY risk_level').fetchall():
    print(f'  {row[0]}: {row[1]}')

print('\nSample emails by category:')
print('\nSpam emails:')
for row in cursor.execute('SELECT subject FROM email_scans WHERE category="spam" LIMIT 3').fetchall():
    print(f'  - {row[0]}')

print('\nMalware emails:')
for row in cursor.execute('SELECT subject FROM email_scans WHERE category="malware" LIMIT 3').fetchall():
    print(f'  - {row[0]}')

conn.close()
