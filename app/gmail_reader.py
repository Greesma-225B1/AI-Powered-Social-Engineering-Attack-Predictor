import imaplib
import email

EMAIL = "b9majorproject@gmail.com"
PASSWORD = "tkgihldwubphcvev"

def fetch_unread_emails():
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")

    status, messages = mail.search(None, '(UNSEEN)')
    email_ids = messages[0].split()

    email_texts = []

    for e_id in email_ids:
        _, msg_data = mail.fetch(e_id, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    email_texts.append(part.get_payload(decode=True).decode())
        else:
            email_texts.append(msg.get_payload(decode=True).decode())

    mail.logout()
    return email_texts
