import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def authenticate_gmail():
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    return service

def create_label_if_not_exists(service, label_name):
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])

    for label in labels:
        if label['name'] == label_name:
            return label['id']

    label_body = {
        'name': label_name,
        'labelListVisibility': 'labelShow',
        'messageListVisibility': 'show'
    }

    created_label = service.users().labels().create(
        userId='me',
        body=label_body
    ).execute()

    return created_label['id']

def apply_label(service, message_id, label_id):
    service.users().messages().modify(
        userId='me',
        id=message_id,
        body={
            'addLabelIds': [label_id]
        }
    ).execute()
    
import base64

def get_unread_messages(service):
    results = service.users().messages().list(
        userId='me',
        labelIds=['INBOX'],
        q="is:unread"
    ).execute()

    messages = results.get('messages', [])
    return messages


def get_email_body(service, message_id):
    message = service.users().messages().get(
        userId='me',
        id=message_id,
        format='full'
    ).execute()

    payload = message.get('payload', {})
    parts = payload.get('parts', [])

    body = ""

    if parts:
        for part in parts:
            if part.get('mimeType') == 'text/plain':
                data = part['body'].get('data')
                if data:
                    body = base64.urlsafe_b64decode(data).decode()
    else:
        data = payload.get('body', {}).get('data')
        if data:
            body = base64.urlsafe_b64decode(data).decode()

    return body