
"""
@file backend/firebase_config.py
@description Firebase Admin SDK initialization.
@jules_hint Service account credentials should be managed via environment variables.
"""

import os
import firebase_admin
from firebase_admin import credentials, firestore, storage

def initialize_firebase():
    """Initializes the Firebase Admin SDK for high-performance retrieval."""
    if not firebase_admin._apps:
        # Assuming creds are injected in environment or local file for dev
        cert_path = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        if cert_path and os.path.exists(cert_path):
            cred = credentials.Certificate(cert_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'eudorax-studio.appspot.com'
            })
        else:
            # Fallback for local simulation mode
            print("[FIREBASE] Running in Local Simulation Mode")
            firebase_admin.initialize_app()
            
    db = firestore.client()
    bucket = storage.bucket()
    return db, bucket
