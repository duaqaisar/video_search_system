"""
Generates a new API key, stores its hash in Postgres, and prints the
raw key ONCE — this is the only time it's shown in plaintext.

Usage:
    python generate_api_key.py "client production key"
"""
import sys
import uuid
from app.db.database import SessionLocal
from app.db.models import ApiKey
from app.auth.api_key import generate_api_key


def main(label: str):
    raw_key, hashed_key = generate_api_key()

    db = SessionLocal()
    try:
        record = ApiKey(id=str(uuid.uuid4()), key_hash=hashed_key, label=label)
        db.add(record)
        db.commit()
    finally:
        db.close()

    print("API key created. Save this now — it will not be shown again:\n")
    print(raw_key)


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "unnamed key"
    main(label)
