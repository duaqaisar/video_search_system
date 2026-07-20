"""
Simple API-key authentication for a single-client deployment.
Keys are stored hashed in Postgres (app.db.models.ApiKey).
"""
import hashlib
import secrets
from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ApiKey


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Returns (raw_key_to_give_the_client, hashed_key_to_store)."""
    raw_key = f"vsk_{secrets.token_urlsafe(32)}"
    return raw_key, hash_key(raw_key)


def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> ApiKey:
    hashed = hash_key(x_api_key)
    key_record = (
        db.query(ApiKey)
        .filter(ApiKey.key_hash == hashed, ApiKey.revoked == "false")
        .first()
    )
    if not key_record:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")
    return key_record
