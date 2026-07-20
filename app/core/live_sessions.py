"""
In-memory registry of active live ingestion sessions, so the API can
start/stop/query them by session_id. Simple dict for now -- fine for a
single-process demo; would need a shared store (Redis etc.) for multi-worker
production deployment later.
"""
from app.core.live_ingest import LiveIngestSession

_active_sessions: dict[str, LiveIngestSession] = {}


def start_session(
    session_id: str,
    source_url: str,
    language: str | None = None,
    owner_id: str | None = None,
) -> LiveIngestSession:
    """Start a new live ingestion session, scoped to the given owner_id
    (the authenticated user who started it -- see app/api/main.py)."""
    if session_id in _active_sessions:
        raise ValueError(f"Session '{session_id}' is already running.")
    session = LiveIngestSession(source_url, session_id, language=language, owner_id=owner_id)
    session.start()
    _active_sessions[session_id] = session
    return session


def get_session(session_id: str, owner_id: str | None = None) -> LiveIngestSession | None:
    """
    Look up a session by id. If owner_id is given, only returns the session
    if it belongs to that owner -- otherwise returns None as if it doesn't
    exist, so one user can't discover or read another user's live session
    (matches the ownership-check pattern used for videos in app/api/main.py's
    _get_owned_video_or_404).
    """
    session = _active_sessions.get(session_id)
    if session is None:
        return None
    if owner_id is not None and session.owner_id != owner_id:
        return None
    return session


def stop_session(session_id: str, owner_id: str | None = None) -> bool:
    """Stop and remove a session by id. Same ownership check as
    get_session() -- returns False (as if the session doesn't exist) if
    owner_id doesn't match, rather than allowing a cross-user stop."""
    session = get_session(session_id, owner_id=owner_id)
    if not session:
        return False
    session.stop()
    del _active_sessions[session_id]
    return True


def list_sessions(owner_id: str | None = None) -> list[str]:
    """List active session ids. If owner_id is given, only lists sessions
    belonging to that owner."""
    if owner_id is None:
        return list(_active_sessions.keys())
    return [sid for sid, s in _active_sessions.items() if s.owner_id == owner_id]
