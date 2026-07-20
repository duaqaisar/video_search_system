"""
Cleanup script: removes test/debug videos, their jobs, and their Qdrant
vectors. Run with --dry-run first to see what would be deleted.

Usage:
    python cleanup_test_data.py --dry-run
    python cleanup_test_data.py --confirm
"""
import sys
from app.db.database import SessionLocal
from app.db.models import Video, ProcessingJob
from app.core.qdrant_indexing import _get_client, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue


def get_all_videos(db):
    return db.query(Video).order_by(Video.created_at.desc()).all()


def delete_video(db, video: Video, dry_run: bool):
    print(f"  Deleting video_id={video.id} ({video.filename})")

    if not dry_run:
        # Delete jobs first (foreign key)
        db.query(ProcessingJob).filter(ProcessingJob.video_id == video.id).delete()
        db.delete(video)
        db.commit()

        # Delete matching vectors from Qdrant
        client = _get_client()
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video.id))]
            ),
        )


def delete_orphan_qdrant_entries(dry_run: bool, known_ids: set):
    """
    Deletes Qdrant points whose video_id isn't a real UUID in our DB
    (e.g. leftover manual-test entries like 'client_test_clip' or 'dua').
    """
    client = _get_client()
    scroll_result, _ = client.scroll(collection_name=COLLECTION_NAME, limit=1000)

    orphan_video_ids = set()
    for point in scroll_result:
        vid = point.payload.get("video_id")
        if vid and vid not in known_ids:
            orphan_video_ids.add(vid)

    for vid in orphan_video_ids:
        print(f"  Deleting orphan Qdrant entries for video_id='{vid}' (not a real DB record)")
        if not dry_run:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="video_id", match=MatchValue(value=vid))]
                ),
            )


def main(dry_run: bool):
    db = SessionLocal()
    try:
        videos = get_all_videos(db)
        print(f"Found {len(videos)} video records in Postgres.\n")

        known_ids = {v.id for v in videos}

        print("=== Videos to delete ===")
        for video in videos:
            delete_video(db, video, dry_run)

        print("\n=== Orphan Qdrant entries (not tied to any DB record) ===")
        delete_orphan_qdrant_entries(dry_run, known_ids)

        if dry_run:
            print("\nDry run complete. No changes made. Re-run with --confirm to actually delete.")
        else:
            print("\nCleanup complete.")
    finally:
        db.close()


if __name__ == "__main__":
    dry_run = "--confirm" not in sys.argv
    main(dry_run)
