import os
import logging
from supabase import create_client

logger = logging.getLogger("rag.storage")

BUCKET = "rag-uploads"


def get_client():
    # creates Supabase client using URL and KEY from .env
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def upload_file(file_bytes: bytes, filename: str, session_id: str) -> str:
    """
    Uploads file to Supabase Storage under session-scoped path.
    Path format: session_id/filename
    upsert=true means it overwrites if same file uploaded again.
    Returns the storage path.
    """
    path = f"{session_id}/{filename}"
    get_client().storage.from_(BUCKET).upload(
        path, file_bytes, {"upsert": "true"}
    )
    logger.info("Uploaded %s to Supabase Storage.", path)
    return path


def delete_session_files(session_id: str):
    """
    Deletes all files for a session from Supabase Storage.
    Called on clear button so no orphaned files remain.
    """
    client = get_client()
    files = client.storage.from_(BUCKET).list(session_id)
    if files:
        paths = [f"{session_id}/{f['name']}" for f in files]
        client.storage.from_(BUCKET).remove(paths)
        logger.info("Deleted %d file(s) for session %s", len(paths), session_id)
