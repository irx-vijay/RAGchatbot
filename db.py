import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger("rag.db")


def get_conn():
    # creates a new PostgreSQL connection using DATABASE_URL from .env
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def init_db():
    """
    Runs once on app startup.
    Creates the chat_history table if it doesn't already exist.
    Safe to call multiple times — IF NOT EXISTS prevents duplicates.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id         SERIAL PRIMARY KEY,
                    session_id TEXT      NOT NULL,
                    role       TEXT      NOT NULL,
                    content    TEXT      NOT NULL,
                    sources    TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_session_id
                    ON chat_history(session_id);
            """)
        conn.commit()
    logger.info("DB initialised.")


def save_message(session_id: str, role: str, content: str, sources: list = None):
    """
    Saves one message (user or assistant) to PostgreSQL.
    sources is a list of filenames, stored as JSON string.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO chat_history (session_id, role, content, sources)
                   VALUES (%s, %s, %s, %s)""",
                (session_id, role, content, json.dumps(sources) if sources else None)
            )
        conn.commit()


def load_history(session_id: str, limit: int = 20) -> list:
    """
    Loads last 20 messages for a session from PostgreSQL.
    Returns them oldest first so chat displays in correct order.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT role, content
                   FROM chat_history
                   WHERE session_id = %s
                   ORDER BY created_at DESC
                   LIMIT %s""",
                (session_id, limit)
            )
            rows = cur.fetchall()
    # reversed because we fetched DESC (newest first) but want oldest first
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def delete_session(session_id: str):
    """Deletes all chat history for a session. Called on clear button."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_history WHERE session_id = %s",
                (session_id,)
            )
        conn.commit()
    logger.info("Deleted DB history for session %s", session_id)
