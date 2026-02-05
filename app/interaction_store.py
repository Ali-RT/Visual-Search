from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


def now_unix() -> int:
    return int(time.time())


@dataclass(frozen=True)
class ImpressionRow:
    search_id: str
    query_sha256: str
    image_id: str
    position: int
    timestamp: int


class InteractionStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT,
              search_id TEXT NOT NULL,
              query_sha256 TEXT NOT NULL,
              displayed_image_id TEXT NOT NULL,
              position INTEGER NOT NULL,
              event_type TEXT NOT NULL,   -- 'impression' | 'click'
              timestamp INTEGER NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_interactions_type_time ON interactions(event_type, timestamp);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_interactions_search ON interactions(search_id);")
        conn.commit()
        conn.close()

    def log_impressions(self, rows: List[ImpressionRow]) -> None:
        if not rows:
            return
        conn = self._connect()
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO interactions(search_id, query_sha256, displayed_image_id, position, event_type, timestamp)
            VALUES(?,?,?,?, 'impression', ?)
            """,
            [(r.search_id, r.query_sha256, r.image_id, r.position, r.timestamp) for r in rows],
        )
        conn.commit()
        conn.close()

    def log_click(self, search_id: str, query_sha256: str, image_id: str, position: int) -> None:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO interactions(search_id, query_sha256, displayed_image_id, position, event_type, timestamp)
            VALUES(?,?,?,?, 'click', ?)
            """,
            (search_id, query_sha256, image_id, int(position), now_unix()),
        )
        conn.commit()
        conn.close()
