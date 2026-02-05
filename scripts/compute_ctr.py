from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config


def now_unix() -> int:
    return int(time.time())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--since_hours", type=int, default=24)
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path: Path = cfg.data.sqlite_path

    since_ts = now_unix() - int(args.since_hours) * 3600

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Item CTR@K = clicks / impressions for positions <= K
    cur.execute(
        """
        SELECT COUNT(*) FROM interactions
        WHERE event_type='impression' AND position <= ? AND timestamp >= ?
        """,
        (int(args.k), since_ts),
    )
    impressions = int(cur.fetchone()[0])

    cur.execute(
        """
        SELECT COUNT(*) FROM interactions
        WHERE event_type='click' AND position <= ? AND timestamp >= ?
        """,
        (int(args.k), since_ts),
    )
    clicks = int(cur.fetchone()[0])

    item_ctr = (clicks / impressions) if impressions > 0 else 0.0

    # Search CTR@K = searches with >=1 click / total searches (within window)
    cur.execute(
        """
        SELECT COUNT(DISTINCT search_id) FROM interactions
        WHERE event_type='impression' AND timestamp >= ?
        """,
        (since_ts,),
    )
    searches = int(cur.fetchone()[0])

    cur.execute(
        """
        SELECT COUNT(DISTINCT search_id) FROM interactions
        WHERE event_type='click' AND position <= ? AND timestamp >= ?
        """,
        (int(args.k), since_ts),
    )
    searches_with_click = int(cur.fetchone()[0])

    search_ctr = (searches_with_click / searches) if searches > 0 else 0.0

    conn.close()

    print(f"Window: last {args.since_hours} hours (since {since_ts})")
    print(f"Impressions@{args.k}: {impressions}")
    print(f"Clicks@{args.k}:      {clicks}")
    print(f"Item CTR@{args.k}:    {item_ctr:.4f}")
    print(f"Searches:             {searches}")
    print(f"Searches w/ click@{args.k}: {searches_with_click}")
    print(f"Search CTR@{args.k}:  {search_ctr:.4f}")


if __name__ == "__main__":
    main()
