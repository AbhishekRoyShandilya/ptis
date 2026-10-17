"""
Reset the TEST database to a pristine, empty state.

Wipes ALL data (users, strategies, watchlists, trades, learning sessions,
AVT/judgment state, etc.) while preserving the schema and the Alembic migration
version — so you can sign up fresh and test the real user journey from zero.

SAFETY: refuses to run unless the target DB name contains "test". It can never
touch the production database.

Usage (PowerShell):
    $env:PTIS_ENV = "test"
    python scripts/reset_test_data.py
"""

from __future__ import annotations

import os
import sys

# Make the project root importable when run as `python scripts/reset_test_data.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from config.settings import settings
from core.database import SessionLocal


# Tables to preserve (never truncated)
_KEEP = {"alembic_version"}


def main() -> None:
    url = settings.DATABASE_URL

    # ── SAFETY GUARD — only ever run against a test database ──
    if "test" not in url.rsplit("/", 1)[-1].lower():
        print(f"REFUSING: target DB is not a test database -> {url}")
        print("Set PTIS_ENV=test so this targets ptis_test, then re-run.")
        sys.exit(1)

    db = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )).fetchall()
        tables = [r[0] for r in rows if r[0] not in _KEEP]

        if not tables:
            print("Nothing to truncate.")
            return

        quoted = ", ".join(f'"{t}"' for t in tables)
        # CASCADE handles all FK dependencies; RESTART IDENTITY resets sequences.
        db.execute(text(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE"))
        db.commit()

        db_name = url.rsplit("/", 1)[-1]
        print(f"Reset complete: truncated {len(tables)} tables in '{db_name}'.")
        print("Schema + migration version preserved. Sign up fresh to start clean.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
