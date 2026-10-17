"""
scripts/setup_test_env.py
─────────────────────────
One-time setup for the PTIS test environment.

What this does:
  1. Creates the ptis_test PostgreSQL database
  2. Runs all Alembic migrations on it
  3. Creates a 'dev' user (same as prod — DEV_AUTH_BYPASS uses this)
  4. Creates data/test/ directories

Run ONCE before using the test environment:
  python scripts/setup_test_env.py

After this, seed test data with:
  python scripts/seed_test_data.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Point to test env
os.environ["PTIS_ENV"] = "test"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def create_test_database():
    print("Creating ptis_test database...")
    # Connect to postgres default DB to create ptis_test
    import psycopg2
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="ptis",
            password="ptis",
            host="localhost",
            port=5432,
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = 'ptis_test'")
        if cur.fetchone():
            print("  ptis_test already exists — skipping creation")
        else:
            cur.execute("CREATE DATABASE ptis_test OWNER ptis")
            print("  ptis_test created ✓")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"  ERROR creating database: {e}")
        print("  Make sure PostgreSQL is running and 'ptis' user exists.")
        sys.exit(1)


def run_migrations():
    print("Running Alembic migrations on ptis_test...")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        env={**os.environ, "PTIS_ENV": "test"},
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr}")
        sys.exit(1)
    print("  Migrations applied ✓")


def create_dev_user():
    print("Creating dev user in ptis_test...")
    try:
        from core.database import SessionLocal
        from core.models import User
        from api.auth import pwd_context

        db = SessionLocal()
        existing = db.query(User).filter(User.username == "dev").first()
        if existing:
            print("  dev user already exists ✓")
        else:
            user = User(
                username="dev",
                email="dev@ptis.local",
                hashed_password=pwd_context.hash("dev"),
                is_active=True,
            )
            db.add(user)
            db.commit()
            print(f"  dev user created: id={user.id} ✓")
        db.close()
    except Exception as e:
        print(f"  ERROR creating dev user: {e}")


def create_directories():
    print("Creating test data directories...")
    dirs = [
        ROOT / "data" / "test" / "vector_store",
        ROOT / "data" / "test" / "screenshots",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  {d} ✓")


def main():
    print("=" * 55)
    print("  PTIS Test Environment Setup")
    print("=" * 55)
    print()
    create_test_database()
    run_migrations()
    create_dev_user()
    create_directories()
    print()
    print("=" * 55)
    print("  Test environment ready!")
    print()
    print("  Next step — seed test data:")
    print("    python scripts/seed_test_data.py")
    print()
    print("  Then start the test server:")
    print("    start_test.bat")
    print("  And test frontend (new terminal):")
    print("    start_test_frontend.bat")
    print("=" * 55)


if __name__ == "__main__":
    main()
