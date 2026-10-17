#!/bin/bash
set -e

echo "══ PTIS Docker Entrypoint ══"

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
until python -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.connect(('db', 5432))
    s.close()
    exit(0)
except:
    exit(1)
" 2>/dev/null; do
  sleep 1
done
echo "PostgreSQL is ready."

# Run Alembic migrations
echo "Running database migrations..."
alembic upgrade head
echo "Migrations complete."

# Seed strategies (idempotent — skips existing)
echo "Seeding strategy data..."
python scripts/init_db.py
echo "Seeding complete."

# Start server
echo "Starting PTIS backend..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
