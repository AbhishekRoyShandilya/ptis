#!/bin/bash
# scripts/backup_db.sh
# PostgreSQL backup with 7-day retention.
# Usage:  bash scripts/backup_db.sh
# Docker: docker compose exec db bash -c "pg_dump -U ptis ptis" | gzip > backup.gz

set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-./data/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
DB_USER="${POSTGRES_USER:-ptis}"
DB_NAME="${POSTGRES_DB:-ptis}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILENAME="ptis_${TIMESTAMP}.sql.gz"

mkdir -p "$BACKUP_DIR"

echo "Backing up $DB_NAME to $BACKUP_DIR/$FILENAME ..."

# Auto-detect: Docker compose or direct PostgreSQL
if command -v docker &> /dev/null && docker compose ps db 2>/dev/null | grep -q "running"; then
    docker compose exec -T db pg_dump -U "$DB_USER" "$DB_NAME" | gzip > "$BACKUP_DIR/$FILENAME"
else
    PGPASSWORD="${POSTGRES_PASSWORD:-ptis}" pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" | gzip > "$BACKUP_DIR/$FILENAME"
fi
echo "Backup complete: $BACKUP_DIR/$FILENAME ($(du -h "$BACKUP_DIR/$FILENAME" | cut -f1))"

# Prune backups older than retention period
PRUNED=$(find "$BACKUP_DIR" -name "ptis_*.sql.gz" -mtime +"$RETENTION_DAYS" -print -delete | wc -l)
if [ "$PRUNED" -gt 0 ]; then
    echo "Pruned $PRUNED backup(s) older than $RETENTION_DAYS days"
fi

echo "Current backups:"
ls -lh "$BACKUP_DIR"/ptis_*.sql.gz 2>/dev/null || echo "  (none)"
