#!/usr/bin/env bash
# ============================================================
# auto_sync.sh — Automatic background rsync sync to GPU server
# Syncs ~/Desktop/sem7/UGP/ -> vm:/SML_DISK_24TB/rajeshr/Aryamann/UGP/
# ============================================================

LOCAL_DIR="/Users/aryamannsrivastava/Desktop/sem7/UGP"
REMOTE_DEST="vm:/SML_DISK_24TB/rajeshr/Aryamann/UGP"
INTERVAL=30 # seconds

echo "[auto_sync] Started background auto-sync to GPU server ($INTERVAL s interval)"
echo "  Local : $LOCAL_DIR"
echo "  Remote: $REMOTE_DEST"

while true; do
    rsync -az \
        --exclude 'venv/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude '.DS_Store' \
        --exclude '.git/' \
        --exclude 'node_modules/' \
        "$LOCAL_DIR/" "$REMOTE_DEST/" 2>/dev/null
    sleep "$INTERVAL"
done
