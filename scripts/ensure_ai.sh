#!/bin/bash
# Cron-friendly script: starts the AI server if it's not running.
# Does nothing if the stop sentinel exists (server was intentionally stopped).
#
# Add to crontab with: crontab -e
# * * * * * /path/to/duat-bot/scripts/ensure_ai.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# If stop sentinel exists, server was intentionally stopped — do nothing
if [ -f data/ai_server.stop ]; then
    exit 0
fi

# Check if already running
if [ -f data/ai_server.pid ]; then
    PID=$(cat data/ai_server.pid)
    if kill -0 "$PID" 2>/dev/null; then
        exit 0
    fi
fi

# Not running — start it
mkdir -p data
echo "[$(date)] Cron restarting AI server" >> data/ai_server.log
nohup python3 src/ai_server.py < /dev/null >> data/ai_server.log 2>&1 &
PID=$!
disown $PID
echo $PID > data/ai_server.pid
