#!/bin/bash
# Start the Duat AI server as a background process

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Ensure data directory exists
mkdir -p data

# Check if already running
if [ -f data/ai_server.pid ]; then
    PID=$(cat data/ai_server.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo "AI server already running (PID $PID)"
        exit 1
    else
        rm data/ai_server.pid
    fi
fi

echo "Starting AI server..."
nohup python3 src/ai_server.py > data/ai_server.log 2>&1 &
PID=$!
echo $PID > data/ai_server.pid

# Wait briefly to check if it started successfully
sleep 1
if kill -0 "$PID" 2>/dev/null; then
    echo "AI server started (PID $PID)"
    echo "Log file: data/ai_server.log"
else
    echo "Failed to start AI server. Check data/ai_server.log for errors."
    rm -f data/ai_server.pid
    exit 1
fi
