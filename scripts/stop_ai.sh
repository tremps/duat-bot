#!/bin/bash
# Stop the Duat AI server gracefully via API.
# Creates a stop sentinel so the cron job won't restart it.
# Use start_ai.sh to clear the sentinel and start again.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

AI_SERVER_URL="${AI_SERVER_URL:-http://localhost:8000}"

# Create stop sentinel so cron doesn't restart the server
touch data/ai_server.stop

echo "Sending stop request to AI server..."
RESPONSE=$(curl -s -X POST "$AI_SERVER_URL/stop" 2>&1)

if echo "$RESPONSE" | grep -q "stopping"; then
    echo "Stop request sent successfully. Server will save and exit."

    # Wait for the server to actually stop
    echo "Waiting for server to stop..."
    for i in {1..30}; do
        if ! curl -s "$AI_SERVER_URL/stats" > /dev/null 2>&1; then
            echo "Server stopped."
            rm -f data/ai_server.pid
            exit 0
        fi
        sleep 1
    done
    echo "Warning: Server may still be stopping. Check data/ai_server.log"
else
    echo "Failed to send stop request: $RESPONSE"
    echo "Server may not be running or may be unreachable."

    # Kill the process directly if API isn't responding
    if [ -f data/ai_server.pid ]; then
        PID=$(cat data/ai_server.pid)
        if kill -0 "$PID" 2>/dev/null; then
            echo "Killing process (PID $PID)..."
            kill "$PID" 2>/dev/null
            rm -f data/ai_server.pid
            echo "Killed."
        else
            rm -f data/ai_server.pid
        fi
    fi
    exit 1
fi
