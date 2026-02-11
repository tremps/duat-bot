#!/bin/bash
# Stop the Duat AI server gracefully via API

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

AI_SERVER_URL="${AI_SERVER_URL:-http://localhost:8000}"

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
    exit 1
fi
