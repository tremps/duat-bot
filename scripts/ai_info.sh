#!/bin/bash
curl -s "${AI_SERVER_URL:-http://localhost:8000}/stats"
echo
