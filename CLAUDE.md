# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A bot for the board game **Duat** using MENACE-style matchbox learning with self-play.

## Game Rules (Duat)

- 4x4 grid, 3 pieces per player
- Pieces have direction (can only move forward)
- Starting position:
  ```
  V..V
  ..V.
  .^..
  ^..^
  ```
- On your turn: make exactly **1 or 3 moves** (not 2)
- A move is either: slide a piece forward OR rotate a piece 90 degrees
- You can push pieces that aren't facing you
- **Win condition:** push an opponent's piece off the board

## Bot Architecture

**MENACE-style learning with self-play:**
- **State** = board state (position + direction of all 6 pieces), always from P0's perspective
- **Bead** = a legal turn from that state (either a 1-move or 3-move sequence)
- 1-move lookahead: always take immediate winning moves
- On loss: remove the bead for the last move made
- Empty state = proven losing position, with distance tracking

**Distance tracking:**
- `distance_to_win`: turns until forced win (if state is proven winner)
- `distance_to_loss`: turns until forced loss (if state is proven loser)
- Used for optimal play: prefer quickest wins, delay longest when losing

**State space:** Estimated 10k-100k reachable states. Small enough to potentially solve the game completely.

**Symmetry reduction:** Implemented with caching. Equivalent states are consolidated:
- 4 board rotations (0, 90, 180, 270 degrees)
- Player perspective normalization (always P0's turn in stored state)
- Piece interchangeability (a player's 3 pieces are indistinguishable)

## Project Structure

```
duat-bot/
├── src/           # Python source code
│   ├── duat.py        # Game logic
│   ├── ai_engine.py   # AI training and move selection
│   ├── ai_server.py   # FastAPI REST server
│   ├── storage.py     # Save/load AI state
│   └── web.py         # NiceGUI web interface
├── scripts/       # Shell scripts
│   ├── start_ai.sh    # Start AI server in background
│   └── stop_ai.sh     # Stop AI server gracefully
├── static/        # Static assets (images, favicon)
├── data/          # AI state files and logs (created at runtime)
└── CLAUDE.md
```

## Running the AI Server

```bash
pip install fastapi uvicorn httpx

# Option 1: Run in foreground
python3 src/ai_server.py

# Option 2: Run in background
./scripts/start_ai.sh
./scripts/stop_ai.sh  # To stop gracefully
```

Server runs on http://localhost:8000

The server:
- Trains when idle (no pending requests) from the starting position
- When a best_move request arrives, trains for 5 seconds from that state first
- Auto-saves every 1000 games to `data/ai_state.pkl`
- Provides REST endpoints for the web UI

## Web UI

```bash
pip install nicegui httpx
# First, start the AI server in another terminal:
python3 src/ai_server.py
# Then run the web UI:
python3 src/web.py
# Open http://localhost:8080
```

Features:
- Player vs Player
- Player vs AI (choose White or Black)
- AI vs AI (step through or auto-play)
- AI stats display (states, games trained, file size, training speed, proven wins/losses, win rate)

## REST API Endpoints

- `POST /best_move` - Train for 5 seconds from state, return best move
- `GET /stats` - Get AI statistics (includes games/sec, proven wins/losses)
- `GET /win_rate` - Get white win rate from recent training games
- `POST /save` - Force save AI state
- `POST /stop` - Stop server gracefully (finish batch, save, exit)

## Command Line Training

```bash
python3 src/ai_engine.py 10  # Train for 10,000 games
```

Or programmatically:

```python
from ai_engine import DuatAI
from storage import save_ai, load_ai

# Train a new AI
ai = DuatAI()
ai.train(10000, progress_interval=1000)
save_ai(ai, "data/trained.pkl")

# Load and use
ai = load_ai("data/trained.pkl")
turn = ai.get_best_move(state)  # Get optimal move for a GameState
```
