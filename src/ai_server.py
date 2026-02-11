"""FastAPI REST server for Duat AI.

All CPU-bound work (training, move computation) runs in a background thread.
A threading.Lock ensures the AI is only accessed by one thread at a time.
The asyncio event loop stays free to handle HTTP requests.
"""

import asyncio
import os
import signal
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from duat import GameState, Direction
from ai_engine import DuatAI
from storage import save_ai, load_ai


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AI_FILE = DATA_DIR / "ai_state.pkl"

# Config
TRAIN_BATCH_SIZE = 10
AUTO_SAVE_INTERVAL = 1000
BEST_MOVE_TRAIN_DURATION = 5.0
WIN_RATE_WINDOW = 1000

# Global state
ai: Optional[DuatAI] = None
_ai_lock = threading.Lock()
_stop_event = threading.Event()
_train_thread: Optional[threading.Thread] = None
_last_saved_at = 0
_recent_results: deque = deque(maxlen=WIN_RATE_WINDOW)
_games_per_sec: float = 0.0

# Cached proven state counts (updated by training thread)
_proven_wins: int = 0
_proven_losses: int = 0
_proven_counts_stale: bool = True
PROVEN_REFRESH_INTERVAL = 100  # Refresh every N batches


def _load_or_create_ai() -> DuatAI:
    """Load AI from file or create new one."""
    if os.path.exists(AI_FILE):
        try:
            return load_ai(AI_FILE)
        except Exception as e:
            print(f"Warning: Could not load AI from {AI_FILE}: {e}")
    return DuatAI()


def _save_ai():
    """Save AI to file. Must be called with _ai_lock held."""
    global _last_saved_at
    if ai:
        DATA_DIR.mkdir(exist_ok=True)
        save_ai(ai, AI_FILE)
        _last_saved_at = ai.games_trained


def _maybe_save():
    """Save if enough games since last save. Must be called with _ai_lock held."""
    if ai and ai.games_trained - _last_saved_at >= AUTO_SAVE_INTERVAL:
        _save_ai()


def _train_batch():
    """Train a batch of games. Must be called with _ai_lock held."""
    global _games_per_sec
    start_state = GameState.initial()
    start_time = time.monotonic()
    for _ in range(TRAIN_BATCH_SIZE):
        winner = ai.train_game(start_state)
        _recent_results.append(winner)
    elapsed = time.monotonic() - start_time
    if elapsed > 0:
        _games_per_sec = TRAIN_BATCH_SIZE / elapsed
    _maybe_save()


def _refresh_proven_counts():
    """Recount proven wins/losses. Must be called with _ai_lock held."""
    global _proven_wins, _proven_losses, _proven_counts_stale
    wins = 0
    losses = 0
    for info in ai.states.values():
        if info.distance_to_win is not None:
            wins += 1
        if info.distance_to_loss is not None:
            losses += 1
    _proven_wins = wins
    _proven_losses = losses
    _proven_counts_stale = False


def _training_loop():
    """Background thread: train continuously until stop is requested."""
    batches_since_refresh = 0
    while not _stop_event.is_set():
        with _ai_lock:
            _train_batch()
            batches_since_refresh += 1
            if _proven_counts_stale or batches_since_refresh >= PROVEN_REFRESH_INTERVAL:
                _refresh_proven_counts()
                batches_since_refresh = 0
        # Yield to let best_move/stats requests proceed
        _stop_event.wait(0.001)


def _compute_best_move(state: GameState) -> Optional[list]:
    """Train from state, then return best move."""
    global _proven_counts_stale
    start_time = time.monotonic()
    while time.monotonic() - start_time < BEST_MOVE_TRAIN_DURATION:
        if _stop_event.is_set():
            break
        with _ai_lock:
            ai.train_from_state(state, TRAIN_BATCH_SIZE)
    _proven_counts_stale = True
    with _ai_lock:
        _maybe_save()
        turn = ai.get_best_move(state)
        return _serialize_turn(turn)


def _get_stats_snapshot():
    """Collect stats. Simple reads are atomic in CPython; proven counts are cached."""
    states = len(ai.states) if ai else 0
    games_trained = ai.games_trained if ai else 0
    return states, games_trained, _proven_wins, _proven_losses


# --- Helpers ---

def _parse_state_key(state_key: list) -> GameState:
    """Parse a state key from JSON format to GameState."""
    pieces = tuple(
        (r, c, Direction(d), p)
        for r, c, d, p in state_key
    )
    return GameState(pieces)


def _serialize_turn(turn) -> Optional[list]:
    """Serialize a turn to JSON format."""
    if turn is None:
        return None
    return [[piece_idx, action] for piece_idx, action in turn]


# --- Models ---

class StateKeyRequest(BaseModel):
    state_key: list


class BestMoveResponse(BaseModel):
    turn: Optional[list]


class StatsResponse(BaseModel):
    states: int
    games_trained: int
    file_size_bytes: int
    white_win_rate: Optional[float]
    win_rate_sample_size: int
    games_per_second: float
    proven_wins: int
    proven_losses: int


class WinRateResponse(BaseModel):
    white_win_rate: Optional[float]
    sample_size: int
    draws: int


app = FastAPI(title="Duat AI Server")


@app.on_event("startup")
async def startup():
    global ai, _train_thread, _last_saved_at

    print("Loading AI...")
    ai = _load_or_create_ai()
    _last_saved_at = ai.games_trained
    print(f"AI loaded: {len(ai.states):,} states, {ai.games_trained:,} games trained")

    _train_thread = threading.Thread(target=_training_loop, daemon=True)
    _train_thread.start()
    print("Training thread started")


@app.on_event("shutdown")
async def shutdown():
    print("\nShutting down...")
    _stop_event.set()
    _train_thread.join(timeout=10.0)

    print(f"Saving AI to {AI_FILE}...")
    _save_ai()
    print(f"AI saved: {len(ai.states):,} states, {ai.games_trained:,} games")


# --- Endpoints ---

@app.post("/best_move", response_model=BestMoveResponse)
async def best_move(request: StateKeyRequest):
    """Get the best move for the given state."""
    state = _parse_state_key(request.state_key)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _compute_best_move, state)
    return BestMoveResponse(turn=result)


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Get AI statistics. All values are cached/atomic reads — no lock needed."""
    states, games_trained, proven_wins, proven_losses = _get_stats_snapshot()

    file_size = 0
    if os.path.exists(AI_FILE):
        file_size = os.path.getsize(AI_FILE)

    # Win rate from deque snapshot
    results = list(_recent_results)
    win_rate = None
    sample_size = len(results)
    decisive = [r for r in results if r is not None]
    if decisive:
        win_rate = sum(1 for r in decisive if r == 0) / len(decisive)

    return StatsResponse(
        states=states,
        games_trained=games_trained,
        file_size_bytes=file_size,
        white_win_rate=win_rate,
        win_rate_sample_size=sample_size,
        games_per_second=_games_per_sec,
        proven_wins=proven_wins,
        proven_losses=proven_losses,
    )


def _locked_save():
    with _ai_lock:
        _save_ai()


@app.post("/save")
async def save_endpoint():
    """Force save the AI state."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _locked_save)
    return {"status": "saved"}


@app.get("/win_rate", response_model=WinRateResponse)
async def win_rate():
    """Get white win rate from recent training games."""
    results = list(_recent_results)
    if not results:
        return WinRateResponse(white_win_rate=None, sample_size=0, draws=0)

    draws = sum(1 for r in results if r is None)
    decisive = [r for r in results if r is not None]

    if not decisive:
        return WinRateResponse(white_win_rate=None, sample_size=len(results), draws=draws)

    white_wins = sum(1 for r in decisive if r == 0)
    return WinRateResponse(
        white_win_rate=white_wins / len(decisive),
        sample_size=len(results),
        draws=draws,
    )


@app.post("/stop")
async def stop_endpoint():
    """Stop the AI server gracefully."""
    _stop_event.set()
    # Signal uvicorn to shut down cleanly
    os.kill(os.getpid(), signal.SIGINT)
    return {"status": "stopping"}


if __name__ == "__main__":
    import uvicorn

    print("Starting Duat AI Server...")
    print("Press Ctrl+C to stop")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
