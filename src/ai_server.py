"""FastAPI REST server for Duat AI.

A single worker thread handles all AI access (training + move computation).
Commands are sent via a queue; the event loop stays free for HTTP requests.
No locks needed since only the worker thread touches the AI.
"""

import asyncio
import os
import queue
import signal
import threading
import time
from collections import deque
from concurrent.futures import Future
from contextlib import asynccontextmanager
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
PROVEN_REFRESH_INTERVAL = 100  # Refresh proven counts every N batches

# Global state
ai: Optional[DuatAI] = None
_stop_event = threading.Event()
_work_queue: queue.Queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_last_saved_at = 0
_recent_results: deque = deque(maxlen=WIN_RATE_WINDOW)
_games_per_sec: float = 0.0

# Cached proven state counts (updated by worker thread)
_proven_wins: int = 0
_proven_losses: int = 0


def _load_or_create_ai() -> DuatAI:
    """Load AI from file or create new one."""
    if os.path.exists(AI_FILE):
        try:
            return load_ai(AI_FILE)
        except Exception as e:
            print(f"Warning: Could not load AI (or backup): {e}")
    return DuatAI()


def _save_ai():
    """Save AI to file. Only called from worker thread."""
    global _last_saved_at
    if ai:
        DATA_DIR.mkdir(exist_ok=True)
        save_ai(ai, AI_FILE)
        _last_saved_at = ai.games_trained


def _maybe_save():
    """Save if enough games since last save."""
    if ai and ai.games_trained - _last_saved_at >= AUTO_SAVE_INTERVAL:
        _save_ai()


def _train_batch():
    """Train a batch of games from the starting position."""
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
    """Recount proven wins/losses."""
    global _proven_wins, _proven_losses
    wins = 0
    losses = 0
    for info in ai.states.values():
        if info.distance_to_win is not None:
            wins += 1
        if info.distance_to_loss is not None:
            losses += 1
    _proven_wins = wins
    _proven_losses = losses


def _do_best_move(state: GameState) -> Optional[list]:
    """Train from state for BEST_MOVE_TRAIN_DURATION, then return best move."""
    start_time = time.monotonic()
    while time.monotonic() - start_time < BEST_MOVE_TRAIN_DURATION:
        if _stop_event.is_set():
            break
        ai.train_from_state(state, TRAIN_BATCH_SIZE)
    _maybe_save()
    turn = ai.get_best_move(state)
    return _serialize_turn(turn)


def _worker_loop():
    """Single worker thread: processes commands and trains when idle."""
    batches_since_refresh = 0

    while not _stop_event.is_set():
        # Check for commands (short timeout so we keep training)
        try:
            cmd, future = _work_queue.get(timeout=0.005)
        except queue.Empty:
            # No commands — train a batch
            _train_batch()
            batches_since_refresh += 1
            if batches_since_refresh >= PROVEN_REFRESH_INTERVAL:
                _refresh_proven_counts()
                batches_since_refresh = 0
            continue

        # Process command
        try:
            if cmd == "best_move":
                state = future._state_arg  # attached by caller
                result = _do_best_move(state)
                future.set_result(result)
                _refresh_proven_counts()
                batches_since_refresh = 0
            elif cmd == "save":
                _save_ai()
                future.set_result(None)
        except Exception as e:
            future.set_exception(e)


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


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai, _worker_thread, _last_saved_at

    print("Loading AI...")
    ai = _load_or_create_ai()
    _last_saved_at = ai.games_trained
    print(f"AI loaded: {len(ai.states):,} states, {ai.games_trained:,} games trained")

    _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
    _worker_thread.start()
    print("Worker thread started — training when idle")

    yield

    print("\nShutting down...")
    _stop_event.set()
    _worker_thread.join(timeout=10.0)

    print(f"Saving AI to {AI_FILE}...")
    _save_ai()
    print(f"AI saved: {len(ai.states):,} states, {ai.games_trained:,} games")


app = FastAPI(title="Duat AI Server", lifespan=lifespan)


# --- Endpoints ---

def _submit_command(cmd: str, **kwargs) -> Future:
    """Submit a command to the worker thread, return a Future for the result."""
    future = Future()
    for k, v in kwargs.items():
        setattr(future, f"_{k}_arg", v)
    _work_queue.put((cmd, future))
    return future


@app.post("/best_move", response_model=BestMoveResponse)
async def best_move(request: StateKeyRequest):
    """Get the best move for the given state."""
    state = _parse_state_key(request.state_key)
    future = _submit_command("best_move", state=state)
    result = await asyncio.wrap_future(future)
    return BestMoveResponse(turn=result)


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Get AI statistics. All values are cached/atomic — no blocking."""
    states = len(ai.states) if ai else 0
    games_trained = ai.games_trained if ai else 0

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
        proven_wins=_proven_wins,
        proven_losses=_proven_losses,
    )


@app.post("/save")
async def save_endpoint():
    """Force save the AI state."""
    future = _submit_command("save")
    await asyncio.wrap_future(future)
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
