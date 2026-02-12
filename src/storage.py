"""Save and load AI state."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Union

from duat import Direction


def _serialize_key(key: tuple) -> str:
    """Convert state key (pieces tuple) to JSON-compatible string."""
    pieces_list = [(r, c, int(d), p) for r, c, d, p in key]
    return json.dumps(pieces_list)


def _deserialize_key(s: str) -> tuple:
    """Convert JSON string back to state key."""
    pieces_list = json.loads(s)
    pieces = tuple((r, c, Direction(d), p) for r, c, d, p in pieces_list)
    return pieces


def _serialize_turn(turn: tuple) -> str:
    """Convert turn to JSON-compatible string."""
    return json.dumps(turn)


def _deserialize_turn(s: str) -> tuple:
    """Convert JSON string back to turn."""
    moves = json.loads(s)
    return tuple(tuple(m) for m in moves)


def _atomic_write(filepath: Path, write_fn) -> None:
    """Write to a temp file then atomically rename over the target.

    This prevents corruption if the process crashes mid-write.
    Also keeps a .bak backup of the previous file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (same filesystem = atomic rename)
    fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix=filepath.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
        # Keep a backup of the previous file
        if filepath.exists():
            backup = filepath.with_suffix(filepath.suffix + ".bak")
            # Replace backup atomically too
            os.replace(filepath, backup)
        os.rename(tmp_path, filepath)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_ai_pickle(ai, filepath: Union[str, Path]) -> None:
    """Save AI state using pickle (fast, compact)."""
    states_data = {}
    for key, info in ai.states.items():
        states_data[key] = {
            "beads": info.beads,  # set of turns
            "distance_to_win": info.distance_to_win,
            "distance_to_loss": info.distance_to_loss,
        }

    def write_fn(f):
        pickle.dump({
            "states": states_data,
            "games_trained": ai.games_trained,
        }, f)

    _atomic_write(Path(filepath), write_fn)


def load_ai_pickle(filepath: Union[str, Path]):
    """Load AI state from pickle file."""
    from ai_engine import DuatAI, StateInfo

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    ai = DuatAI()
    ai.games_trained = data["games_trained"]

    for key, info_data in data["states"].items():
        beads = info_data["beads"]
        # Ensure beads is a set (handle old list format)
        if not isinstance(beads, set):
            beads = set(beads)
        ai.states[key] = StateInfo(
            beads=beads,
            distance_to_win=info_data["distance_to_win"],
            distance_to_loss=info_data["distance_to_loss"],
        )

    return ai


def save_ai_json(ai, filepath: Union[str, Path]) -> None:
    """Save AI state using JSON (human-readable, portable)."""
    serialized_states = {}
    for key, info in ai.states.items():
        key_str = _serialize_key(key)
        serialized_beads = [_serialize_turn(turn) for turn in info.beads]
        serialized_states[key_str] = {
            "beads": serialized_beads,
            "distance_to_win": info.distance_to_win,
            "distance_to_loss": info.distance_to_loss,
        }

    data = {
        "games_trained": ai.games_trained,
        "states": serialized_states,
    }

    def write_fn(f):
        # json.dump expects text mode, wrap the binary file
        import io
        text_wrapper = io.TextIOWrapper(f, encoding="utf-8")
        json.dump(data, text_wrapper)
        text_wrapper.flush()
        # Detach so closing the wrapper doesn't close the underlying fd
        text_wrapper.detach()

    _atomic_write(Path(filepath), write_fn)


def load_ai_json(filepath: Union[str, Path]):
    """Load AI state from JSON file."""
    from ai_engine import DuatAI, StateInfo

    with open(filepath, "r") as f:
        data = json.load(f)

    ai = DuatAI()
    ai.games_trained = data["games_trained"]

    for key_str, info_data in data["states"].items():
        key = _deserialize_key(key_str)
        beads = {_deserialize_turn(t) for t in info_data["beads"]}
        ai.states[key] = StateInfo(
            beads=beads,
            distance_to_win=info_data["distance_to_win"],
            distance_to_loss=info_data["distance_to_loss"],
        )

    return ai


def save_ai(ai, filepath: Union[str, Path]) -> None:
    """
    Save AI state (auto-detect format from extension).

    .pkl/.pickle -> pickle format
    .json -> JSON format
    """
    filepath = Path(filepath)
    if filepath.suffix in (".pkl", ".pickle"):
        save_ai_pickle(ai, filepath)
    else:
        save_ai_json(ai, filepath)


def load_ai(filepath: Union[str, Path]):
    """
    Load AI state (auto-detect format from extension).
    Falls back to .bak backup if the main file is corrupt.

    .pkl/.pickle -> pickle format
    .json -> JSON format
    """
    filepath = Path(filepath)
    backup = filepath.with_suffix(filepath.suffix + ".bak")
    load_fn = load_ai_pickle if filepath.suffix in (".pkl", ".pickle") else load_ai_json

    # Try main file first
    try:
        return load_fn(filepath)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")

    # Try backup
    if backup.exists():
        print(f"Trying backup {backup}...")
        return load_fn(backup)

    raise FileNotFoundError(f"No valid AI state at {filepath} or {backup}")
