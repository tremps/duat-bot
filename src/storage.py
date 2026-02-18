"""Save and load AI state."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Union

from duat import Direction, encode_key, decode_key, encode_turn, decode_turn


def _serialize_key(key: int) -> str:
    """Convert int state key to JSON-compatible string."""
    pieces = decode_key(key)
    pieces_list = [(r, c, int(d), p) for r, c, d, p in pieces]
    return json.dumps(pieces_list)


def _deserialize_key(s: str) -> int:
    """Convert JSON string back to int state key."""
    pieces_list = json.loads(s)
    pieces = tuple((r, c, Direction(d), p) for r, c, d, p in pieces_list)
    return encode_key(pieces)


def _serialize_turn(turn_int: int) -> str:
    """Convert int-encoded turn to JSON-compatible string."""
    turn = decode_turn(turn_int)
    return json.dumps(turn)


def _deserialize_turn(s: str) -> int:
    """Convert JSON string back to int-encoded turn."""
    moves = json.loads(s)
    turn = tuple(tuple(m) for m in moves)
    return encode_turn(turn)


def _atomic_write(filepath: Path, write_fn) -> None:
    """Write to a temp file then atomically rename over the target.

    This prevents corruption if the process crashes mid-write.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (same filesystem = atomic rename)
    fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix=filepath.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
        os.replace(tmp_path, filepath)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _is_old_tuple_key(key) -> bool:
    """Check if a key is an old-format tuple (vs new int format)."""
    return isinstance(key, tuple)


def _migrate_old_key(key: tuple) -> int:
    """Convert old tuple key to new int key."""
    return encode_key(key)


def _migrate_old_turn(turn: tuple) -> int:
    """Convert old tuple turn to new int-encoded turn."""
    return encode_turn(turn)


def save_ai_pickle(ai, filepath: Union[str, Path]) -> None:
    """Save AI state using pickle (fast, compact)."""
    states_data = {}
    for key, info in ai.states.items():
        states_data[key] = {
            "beads": info.beads,  # set[int]
            "distance_to_win": info.distance_to_win,
            "distance_to_loss": info.distance_to_loss,
            "explored_depth": info.explored_depth,
        }

    def write_fn(f):
        pickle.dump({
            "states": states_data,
            "games_trained": ai.games_trained,
            "version": 2,  # int-encoded format
        }, f)

    _atomic_write(Path(filepath), write_fn)


def load_ai_pickle(filepath: Union[str, Path]):
    """Load AI state from pickle file, auto-migrating old tuple format."""
    from ai_engine import DuatAI, StateInfo

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    ai = DuatAI()
    ai.games_trained = data["games_trained"]

    version = data.get("version", 1)
    needs_migration = version < 2

    if needs_migration:
        print("Migrating AI state from tuple format to compact int format...")
        migrated = 0

    for key, info_data in data["states"].items():
        beads = info_data["beads"]

        if needs_migration and _is_old_tuple_key(key):
            # Migrate old tuple key to int
            key = _migrate_old_key(key)
            # Migrate old tuple beads to int set
            if not isinstance(beads, set):
                beads = set(beads)
            beads = {_migrate_old_turn(t) for t in beads}
            migrated += 1
        else:
            # New format or already int
            if not isinstance(beads, set):
                beads = set(beads)

        ai.states[key] = StateInfo(
            beads=beads,
            distance_to_win=info_data["distance_to_win"],
            distance_to_loss=info_data["distance_to_loss"],
            explored_depth=info_data.get("explored_depth", 0),
        )

    if needs_migration:
        print(f"Migration complete: {migrated:,} states converted")

    return ai


def save_ai_json(ai, filepath: Union[str, Path]) -> None:
    """Save AI state using JSON (human-readable, portable)."""
    serialized_states = {}
    for key, info in ai.states.items():
        key_str = _serialize_key(key)
        serialized_beads = [_serialize_turn(turn_int) for turn_int in info.beads]
        serialized_states[key_str] = {
            "beads": serialized_beads,
            "distance_to_win": info.distance_to_win,
            "distance_to_loss": info.distance_to_loss,
            "explored_depth": info.explored_depth,
        }

    data = {
        "games_trained": ai.games_trained,
        "states": serialized_states,
        "version": 2,
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
            explored_depth=info_data.get("explored_depth", 0),
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

    .pkl/.pickle -> pickle format
    .json -> JSON format
    """
    filepath = Path(filepath)
    load_fn = load_ai_pickle if filepath.suffix in (".pkl", ".pickle") else load_ai_json
    return load_fn(filepath)
