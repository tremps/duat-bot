"""Save and load AI state."""

from __future__ import annotations

import json
import pickle
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


def save_ai_pickle(ai, filepath: Union[str, Path]) -> None:
    """Save AI state using pickle (fast, compact)."""
    states_data = {}
    for key, info in ai.states.items():
        states_data[key] = {
            "beads": info.beads,  # set of turns
            "distance_to_win": info.distance_to_win,
            "distance_to_loss": info.distance_to_loss,
        }

    with open(filepath, "wb") as f:
        pickle.dump({
            "states": states_data,
            "games_trained": ai.games_trained,
        }, f)


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

    with open(filepath, "w") as f:
        json.dump(data, f)


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

    .pkl/.pickle -> pickle format
    .json -> JSON format
    """
    filepath = Path(filepath)
    if filepath.suffix in (".pkl", ".pickle"):
        return load_ai_pickle(filepath)
    else:
        return load_ai_json(filepath)
