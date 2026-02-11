"""MENACE-style matchbox AI with distance tracking for Duat."""

import random
from dataclasses import dataclass, field
from typing import Optional
from duat import GameState, Turn


@dataclass
class StateInfo:
    """Information about a game state."""

    beads: set[Turn] = field(default_factory=set)  # available turns (removed when proven to lose)
    distance_to_win: Optional[int] = None   # turns until forced win (if proven)
    distance_to_loss: Optional[int] = None  # turns until forced loss (if proven)

    def is_proven_loss(self) -> bool:
        """True if this state is a proven loss (no beads left)."""
        return len(self.beads) == 0

    def available_turns(self) -> list[Turn]:
        """Get turns that still have beads."""
        return list(self.beads)


class DuatAI:
    """
    Matchbox learning AI with distance tracking.

    Each state stores:
    - beads: possible turns with their counts
    - distance_to_win: how many turns until forced win (if proven winner)
    - distance_to_loss: how many turns until forced loss (if proven loser)

    Key insight: After a move from state S to S', we swap players.
    So from S's perspective:
    - S'.distance_to_loss = opponent loses in N turns -> we win in N+1
    - S'.distance_to_win = opponent wins in N turns -> we lose in N+1
    """

    def __init__(self):
        """Initialize the AI."""
        self.states: dict[tuple, StateInfo] = {}
        self.games_trained = 0

    def get_or_create_state(self, canonical: GameState, key: tuple) -> StateInfo:
        """Get or create the StateInfo for a canonical state."""
        if key not in self.states:
            # Check for immediate win first - this is fast and avoids
            # computing all legal turns when we can just win
            winning_turn = canonical.find_winning_turn()
            if winning_turn:
                # Immediate win - store just this one turn
                self.states[key] = StateInfo(
                    beads={winning_turn},
                    distance_to_win=1
                )
            else:
                # No immediate win - compute all legal turns
                turns = canonical.get_legal_turns()
                self.states[key] = StateInfo(beads=set(turns))

        return self.states[key]

    def _transform_turn(self, turn: Turn, idx_mapping: dict[int, int], flip_rotations: bool = False) -> Turn:
        """
        Transform piece indices in a turn using the given mapping.

        If flip_rotations is True, swap CW <-> CCW (needed when canonical used horizontal flip).
        """
        result = []
        for piece_idx, action in turn:
            new_idx = idx_mapping[piece_idx]
            if flip_rotations:
                if action == "CW":
                    action = "CCW"
                elif action == "CCW":
                    action = "CW"
            result.append((new_idx, action))
        return tuple(result)

    def _invert_mapping(self, mapping: dict[int, int]) -> dict[int, int]:
        """Invert a mapping: {a: b} -> {b: a}."""
        return {v: k for k, v in mapping.items()}

    def get_best_move(self, state: GameState) -> Optional[Turn]:
        """
        Get the best move for a state.

        Priority:
        1. Take immediate winning move (prefer 1-move over 3-move)
        2. If we have beads, prefer moves leading to opponent's proven loss
        3. If no beads (proven loss), maximize survival (opponent's longest win)
        4. Random among available options
        """
        canonical, orig_to_canon, is_flipped = state.canonical_with_mapping()
        key = canonical.to_key()
        info = self.get_or_create_state(canonical, key)
        available = info.available_turns()

        canon_to_orig = self._invert_mapping(orig_to_canon)

        if available:
            # Check for 1-move wins first
            for turn in available:
                if len(turn) == 1:
                    result = canonical.apply_turn(turn)
                    if result and result.get_winner() == 0:
                        # Immediate win - prune to just this move
                        info.beads = {turn}
                        info.distance_to_win = 1
                        return self._transform_turn(turn, canon_to_orig, is_flipped)

            # Check for 3-move wins
            for turn in available:
                if len(turn) == 3:
                    result = canonical.apply_turn(turn)
                    if result and result.get_winner() == 0:
                        # Immediate win - prune to just this move
                        info.beads = {turn}
                        info.distance_to_win = 1
                        return self._transform_turn(turn, canon_to_orig, is_flipped)

            # Have beads - pick move with lowest opponent distance_to_loss (quickest win for us)
            best_turn = None
            best_score = float('inf')

            for turn in available:
                result = canonical.apply_turn(turn)
                if result is None:
                    continue
                result_swapped = result.swap_players()
                result_key = result_swapped.canonical().to_key()
                result_info = self.states.get(result_key)

                if result_info and result_info.distance_to_loss is not None:
                    if result_info.distance_to_loss < best_score:
                        best_score = result_info.distance_to_loss
                        best_turn = turn

            if best_turn is None:
                # No proven path, pick random
                best_turn = random.choice(available)

            return self._transform_turn(best_turn, canon_to_orig, is_flipped)

        # No beads - proven loss, maximize survival
        all_turns = canonical.get_legal_turns()
        if not all_turns:
            return None

        best_turn = None
        best_dtw = -1

        for turn in all_turns:
            result = canonical.apply_turn(turn)
            if result is None:
                continue
            result_swapped = result.swap_players()
            result_key = result_swapped.canonical().to_key()
            result_info = self.states.get(result_key)

            if result_info and result_info.distance_to_win is not None:
                if result_info.distance_to_win > best_dtw:
                    best_dtw = result_info.distance_to_win
                    best_turn = turn

        if best_turn is None:
            best_turn = random.choice(all_turns)

        return self._transform_turn(best_turn, canon_to_orig, is_flipped)

    def choose_turn(self, canonical: GameState, key: tuple,
                    orig_to_canon: dict[int, int], is_flipped: bool) -> Optional[tuple[Turn, Turn]]:
        """
        Choose a turn for the current state (for training).

        1. If there's a winning move, take it (prefer 1-move over 3-move).
        2. Otherwise, random selection from available beads.
        3. Returns None if this is a losing state (no beads).

        Returns (original_space_turn, canonical_space_turn) or None.
        """
        info = self.get_or_create_state(canonical, key)
        available = info.available_turns()

        if not available:
            return None

        canon_to_orig = self._invert_mapping(orig_to_canon)

        # Check for 1-move wins first
        for turn in available:
            if len(turn) == 1:
                result = canonical.apply_turn(turn)
                if result and result.get_winner() == 0:
                    # Immediate win - prune to just this move
                    info.beads = {turn}
                    info.distance_to_win = 1
                    return (self._transform_turn(turn, canon_to_orig, is_flipped), turn)

        # Check for 3-move wins
        for turn in available:
            if len(turn) == 3:
                result = canonical.apply_turn(turn)
                if result and result.get_winner() == 0:
                    # Immediate win - prune to just this move
                    info.beads = {turn}
                    info.distance_to_win = 1
                    return (self._transform_turn(turn, canon_to_orig, is_flipped), turn)

        # Check for forced win in 2 turns (move to state with distance_to_loss == 1)
        for turn in available:
            result = canonical.apply_turn(turn)
            if result is None:
                continue
            result_swapped = result.swap_players()
            result_key = result_swapped.canonical().to_key()
            result_info = self.states.get(result_key)
            if result_info and result_info.distance_to_loss == 1:
                # Forced win in 2 - prune to this move
                info.beads = {turn}
                info.distance_to_win = 2
                return (self._transform_turn(turn, canon_to_orig, is_flipped), turn)

        # No winning move, pick random from available beads
        canonical_turn = random.choice(available)
        return (self._transform_turn(canonical_turn, canon_to_orig, is_flipped), canonical_turn)

    def update_loss(self, key: tuple, canonical_turn: Turn) -> Optional[tuple]:
        """
        Remove a bead for a losing move.

        Args:
            key: Canonical state key.
            canonical_turn: Turn in canonical space.

        Returns the canonical key if this caused the state to become a proven loss,
        or None otherwise.
        """
        info = self.states.get(key)
        if info is None:
            return None

        if canonical_turn in info.beads:
            info.beads.discard(canonical_turn)

            # Check if state just became a proven loss
            if len(info.beads) == 0:
                return key

        return None

    def _mark_proven_loss(self, key: tuple, previous_key: Optional[tuple] = None):
        """
        Mark a state as proven loss and calculate distance_to_loss.

        When a state has 0 beads, find the best survival move
        (the move leading to opponent's highest distance_to_win).
        """
        info = self.states.get(key)
        if info is None:
            return

        state = GameState.from_key(key)

        # Find best survival move (maximize opponent's distance_to_win)
        best_dtw = -1
        for turn in state.get_legal_turns():
            result = state.apply_turn(turn)
            if result is None:
                continue
            result_swapped = result.swap_players()
            result_key = result_swapped.canonical().to_key()
            result_info = self.states.get(result_key)

            if result_info and result_info.distance_to_win is not None:
                best_dtw = max(best_dtw, result_info.distance_to_win)

        if best_dtw >= 0:
            info.distance_to_loss = best_dtw + 1

        # Trigger backpropagation to previous state
        if previous_key:
            self._check_winning_state(previous_key)

    def _check_winning_state(self, key: tuple):
        """
        Check if a state is now a proven win.

        A state is a proven win if ALL moves with beads lead to
        opponent states that have distance_to_loss set.
        """
        info = self.states.get(key)
        if info is None or info.distance_to_win is not None:
            return

        state = GameState.from_key(key)
        available = info.available_turns()

        if not available:
            return  # This is a loss, not a win

        all_have_dtl = True
        min_dtl = float('inf')
        best_turn = None

        for turn in available:
            result = state.apply_turn(turn)
            if result is None:
                all_have_dtl = False
                break

            result_swapped = result.swap_players()
            result_key = result_swapped.canonical().to_key()
            result_info = self.states.get(result_key)

            if result_info and result_info.distance_to_loss is not None:
                if result_info.distance_to_loss < min_dtl:
                    min_dtl = result_info.distance_to_loss
                    best_turn = turn
            else:
                all_have_dtl = False
                break

        if all_have_dtl and best_turn is not None:
            info.beads = {best_turn}
            info.distance_to_win = int(min_dtl) + 1

    def train_game(self, start_state: Optional[GameState] = None) -> Optional[int]:
        """
        Play one self-play training game.

        Args:
            start_state: Optional starting state. If None, uses initial position.

        Returns the winning player (0 or 1), or None for draw/error.

        Learning logic:
        - If a player reaches a losing state (no beads), the move that
          brought them there is removed.
        - If a player's move allows opponent to win immediately, that
          move is removed.
        - Draw detected when same state repeats 3 times.
        """
        self.games_trained += 1
        state = start_state if start_state else GameState.initial()
        current_player = 0  # Track who's actually playing (for external reference)

        # (canonical_key, turn) for each player's last move
        last_move: dict[int, tuple[tuple, Turn]] = {}
        state_counts: dict[tuple, int] = {}  # track repetitions

        while True:
            # Compute canonical form once per iteration
            canonical, orig_to_canon, is_flipped = state.canonical_with_mapping()
            key = canonical.to_key()

            # Check for draw by repetition
            state_counts[key] = state_counts.get(key, 0) + 1
            if state_counts[key] >= 3:
                return None  # Draw

            result = self.choose_turn(canonical, key, orig_to_canon, is_flipped)

            if result is None:
                # No beads available - this is a losing state for current player
                loser = current_player
                actual_winner = 1 - loser

                # Remove the bead that brought us here (loser's previous move)
                if loser in last_move:
                    prev_key, prev_turn = last_move[loser]
                    proven_loss_key = self.update_loss(prev_key, prev_turn)
                    if proven_loss_key:
                        self._mark_proven_loss(proven_loss_key)

                return actual_winner

            turn, canonical_turn = result

            # Check if current state is a proven win (forced win in >= 2 turns)
            state_info = self.states.get(key)
            if state_info and state_info.distance_to_win is not None and state_info.distance_to_win >= 2:
                # Proven win - end game early without playing it out
                actual_winner = current_player
                loser = 1 - actual_winner

                if loser in last_move:
                    prev_key, prev_turn = last_move[loser]
                    proven_loss_key = self.update_loss(prev_key, prev_turn)
                    if proven_loss_key:
                        self._mark_proven_loss(proven_loss_key)
                        if actual_winner in last_move:
                            winner_prev_key, _ = last_move[actual_winner]
                            self._check_winning_state(winner_prev_key)

                return actual_winner

            # Store canonical move
            last_move[current_player] = (key, canonical_turn)

            # Apply the turn
            new_state = state.apply_turn(turn)
            if new_state is None:
                # Invalid state (shouldn't happen)
                return None

            # Check for win BEFORE swapping (winner detection uses piece counts)
            winner = new_state.get_winner()
            if winner is not None:
                # winner is 0 if P0 pieces are intact (P1 lost), 1 if P1 pieces intact (P0 lost)
                # Map to actual player: after swaps, P0 in state = current_player's opponent
                # when current_player=0: no swaps, so winner maps directly
                # when current_player=1: one swap, so winner maps inversely
                if current_player == 0:
                    actual_winner = winner
                else:
                    actual_winner = 1 - winner
                loser = 1 - actual_winner

                # The loser's last move led to this - remove that bead
                if loser in last_move:
                    prev_key, prev_turn = last_move[loser]
                    proven_loss_key = self.update_loss(prev_key, prev_turn)
                    if proven_loss_key:
                        self._mark_proven_loss(proven_loss_key)
                        # Check if the state before that is now a proven win
                        if actual_winner in last_move:
                            winner_prev_key, _ = last_move[actual_winner]
                            self._check_winning_state(winner_prev_key)

                return actual_winner

            # Swap players for next turn
            state = new_state.swap_players()
            current_player = 1 - current_player

    def train(self, num_games: int, progress_interval: int = 0,
              start_state: Optional[GameState] = None) -> dict:
        """
        Run multiple training games.

        Args:
            num_games: Number of games to play.
            progress_interval: Print progress every N games (0 to disable).
            start_state: Optional starting state for all games.

        Returns:
            Statistics dict with win counts, draws, and state count.
        """
        wins = {0: 0, 1: 0}
        draws = 0

        for i in range(num_games):
            winner = self.train_game(start_state)
            if winner is None:
                draws += 1
            elif winner in wins:
                wins[winner] += 1

            if progress_interval > 0 and (i + 1) % progress_interval == 0:
                print(f"Games: {i + 1}, P0 wins: {wins[0]}, P1 wins: {wins[1]}, "
                      f"Draws: {draws}, States: {len(self.states)}")

        return {
            "games": num_games,
            "wins": wins,
            "draws": draws,
            "states": len(self.states),
        }

    def train_from_state(self, state: GameState, num_games: int) -> dict:
        """Train starting from a specific state."""
        return self.train(num_games, start_state=state)

    def get_state_info(self, state: GameState) -> dict:
        """Get information about a state for debugging."""
        canonical = state.canonical()
        key = canonical.to_key()
        info = self.get_or_create_state(canonical, key)
        return {
            "available_turns": len(info.beads),
            "is_proven_loss": info.is_proven_loss(),
            "distance_to_win": info.distance_to_win,
            "distance_to_loss": info.distance_to_loss,
        }


def main() -> None:
    """Train the AI from the command line."""
    import os
    import signal
    import sys
    from pathlib import Path
    from storage import save_ai, load_ai

    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    AI_FILE = DATA_DIR / "ai_state.pkl"

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <thousands_of_games>")
        print("Example: python ai_engine.py 10  # trains for 10,000 games")
        sys.exit(1)

    try:
        thousands = int(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid integer")
        sys.exit(1)

    num_games = thousands * 1000
    batch_size = 500
    stop_requested = False

    def handle_signal(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            print("\nForce quit.")
            sys.exit(1)
        print("\nStop requested, finishing current batch...")
        stop_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Load existing AI or create new one
    if os.path.exists(AI_FILE):
        ai = load_ai(AI_FILE)
        print(f"Loaded existing AI: {len(ai.states):,} states, "
              f"{ai.games_trained:,} games trained")
    else:
        ai = DuatAI()
        print("Starting new AI")

    print(f"Training for {num_games:,} games (Ctrl+C to stop safely)...")
    wins = {0: 0, 1: 0}
    draws = 0
    games_played = 0

    while games_played < num_games and not stop_requested:
        remaining = num_games - games_played
        batch = min(batch_size, remaining)

        for _ in range(batch):
            winner = ai.train_game()
            if winner is None:
                draws += 1
            elif winner in wins:
                wins[winner] += 1

        games_played += batch
        print(f"Games: {games_played:,}, P0 wins: {wins[0]:,}, P1 wins: {wins[1]:,}, "
              f"Draws: {draws:,}, States: {len(ai.states):,}")

    if stop_requested:
        print(f"\nStopped early after {games_played:,} games.")
    else:
        print("\nTraining complete!")

    print(f"  Games: {games_played:,}")
    print(f"  Player 0 wins: {wins[0]:,}")
    print(f"  Player 1 wins: {wins[1]:,}")
    print(f"  Draws: {draws:,}")
    print(f"  States learned: {len(ai.states):,}")

    DATA_DIR.mkdir(exist_ok=True)
    save_ai(ai, AI_FILE)
    print(f"Saved to {AI_FILE}")


if __name__ == "__main__":
    main()
