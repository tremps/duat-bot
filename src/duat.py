"""Duat game logic.

GameState always represents the position from P0's perspective (P0 to move).
After a turn, call swap_players() to get the opponent's view.
"""

from enum import IntEnum
from typing import Optional


class Direction(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def rotate_cw(self) -> "Direction":
        return Direction((self + 1) % 4)

    def rotate_ccw(self) -> "Direction":
        return Direction((self - 1) % 4)

    def opposite(self) -> "Direction":
        return Direction((self + 2) % 4)

    def delta(self) -> tuple[int, int]:
        """Returns (row_delta, col_delta) for moving in this direction."""
        return _DIRECTION_DELTAS[self]


_DIRECTION_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # NORTH, EAST, SOUTH, WEST


# Piece representation: (row, col, direction, player)
# player: 0 or 1
Piece = tuple[int, int, Direction, int]

# Move representation: (piece_index, action)
# action: 'F' = forward, 'CW' = rotate clockwise, 'CCW' = rotate counter-clockwise
Move = tuple[int, str]

# Turn: tuple of 1 or 3 moves
Turn = tuple[Move, ...]


# --- Compact int encoding for state keys and turns ---
# Action encoding: F=0, CW=1, CCW=2
_ACTION_TO_INT = {"F": 0, "CW": 1, "CCW": 2}
_INT_TO_ACTION = ("F", "CW", "CCW")


def encode_key(pieces: tuple) -> int:
    """Encode a pieces tuple into a single int (36 bits).

    Each piece uses 6 bits: row(2) + col(2) + dir(2).
    Player is implicit (pieces 0-2 = P0, 3-5 = P1 in canonical form).
    """
    val = 0
    for i, (r, c, d, _p) in enumerate(pieces):
        bits = (r << 4) | (c << 2) | int(d)
        val |= bits << (i * 6)
    return val


def decode_key(val: int) -> tuple:
    """Decode an int back to a pieces tuple."""
    pieces = []
    for i in range(6):
        bits = (val >> (i * 6)) & 0x3F
        r = (bits >> 4) & 0x3
        c = (bits >> 2) & 0x3
        d = Direction(bits & 0x3)
        p = 0 if i < 3 else 1
        pieces.append((r, c, d, p))
    return tuple(pieces)


def encode_turn(turn: Turn) -> int:
    """Encode a turn (1 or 3 moves) into a single int.

    Each move: piece_idx(3 bits) + action(2 bits) = 5 bits.
    1-move turn: bits 0-4 (values 0-17, most cached as small ints).
    3-move turn: bits 0-14 + bit 15 set as flag.
    """
    if len(turn) == 1:
        idx, action = turn[0]
        return (idx << 2) | _ACTION_TO_INT[action]
    # 3-move turn
    val = 0
    for i, (idx, action) in enumerate(turn):
        move_bits = (idx << 2) | _ACTION_TO_INT[action]
        val |= move_bits << (i * 5)
    return val | (1 << 15)


def decode_turn(val: int) -> Turn:
    """Decode an int back to a Turn tuple."""
    if val & (1 << 15):
        # 3-move turn
        moves = []
        for i in range(3):
            move_bits = (val >> (i * 5)) & 0x1F
            idx = move_bits >> 2
            action = _INT_TO_ACTION[move_bits & 0x3]
            moves.append((idx, action))
        return tuple(moves)
    # 1-move turn
    idx = val >> 2
    action = _INT_TO_ACTION[val & 0x3]
    return ((idx, action),)


def is_three_move_turn(val: int) -> bool:
    """Check if an encoded turn int represents a 3-move turn."""
    return bool(val & (1 << 15))


# Module-level cache for canonical keys: raw_int_key -> encoded (canonical_key, is_flipped)
# Encoded as (canonical_key << 1) | is_flipped.  Plain dict with periodic full clear.
_CANONICAL_CACHE_MAX = 500_000
_canonical_cache: dict[int, int] = {}


def _cache_put(key: int, canonical_key: int, is_flipped: bool):
    """Add entry to canonical cache with periodic full clear."""
    global _canonical_cache
    _canonical_cache[key] = (canonical_key << 1) | int(is_flipped)
    if len(_canonical_cache) > _CANONICAL_CACHE_MAX:
        _canonical_cache = {}


def clear_canonical_cache():
    """Clear the canonical key cache (for testing)."""
    global _canonical_cache
    _canonical_cache = {}


class GameState:
    """
    Immutable game state.

    Always represents P0's turn. After applying a turn, call swap_players()
    to get the state from the opponent's perspective.
    """

    def __init__(self, pieces: tuple[Piece, ...]):
        self.pieces = pieces
        self._winner: Optional[int] = None
        self._winner_checked = False

    @classmethod
    def initial(cls) -> "GameState":
        """Create the starting position (P0 to move)."""
        # V..V  (player 0, facing South)
        # ..V.
        # .^..  (player 1, facing North)
        # ^..^
        pieces = (
            (0, 0, Direction.SOUTH, 0),
            (0, 3, Direction.SOUTH, 0),
            (1, 2, Direction.SOUTH, 0),
            (2, 1, Direction.NORTH, 1),
            (3, 0, Direction.NORTH, 1),
            (3, 3, Direction.NORTH, 1),
        )
        return cls(pieces)

    def to_key(self) -> int:
        """Convert to compact int key for storage."""
        return encode_key(self.pieces)

    @classmethod
    def from_key(cls, key: int) -> "GameState":
        """Reconstruct from int key."""
        return cls(decode_key(key))

    def swap_players(self) -> "GameState":
        """
        Swap P0 and P1 pieces. O(n) where n=6.

        This simulates switching whose turn it is - after P0 moves,
        we swap so the opponent becomes the new P0.
        """
        return GameState(tuple((r, c, d, 1 - p) for r, c, d, p in self.pieces))

    def rotate_cw(self) -> "GameState":
        """Rotate the board 90 degrees clockwise. Preserves piece order."""
        new_pieces = []
        for row, col, direction, player in self.pieces:
            # 90 degrees CW: (r, c) -> (c, 3-r)
            new_row = col
            new_col = 3 - row
            new_dir = direction.rotate_cw()
            new_pieces.append((new_row, new_col, new_dir, player))
        return GameState(tuple(new_pieces))

    def flip_horizontal(self) -> "GameState":
        """Flip the board horizontally (mirror left-right). Preserves piece order."""
        new_pieces = []
        for row, col, direction, player in self.pieces:
            # Horizontal flip: (r, c) -> (r, 3-c)
            new_col = 3 - col
            # EAST <-> WEST, NORTH and SOUTH stay the same
            if direction == Direction.EAST:
                new_dir = Direction.WEST
            elif direction == Direction.WEST:
                new_dir = Direction.EAST
            else:
                new_dir = direction
            new_pieces.append((row, new_col, new_dir, player))
        return GameState(tuple(new_pieces))

    def _sort_pieces_with_mapping(self) -> tuple["GameState", dict[int, int]]:
        """
        Sort pieces within each player group for canonical ordering.
        Returns (sorted_state, old_to_new_index_mapping).
        """
        p0_with_idx = [(p, i) for i, p in enumerate(self.pieces) if p[3] == 0]
        p1_with_idx = [(p, i) for i, p in enumerate(self.pieces) if p[3] == 1]

        p0_sorted = sorted(p0_with_idx, key=lambda x: (x[0][0], x[0][1], x[0][2]))
        p1_sorted = sorted(p1_with_idx, key=lambda x: (x[0][0], x[0][1], x[0][2]))

        all_sorted = p0_sorted + p1_sorted
        new_pieces = tuple(p for p, _ in all_sorted)

        # old_idx -> new_idx
        mapping = {old_idx: new_idx for new_idx, (_, old_idx) in enumerate(all_sorted)}

        return GameState(new_pieces), mapping

    def canonical_with_mapping(self) -> tuple["GameState", dict[int, int], bool]:
        """
        Get canonical form, index mapping, and flip status.

        Returns (canonical_state, original_to_canonical_mapping, is_flipped).

        We try all 8 transforms (4 rotations × 2 flip states), sort each,
        and pick the minimum key. is_flipped indicates whether the canonical
        form used a horizontal flip (needed to swap CW/CCW in turns).
        """
        key = self.to_key()

        # Check cache - if hit, we know canonical and is_flipped, just need mapping
        cached = _canonical_cache.get(key)
        if cached is not None:
            canonical_key = cached >> 1
            is_flipped = bool(cached & 1)
            # Find the matching rotation to get the mapping
            base = self.flip_horizontal() if is_flipped else self
            for _ in range(4):
                sorted_state, sort_mapping = base._sort_pieces_with_mapping()
                if sorted_state.to_key() == canonical_key:
                    return sorted_state, sort_mapping, is_flipped
                base = base.rotate_cw()

        # Cache miss - compute all 8 and cache
        best_state = None
        best_key = None
        best_mapping = None
        best_flipped = False

        normal_keys = []
        flipped_keys = []

        # Try without flip (4 rotations)
        state = self
        for _ in range(4):
            sorted_state, sort_mapping = state._sort_pieces_with_mapping()
            k = sorted_state.to_key()
            normal_keys.append(k)

            if best_key is None or k < best_key:
                best_key = k
                best_state = sorted_state
                best_mapping = sort_mapping
                best_flipped = False

            state = state.rotate_cw()

        # Try with flip (4 rotations of flipped state)
        state = self.flip_horizontal()
        for _ in range(4):
            sorted_state, sort_mapping = state._sort_pieces_with_mapping()
            k = sorted_state.to_key()
            flipped_keys.append(k)

            if k < best_key:
                best_key = k
                best_state = sorted_state
                best_mapping = sort_mapping
                best_flipped = True

            state = state.rotate_cw()

        # Cache all variants
        for k in normal_keys:
            _cache_put(k, best_key, best_flipped)
        for k in flipped_keys:
            _cache_put(k, best_key, not best_flipped)

        return best_state, best_mapping, best_flipped

    def canonical(self) -> "GameState":
        """Get canonical form of this state (uses caching)."""
        key = self.to_key()

        # Check cache first
        cached = _canonical_cache.get(key)
        if cached is not None:
            canonical_key = cached >> 1
            return GameState.from_key(canonical_key)

        # Compute canonical and cache all 8 variants
        best_key = None
        best_flipped = False
        normal_keys = []  # 4 rotations without flip
        flipped_keys = []  # 4 rotations with flip

        # Without flip
        state = self
        for _ in range(4):
            sorted_state, _ = state._sort_pieces_with_mapping()
            k = sorted_state.to_key()
            normal_keys.append(k)

            if best_key is None or k < best_key:
                best_key = k
                best_flipped = False

            state = state.rotate_cw()

        # With flip
        state = self.flip_horizontal()
        for _ in range(4):
            sorted_state, _ = state._sort_pieces_with_mapping()
            k = sorted_state.to_key()
            flipped_keys.append(k)

            if k < best_key:
                best_key = k
                best_flipped = True

            state = state.rotate_cw()

        # Cache all variants -> encoded (canonical, is_flipped)
        for k in normal_keys:
            _cache_put(k, best_key, best_flipped)
        for k in flipped_keys:
            _cache_put(k, best_key, not best_flipped)

        return GameState.from_key(best_key)

    def get_player_pieces(self, player: int) -> list[int]:
        """Get indices of pieces belonging to player."""
        return [i for i, p in enumerate(self.pieces) if p[3] == player]

    def get_piece_at(self, row: int, col: int) -> Optional[int]:
        """Get index of piece at position, or None."""
        for i, p in enumerate(self.pieces):
            if p[0] == row and p[1] == col:
                return i
        return None

    def apply_move(self, move: Move) -> Optional["GameState"]:
        """
        Apply a single move. Returns new state or None if illegal.
        Does not switch player - that happens via swap_players() after a full turn.
        P0 is always the moving player.
        """
        piece_idx, action = move
        if piece_idx >= len(self.pieces):
            return None

        piece = self.pieces[piece_idx]
        row, col, direction, player = piece

        # Can only move P0's pieces (current player)
        if player != 0:
            return None

        pieces = list(self.pieces)

        if action == "CW":
            pieces[piece_idx] = (row, col, direction.rotate_cw(), player)
            return GameState(tuple(pieces))

        elif action == "CCW":
            pieces[piece_idx] = (row, col, direction.rotate_ccw(), player)
            return GameState(tuple(pieces))

        elif action == "F":
            dr, dc = direction.delta()
            new_row, new_col = row + dr, col + dc

            # Check bounds - can't move yourself off board
            if not (0 <= new_row < 4 and 0 <= new_col < 4):
                return None

            # Find all pieces in the push chain
            chain = []  # List of piece indices to push
            check_row, check_col = new_row, new_col

            while True:
                idx = self.get_piece_at(check_row, check_col)
                if idx is None:
                    break  # Empty square, chain ends here

                p = pieces[idx]
                # Can't push piece facing you
                if p[2] == direction.opposite():
                    return None

                chain.append(idx)
                check_row += dr
                check_col += dc

            # Push all pieces in chain (in reverse order to avoid conflicts)
            eliminated = None
            for idx in reversed(chain):
                p = pieces[idx]
                p_new_row, p_new_col = p[0] + dr, p[1] + dc

                if not (0 <= p_new_row < 4 and 0 <= p_new_col < 4):
                    # This piece is pushed off the board
                    eliminated = idx
                else:
                    pieces[idx] = (p_new_row, p_new_col, p[2], p[3])

            # Remove eliminated piece if any
            if eliminated is not None:
                pieces = [p for i, p in enumerate(pieces) if i != eliminated]

            # Move our piece - find it in the potentially modified list
            for i, p in enumerate(pieces):
                if p[0] == row and p[1] == col and p[3] == player:
                    pieces[i] = (new_row, new_col, direction, player)
                    break

            return GameState(tuple(pieces))

        return None

    def apply_turn(self, turn: Turn) -> Optional["GameState"]:
        """
        Apply a full turn (1 or 3 moves). Returns new state.
        Does NOT swap players - caller should call swap_players() after if needed.
        """
        state = self
        for move in turn:
            state = state.apply_move(move)
            if state is None:
                return None
            # Check for win mid-turn
            if state.get_winner() is not None:
                return state

        return state

    def get_winner(self) -> Optional[int]:
        """
        Check if someone won. Returns player (0 or 1) or None.

        P0 wins if P1 has fewer than 3 pieces (P0 pushed one off).
        P1 wins if P0 has fewer than 3 pieces (shouldn't happen if P0 is moving).
        """
        if self._winner_checked:
            return self._winner

        p0_count = sum(1 for p in self.pieces if p[3] == 0)
        p1_count = sum(1 for p in self.pieces if p[3] == 1)

        if p1_count < 3:
            self._winner = 0  # P0 wins (pushed P1's piece off)
        elif p0_count < 3:
            self._winner = 1  # P1 wins (shouldn't happen during P0's turn)
        else:
            self._winner = None

        self._winner_checked = True
        return self._winner

    def get_legal_single_moves(self) -> list[Move]:
        """Get all legal single moves for P0 (current player)."""
        moves = []
        for idx in self.get_player_pieces(0):  # Always P0
            # Rotations are always legal
            moves.append((idx, "CW"))
            moves.append((idx, "CCW"))
            # Forward needs validation (bounds, push rules)
            if self.apply_move((idx, "F")) is not None:
                moves.append((idx, "F"))
        return moves

    def get_legal_turns(self) -> list[Turn]:
        """
        Get all legal turns (1 move or 3 moves).

        Turns that lead to the same resulting state (considering symmetry)
        are deduplicated, keeping the shorter turn.

        Terminal states (where someone won) are NOT canonicalized - they use
        a simple key since we don't need to store or query them.
        """
        # Map from resulting state key -> (turn, move_count)
        # We keep the turn with fewer moves for each resulting state
        result_to_turn: dict = {}

        # 1-move turns (process first so they get priority)
        for m in self.get_legal_single_moves():
            turn = (m,)
            result = self.apply_turn(turn)
            if result is not None:
                winner = result.get_winner()
                if winner is not None:
                    # Terminal state - don't canonicalize, use simple key
                    # All winning moves are equivalent, keep just one
                    key = ("terminal", winner)
                else:
                    # Swap players to get opponent's view, then canonicalize
                    key = result.swap_players().canonical().to_key()
                # 1-move turns always win over 3-move turns
                if key not in result_to_turn or len(result_to_turn[key]) > 1:
                    result_to_turn[key] = turn

        # 3-move turns
        for m1 in self.get_legal_single_moves():
            s1 = self.apply_move(m1)
            if s1 is None or s1.get_winner() is not None:
                continue
            for m2 in s1.get_legal_single_moves():
                s2 = s1.apply_move(m2)
                if s2 is None or s2.get_winner() is not None:
                    continue
                for m3 in s2.get_legal_single_moves():
                    s3 = s2.apply_move(m3)
                    if s3 is not None:
                        turn = (m1, m2, m3)
                        winner = s3.get_winner()
                        if winner is not None:
                            # Terminal state - don't canonicalize
                            key = ("terminal", winner)
                        else:
                            # Swap players to get opponent's view, then canonicalize
                            key = s3.swap_players().canonical().to_key()
                        # Only add if no turn exists for this result state
                        if key not in result_to_turn:
                            result_to_turn[key] = turn

        return list(result_to_turn.values())

    def find_winning_turn(self) -> Optional[Turn]:
        """
        Find a turn that wins immediately for P0, or None.

        Prefers 1-move turns over 3-move turns. Short-circuits on first win found.
        Does NOT canonicalize any resulting states (efficient for win detection).
        """
        # Check 1-move turns first (prefer shorter turns)
        for move in self.get_legal_single_moves():
            turn = (move,)
            result = self.apply_turn(turn)
            if result and result.get_winner() == 0:
                return turn

        # Check 3-move turns
        for m1 in self.get_legal_single_moves():
            s1 = self.apply_move(m1)
            if s1 is None or s1.get_winner() is not None:
                continue
            for m2 in s1.get_legal_single_moves():
                s2 = s1.apply_move(m2)
                if s2 is None or s2.get_winner() is not None:
                    continue
                for m3 in s2.get_legal_single_moves():
                    s3 = s2.apply_move(m3)
                    if s3 and s3.get_winner() == 0:
                        return (m1, m2, m3)

        return None

    def display(self, current_player: int = 0) -> str:
        """Return string representation of the board."""
        symbols = {
            (0, Direction.NORTH): "A",
            (0, Direction.EAST): ">",
            (0, Direction.SOUTH): "V",
            (0, Direction.WEST): "<",
            (1, Direction.NORTH): "a",
            (1, Direction.EAST): ")",
            (1, Direction.SOUTH): "v",
            (1, Direction.WEST): "(",
        }
        grid = [["." for _ in range(4)] for _ in range(4)]
        for row, col, direction, player in self.pieces:
            grid[row][col] = symbols[(player, direction)]

        lines = ["".join(row) for row in grid]
        lines.append(f"Player {current_player}'s turn")
        return "\n".join(lines)


# --- Compact human-readable state string ---
# Format: "S..S/..S./.n../n..n w"
# 4 rows separated by '/', each row 4 chars.
# White (P0 when current_player==0): N E S W (uppercase)
# Black (P1 when current_player==0): n e s w (lowercase)
# '.' = empty. Space then 'w' or 'b' = whose turn.

_DIR_TO_WHITE = {Direction.NORTH: 'N', Direction.EAST: 'E',
                 Direction.SOUTH: 'S', Direction.WEST: 'W'}
_DIR_TO_BLACK = {Direction.NORTH: 'n', Direction.EAST: 'e',
                 Direction.SOUTH: 's', Direction.WEST: 'w'}
_CHAR_TO_DIR = {'N': Direction.NORTH, 'E': Direction.EAST,
                'S': Direction.SOUTH, 'W': Direction.WEST,
                'n': Direction.NORTH, 'e': Direction.EAST,
                's': Direction.SOUTH, 'w': Direction.WEST}


def state_to_string(state: GameState, current_player: int) -> str:
    """Encode a game state as a compact human-readable string.

    When current_player == 0, P0 in state = white (uppercase).
    When current_player == 1, P0 in state = black (lowercase).
    """
    grid = [['.' for _ in range(4)] for _ in range(4)]
    for row, col, direction, player in state.pieces:
        # Map internal player to display color
        if current_player == 0:
            actual = player  # P0=white, P1=black
        else:
            actual = 1 - player  # P0=black, P1=white
        if actual == 0:
            grid[row][col] = _DIR_TO_WHITE[direction]
        else:
            grid[row][col] = _DIR_TO_BLACK[direction]
    rows = '/'.join(''.join(r) for r in grid)
    turn = 'w' if current_player == 0 else 'b'
    return f'{rows} {turn}'


def string_to_state(s: str) -> tuple[GameState, int]:
    """Decode a state string back to (GameState, current_player).

    Returns the state with current_player as P0.
    """
    s = s.strip()
    parts = s.rsplit(' ', 1)
    if len(parts) != 2 or parts[1] not in ('w', 'b'):
        raise ValueError(f"Invalid state string: {s!r}")
    board_str, turn_char = parts
    current_player = 0 if turn_char == 'w' else 1

    rows = board_str.split('/')
    if len(rows) != 4:
        raise ValueError(f"Expected 4 rows, got {len(rows)}")

    pieces = []
    for r, row in enumerate(rows):
        if len(row) != 4:
            raise ValueError(f"Row {r} has {len(row)} chars, expected 4")
        for c, ch in enumerate(row):
            if ch == '.':
                continue
            if ch not in _CHAR_TO_DIR:
                raise ValueError(f"Unknown piece char: {ch!r}")
            direction = _CHAR_TO_DIR[ch]
            is_white = ch.isupper()
            # Map display color to internal player
            if current_player == 0:
                player = 0 if is_white else 1  # white=P0, black=P1
            else:
                player = 1 if is_white else 0  # white=P1, black=P0
            pieces.append((r, c, direction, player))

    return GameState(tuple(pieces)), current_player
