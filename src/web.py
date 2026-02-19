"""Web UI for Duat game using NiceGUI."""

import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
from nicegui import ui, app

from duat import GameState, Direction, Turn, state_to_string, string_to_state

# AI Server configuration
AI_SERVER_URL = os.environ.get("AI_SERVER_URL", "http://localhost:8000")

# Static files
STATIC_DIR = Path(__file__).parent.parent / 'static'
STATIC_URL = '/static'
LOGO_URL = f'{STATIC_URL}/logo.avif'

app.add_static_files(STATIC_URL, str(STATIC_DIR))

# Timing constants (in seconds)
MOVE_DELAY = 0.5      # Delay between moves within a turn
TURN_DELAY = 0.75     # Delay between turns

# Flags for graceful shutdown and AI turns
_shutting_down = False
_ai_turn_in_progress = False
_auto_playing = False
_stop_auto_play = False

# Animation state
_prev_grid: dict = {}   # {(display_row, col): (actual_player, game_direction)}
_animate_next = False
CELL_STEP = 124  # 120px cell + 4px (gap-1)

# Analysis state
_analysis_result: Optional[dict] = None
_analysis_state_key: Optional[str] = None
_analyzing = False
_analysis_elapsed: Optional[float] = None
_ai_move_elapsed: Optional[float] = None


def _build_grid() -> dict:
    """Build map of current piece display positions."""
    grid = {}
    for piece in game.state.pieces:
        row, col, direction, player = piece
        display_row = game.get_display_row(row)
        actual_player = player if game.current_player == 0 else (1 - player)
        grid[(display_row, col)] = (actual_player, direction)
    return grid


def _compute_animations(new_grid: dict) -> tuple:
    """Compute CSS animation styles by diffing prev and new grids.

    Each call covers a single game move — either a forward (slide + push)
    or a rotation, never both. Returns (animations_dict, ghosts_list).
    Ghosts are eliminated pieces that should slide off the board edge.
    """
    empty = ({}, [])
    if not _prev_grid:
        return empty

    # Positions where the occupant changed (different player, or appeared/vanished)
    changed = set()
    for pos in set(_prev_grid) | set(new_grid):
        if _prev_grid.get(pos) != new_grid.get(pos):
            changed.add(pos)

    if not changed:
        return empty

    # Detect forward move: find the cell that was occupied and is now empty
    source = None
    for pos in changed:
        if pos in _prev_grid and pos not in new_grid:
            source = pos
            break

    if source is not None:
        # Use the piece's facing direction to determine slide direction
        _, old_dir = _prev_grid[source]
        _dir_display = {
            Direction.NORTH: (1, 0),
            Direction.SOUTH: (-1, 0),
            Direction.EAST: (0, 1),
            Direction.WEST: (0, -1),
        }
        norm_dr, norm_dc = _dir_display[old_dir]

        # Walk from source along direction through prev_grid to find full chain
        chain = [source]
        r, c = source[0] + norm_dr, source[1] + norm_dc
        while 0 <= r <= 3 and 0 <= c <= 3 and (r, c) in _prev_grid:
            chain.append((r, c))
            r += norm_dr
            c += norm_dc

        last = chain[-1]
        off_r = last[0] + norm_dr
        off_c = last[1] + norm_dc
        pushed_off = not (0 <= off_r <= 3 and 0 <= off_c <= 3)

        dx = -norm_dc * CELL_STEP
        dy = -norm_dr * CELL_STEP
        slide = f'animation: piece-slide 0.3s ease-out; --slide-x: {dx}px; --slide-y: {dy}px'

        # Animate pieces that slid into new positions
        animations = {}
        for i in range(1, len(chain)):
            pos = chain[i]
            if pos in new_grid:
                animations[pos] = slide
        # If last piece wasn't pushed off, it slid one step further
        if not pushed_off and (off_r, off_c) in new_grid:
            animations[(off_r, off_c)] = slide

        # Ghost for the piece pushed off the board
        ghosts = []
        if pushed_off:
            ghost_piece = _prev_grid.get(last)
            if ghost_piece:
                gdx = norm_dc * CELL_STEP
                gdy = norm_dr * CELL_STEP
                ghost_style = (
                    f'animation: piece-slide-off 0.3s ease-out forwards;'
                    f' --off-x: {gdx}px; --off-y: {gdy}px'
                )
                ghosts.append((last[0], last[1], ghost_piece[0], ghost_piece[1], ghost_style))

        return animations, ghosts

    # No slide detected — must be a rotation
    animations = {}
    for pos in changed:
        old = _prev_grid.get(pos)
        new = new_grid.get(pos)
        if old and new and old[0] == new[0] and old[1] != new[1]:
            if new[1] == old[1].rotate_cw():
                animations[pos] = 'animation: piece-spin-ccw 0.3s ease-out'
            elif new[1] == old[1].rotate_ccw():
                animations[pos] = 'animation: piece-spin-cw 0.3s ease-out'
    return animations, []


# Inline action button positions (CSS absolute positioning inside piece)
_BUTTON_POS = {
    'top': 'top: 10px; left: 50%; transform: translateX(-50%)',
    'right': 'top: 50%; right: 10px; transform: translateY(-50%)',
    'bottom': 'bottom: 10px; left: 50%; transform: translateX(-50%)',
    'left': 'top: 50%; left: 10px; transform: translateY(-50%)',
}

# For each game Direction: (position, symbol, game_action)
# Display is vertically flipped so CW/CCW swap.
# Rotation buttons only (forward is the center arrow).
# Positions follow the "destination" rule: button is on the side
# the piece will face after rotating that way.
_PIECE_ACTIONS = {
    Direction.NORTH: [  # display ↓
        ('left', '\u21ba', 'CCW'),
        ('right', '\u21bb', 'CW'),
    ],
    Direction.SOUTH: [  # display ↑
        ('right', '\u21ba', 'CCW'),
        ('left', '\u21bb', 'CW'),
    ],
    Direction.EAST: [  # display →
        ('bottom', '\u21ba', 'CCW'),
        ('top', '\u21bb', 'CW'),
    ],
    Direction.WEST: [  # display ←
        ('top', '\u21ba', 'CCW'),
        ('bottom', '\u21bb', 'CW'),
    ],
}


def _serialize_state_key(state: GameState) -> list:
    """Convert GameState to JSON-serializable format for API calls."""
    return [[r, c, int(d), p] for r, c, d, p in state.pieces]


def _deserialize_turn(turn_data: list) -> Optional[Turn]:
    """Convert API response turn to Turn type."""
    if turn_data is None:
        return None
    return tuple((piece_idx, action) for piece_idx, action in turn_data)


async def get_ai_move(state: GameState) -> Optional[Turn]:
    """Query AI server for best move."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            state_key = _serialize_state_key(state)
            resp = await client.post(
                f"{AI_SERVER_URL}/best_move",
                json={"state_key": state_key}
            )
            resp.raise_for_status()
            data = resp.json()
            return _deserialize_turn(data.get("turn"))
    except Exception as e:
        print(f"Error getting AI move: {e}")
        return None


async def analyze_state(state: GameState) -> Optional[dict]:
    """Query AI server to analyze a state."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            state_key = _serialize_state_key(state)
            resp = await client.post(
                f"{AI_SERVER_URL}/analyze",
                json={"state_key": state_key}
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"Error analyzing state: {e}")
        return None


async def get_ai_stats() -> dict:
    """Get AI server statistics."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(f"{AI_SERVER_URL}/stats")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"Error getting AI stats: {e}")
        return {
            "states": 0,
            "games_trained": 0,
            "file_size_bytes": 0,
            "white_win_rate": None,
            "win_rate_sample_size": 0,
            "games_per_second": 0,
            "proven_wins": 0,
            "proven_losses": 0,
        }


class DuatGame:
    """Game state and UI controller."""

    def __init__(self):
        self.state: GameState = GameState.initial()
        self.selected_piece: Optional[int] = None
        self.current_turn_moves: list = []
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.undo_stack: list = []
        self.current_player: int = 0
        self.history: list[str] = []
        self._add_to_history()

    def _add_to_history(self):
        """Append current state string to history."""
        s = state_to_string(self.state, self.current_player)
        self.history.append(s)

    def new_game(self):
        """Reset game state. History is preserved (use clear_history to wipe it)."""
        self.state = GameState.initial()
        self.selected_piece = None
        self.current_turn_moves = []
        self.game_over = False
        self.winner = None
        self.undo_stack = []
        self.current_player = 0
        self._add_to_history()

    def clear_history(self):
        """Clear the history list."""
        self.history = []

    def get_display_row(self, game_row: int) -> int:
        """Convert game row to display row (flipped so White/P0 is at bottom)."""
        return 3 - game_row

    def get_game_row(self, display_row: int) -> int:
        """Convert display row to game row."""
        return 3 - display_row

    def get_piece_at_display(self, display_row: int, col: int) -> Optional[tuple]:
        """Get piece at display coordinates. Returns (piece_idx, piece) or None."""
        game_row = self.get_game_row(display_row)
        for idx, piece in enumerate(self.state.pieces):
            if piece[0] == game_row and piece[1] == col:
                return (idx, piece)
        return None

    def get_direction_arrow(self, direction: Direction) -> str:
        """Get arrow character for direction (flipped for display)."""
        arrows = {
            Direction.NORTH: "\u2193",
            Direction.SOUTH: "\u2191",
            Direction.EAST: "\u2192",
            Direction.WEST: "\u2190",
        }
        return arrows[direction]

    def get_current_player_name(self) -> str:
        """Get display name for current player."""
        return "White" if self.current_player == 0 else "Black"

    def is_mid_turn(self) -> bool:
        """Returns True if there are partial moves in the current turn."""
        return len(self.current_turn_moves) > 0

    def can_make_move(self, action: str) -> bool:
        """Check if the selected piece can make this move."""
        if self.selected_piece is None:
            return False
        if len(self.current_turn_moves) >= 3:
            return False
        move = (self.selected_piece, action)
        return self.state.apply_move(move) is not None

    def make_move(self, action: str) -> bool:
        """Make a move with the selected piece. Returns True if successful."""
        if not self.can_make_move(action):
            return False

        self.undo_stack.append((self.state, self.selected_piece))

        move = (self.selected_piece, action)
        new_state = self.state.apply_move(move)
        if new_state is None:
            self.undo_stack.pop()
            return False

        self.current_turn_moves.append(move)
        self.state = new_state

        # Check for win mid-turn
        winner = self.state.get_winner()
        if winner is not None:
            self.game_over = True
            if self.current_player == 0:
                self.winner = winner
            else:
                self.winner = 1 - winner
            self.selected_piece = None
            self.current_turn_moves = []
            self.undo_stack = []
            self._add_to_history()
            return True

        return True

    def undo_move(self) -> bool:
        """Undo the last move. Returns True if successful."""
        if not self.undo_stack or not self.current_turn_moves:
            return False
        self.state, self.selected_piece = self.undo_stack.pop()
        self.current_turn_moves.pop()
        return True

    def can_submit_turn(self) -> bool:
        """Check if the current turn can be submitted."""
        return len(self.current_turn_moves) in [1, 3] and not self.game_over

    def submit_turn(self):
        """Submit the current turn."""
        if not self.can_submit_turn():
            return
        self.state = self.state.swap_players()
        self.current_player = 1 - self.current_player
        self.current_turn_moves = []
        self.selected_piece = None
        self.undo_stack = []
        self._add_to_history()

    def get_state_for_ai(self) -> GameState:
        """Get state in format suitable for AI query."""
        return self.state

    def apply_ai_turn(self, turn: Turn) -> bool:
        """Apply an AI turn. Returns True if game continues."""
        for move in turn:
            new_state = self.state.apply_move(move)
            if new_state is None:
                return False

            self.state = new_state

            winner = self.state.get_winner()
            if winner is not None:
                self.game_over = True
                if self.current_player == 0:
                    self.winner = winner
                else:
                    self.winner = 1 - winner
                self.selected_piece = None
                self._add_to_history()
                return False

        return True

    def finish_ai_turn(self):
        """Switch player after AI turn completes."""
        if not self.game_over:
            self.state = self.state.swap_players()
            self.current_player = 1 - self.current_player
            self._add_to_history()
        self.selected_piece = None

    def load_state(self, state: GameState, current_player: int):
        """Load a state directly, clearing partial moves."""
        self.state = state
        self.current_player = current_player
        self.selected_piece = None
        self.current_turn_moves = []
        self.undo_stack = []
        self.game_over = False
        self.winner = None
        # Check if this state is already game over
        winner = self.state.get_winner()
        if winner is not None:
            self.game_over = True
            if self.current_player == 0:
                self.winner = winner
            else:
                self.winner = 1 - winner

    def load_state_string(self, s: str):
        """Parse a state string and load it."""
        state, current_player = string_to_state(s)
        self.load_state(state, current_player)
        self._add_to_history()

    def load_history_entry(self, index: int):
        """Load a history entry without adding a duplicate."""
        if 0 <= index < len(self.history):
            s = self.history[index]
            state, current_player = string_to_state(s)
            self.load_state(state, current_player)


# Global game instance
game = DuatGame()


@ui.refreshable
def game_board():
    """Render the game board with inline piece controls."""
    global _prev_grid, _animate_next
    controls_active = not game.game_over and not _ai_turn_in_progress
    at_limit = len(game.current_turn_moves) >= 3

    # Compute animations by diffing current vs previous grid
    cur_grid = _build_grid()
    if _animate_next:
        animations, ghosts = _compute_animations(cur_grid)
        _animate_next = False
    else:
        animations, ghosts = {}, []
    _prev_grid = cur_grid

    with ui.element('div').classes('relative overflow-visible'):
        # Board layer — just the colored squares
        with ui.element('div').classes('grid grid-cols-4 gap-1 p-4 bg-amber-800 rounded-lg'):
            for _ in range(16):
                ui.element('div').classes('rounded bg-amber-200').style('width: 120px; height: 120px')

        # Piece layer — overlaid on top so pieces are always in front of the board
        with ui.element('div').classes('grid grid-cols-4 gap-1 p-4 absolute inset-0 pointer-events-none'):
            for display_row in range(4):
                for col in range(4):
                    piece_info = game.get_piece_at_display(display_row, col)
                    piece_idx = None
                    if piece_info:
                        piece_idx, _ = piece_info

                    with ui.element('div').classes('relative overflow-visible').style('width: 120px; height: 120px'):
                        if piece_info:
                            _, piece = piece_info
                            row, col_pos, direction, player = piece

                            actual_player = player if game.current_player == 0 else (1 - player)

                            if actual_player == 0:
                                bg = 'bg-gray-100 border-2 border-gray-400'
                                color = 'text-gray-800'
                            else:
                                bg = 'bg-gray-800'
                                color = 'text-gray-100'
                            arrow = game.get_direction_arrow(direction)

                            # Animation wrapper — piece + buttons animate together
                            anim_style = animations.get((display_row, col), '')
                            with ui.element('div').classes('absolute inset-0 flex items-center justify-center').style(anim_style):

                                # Piece circle — doubles as forward button for P0
                                can_fwd = (player == 0 and controls_active
                                           and not at_limit
                                           and game.state.apply_move((piece_idx, 'F')) is not None)
                                piece_cls = f'w-24 h-24 rounded-full {bg} flex items-center justify-center {color} pointer-events-auto'
                                if can_fwd:
                                    piece_cls += ' cursor-pointer group'

                                    def make_fwd_handler(pidx=piece_idx):
                                        def handler():
                                            global _animate_next
                                            if game.game_over or _ai_turn_in_progress:
                                                return
                                            game.selected_piece = pidx
                                            game.make_move('F')
                                            _animate_next = True
                                            refresh_all()
                                        return handler

                                    with ui.element('div').classes(piece_cls).on('click.stop', make_fwd_handler()):
                                        ui.label(arrow).classes('text-4xl font-bold pointer-events-none group-hover:scale-125 transition-transform duration-150')
                                else:
                                    with ui.element('div').classes(piece_cls):
                                        ui.label(arrow).classes('text-4xl font-bold pointer-events-none')

                                # Rotation buttons for P0 pieces
                                if player == 0 and controls_active:
                                    for pos_key, symbol, action in _PIECE_ACTIONS[direction]:
                                        can_do = not at_limit

                                        btn_cls = f'absolute w-9 h-9 rounded-full flex items-center justify-center {color} group pointer-events-auto '
                                        if can_do:
                                            btn_cls += 'cursor-pointer'
                                        else:
                                            btn_cls += 'opacity-[0.12] pointer-events-none'

                                        def make_action_handler(pidx=piece_idx, act=action):
                                            def handler():
                                                global _animate_next
                                                if game.game_over or _ai_turn_in_progress:
                                                    return
                                                game.selected_piece = pidx
                                                game.make_move(act)
                                                _animate_next = True
                                                refresh_all()
                                            return handler

                                        with ui.element('div').classes(btn_cls).style(_BUTTON_POS[pos_key]).on('click.stop', make_action_handler()):
                                            ui.label(symbol).classes('text-xl leading-none pointer-events-none font-bold group-hover:scale-125 transition-transform duration-150').style('transform: scaleY(-1)')

        # Ghost layer — eliminated pieces that slide off the board edge
        for ghost_row, ghost_col, ghost_player, ghost_dir, ghost_style in ghosts:
            ghost_x = 16 + ghost_col * CELL_STEP
            ghost_y = 16 + ghost_row * CELL_STEP
            if ghost_player == 0:
                ghost_bg = 'bg-gray-100 border-2 border-gray-400'
                ghost_color = 'text-gray-800'
            else:
                ghost_bg = 'bg-gray-800'
                ghost_color = 'text-gray-100'
            ghost_arrow = game.get_direction_arrow(ghost_dir)
            ghost_pos = f'position: absolute; left: {ghost_x}px; top: {ghost_y}px; width: 120px; height: 120px;'
            with ui.element('div').style(ghost_pos + ghost_style).classes('flex items-center justify-center pointer-events-none'):
                with ui.element('div').classes(f'w-24 h-24 rounded-full {ghost_bg} flex items-center justify-center {ghost_color}'):
                    ui.label(ghost_arrow).classes('text-4xl font-bold')


@ui.refreshable
def game_status():
    """Show current game status with analyze button."""
    with ui.card().classes('px-3 py-2 mb-2'):
        # Turn indicator
        if game.game_over:
            winner_name = "White" if game.winner == 0 else "Black"
            ui.label(f'{winner_name} wins!').classes('text-lg font-bold text-green-600')
        else:
            player_name = game.get_current_player_name()
            moves = len(game.current_turn_moves)
            if moves > 0:
                ui.label(f"{player_name}'s Turn ({moves} move{'s' if moves != 1 else ''})").classes('text-lg font-bold')
            else:
                ui.label(f"{player_name}'s Turn").classes('text-lg font-bold')

        # Analyze
        with ui.row().classes('items-center gap-2 mt-1'):
            async def on_analyze():
                global _analysis_result, _analysis_state_key, _analyzing, _analysis_elapsed
                _analyzing = True
                game_status.refresh()
                state = game.get_state_for_ai()
                _analysis_state_key = state_to_string(state, game.current_player)
                start = time.monotonic()
                result = await analyze_state(state)
                _analysis_elapsed = time.monotonic() - start
                _analysis_result = result
                _analyzing = False
                game_status.refresh()

            analyze_btn = ui.button('Analyze', on_click=on_analyze).props('dense size=sm').classes('bg-teal-500')
            if _analyzing or _ai_turn_in_progress:
                analyze_btn.disable()

            # Show result if available and still matches current state
            current_key = state_to_string(game.state, game.current_player)
            if _analysis_result is not None and _analysis_state_key == current_key:
                dtw = _analysis_result.get("distance_to_win")
                dtl = _analysis_result.get("distance_to_loss")
                turns = _analysis_result.get("available_turns", 0)
                time_str = f' ({_analysis_elapsed:.1f}s)' if _analysis_elapsed is not None else ''
                if dtw is not None:
                    ui.label(f'Win in {dtw}{time_str}').classes('text-sm font-bold text-green-600')
                elif dtl is not None:
                    ui.label(f'Loss in {dtl}{time_str}').classes('text-sm font-bold text-red-600')
                else:
                    ui.label(f'Unknown ({turns} beads){time_str}').classes('text-sm font-bold text-gray-500')
            elif _analyzing:
                ui.spinner(size='sm')


@ui.refreshable
def game_controls_bottom():
    """Game control buttons."""
    move_count = len(game.current_turn_moves)
    ai_disabled = game.is_mid_turn() or _ai_turn_in_progress or game.game_over

    with ui.card().classes('px-3 py-2 w-full'):
        with ui.row().classes('items-center justify-between w-full mb-1'):
            ui.label('Controls').classes('text-sm font-semibold text-gray-700')
            if _ai_move_elapsed is not None:
                ui.label(f'AI: {_ai_move_elapsed:.1f}s').classes('text-xs text-gray-500')

        with ui.row().classes('gap-1 flex-wrap'):
            undo_btn = ui.button('Undo', on_click=lambda: (game.undo_move(), refresh_all())).props('dense').classes('bg-yellow-500')
            if move_count == 0 or _ai_turn_in_progress:
                undo_btn.disable()

            can_submit = game.can_submit_turn()
            submit_text = f'End Turn ({move_count})'
            submit_btn = ui.button(submit_text, on_click=lambda: (game.submit_turn(), refresh_all())).props('dense').classes('bg-orange-500')
            if not can_submit or _ai_turn_in_progress:
                submit_btn.disable()

            ai_btn = ui.button('AI Move', on_click=do_ai_move).props('dense').classes('bg-blue-500')
            if ai_disabled:
                ai_btn.disable()

            if _auto_playing:
                ui.button('Stop', on_click=stop_auto_play_fn).props('dense').classes('bg-red-500')
            else:
                auto_btn = ui.button('Auto Play', on_click=do_auto_play).props('dense').classes('bg-purple-500')
                if ai_disabled:
                    auto_btn.disable()

            new_btn = ui.button('New Game', on_click=lambda: (game.new_game(), refresh_all())).props('dense').classes('bg-green-500')
            if _ai_turn_in_progress:
                new_btn.disable()


@ui.refreshable
def history_panel():
    """State history sidebar with paste input and clickable entries."""
    with ui.row().classes('w-full items-center justify-between mb-1'):
        ui.label('Game States').classes('text-sm font-semibold text-gray-700')
        ui.button('Clear', on_click=lambda: (game.clear_history(), refresh_all())).props('flat dense size=sm').classes('text-gray-500')

    # Paste input + Load button
    with ui.row().classes('w-full gap-1 mb-1'):
        paste_input = ui.input(placeholder='Paste state string...').classes('flex-grow').props('dense')

        def do_load():
            val = paste_input.value
            if val and val.strip():
                try:
                    game.load_state_string(val.strip())
                    paste_input.value = ''
                    refresh_all()
                except ValueError as e:
                    ui.notify(str(e), type='negative')

        load_btn = ui.button('Load', on_click=do_load).classes('bg-blue-500').props('dense')
        if _ai_turn_in_progress:
            load_btn.disable()

    # Scrollable history list (newest first)
    with ui.scroll_area().classes('w-full flex-grow'):
        for i in range(len(game.history) - 1, -1, -1):
            entry = game.history[i]
            with ui.row().classes('w-full items-center gap-0 py-0'):
                idx = i

                def make_load_handler(idx=idx):
                    def handler():
                        if not _ai_turn_in_progress:
                            game.load_history_entry(idx)
                            refresh_all()
                    return handler

                ui.button(
                    entry,
                    on_click=make_load_handler(),
                ).props('flat dense no-caps').classes('text-xs font-mono text-left flex-grow')

                entry_str = entry

                def make_copy_handler(s=entry_str):
                    async def handler():
                        escaped = s.replace('\\', '\\\\').replace("'", "\\'")
                        await ui.run_javascript(f"navigator.clipboard.writeText('{escaped}')")
                        ui.notify('Copied!', type='positive', position='bottom', timeout=1000)
                    return handler

                ui.button(
                    icon='content_copy',
                    on_click=make_copy_handler(),
                ).props('flat dense round size=sm').classes('text-gray-500')


@ui.refreshable
def ai_stats_panel():
    """Display AI server statistics."""
    async def update_stats():
        stats = await get_ai_stats()
        states_count = stats.get('states', 0)
        games_trained = stats.get('games_trained', 0)
        file_size = stats.get('file_size_bytes', 0)
        white_win_rate = stats.get('white_win_rate')
        win_rate_sample_size = stats.get('win_rate_sample_size', 0)
        games_per_second = stats.get('games_per_second', 0)
        proven_wins = stats.get('proven_wins', 0)
        proven_losses = stats.get('proven_losses', 0)

        if file_size > 1024 * 1024 * 1024:
            size_str = f"{file_size / (1024**3):.1f} GB"
        elif file_size > 1024 * 1024:
            size_str = f"{file_size / (1024**2):.1f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} B"

        if white_win_rate is not None:
            win_rate_str = f"{white_win_rate:.1%}"
            win_rate_sample = f"n={win_rate_sample_size}"
        else:
            win_rate_str = "N/A"
            win_rate_sample = ""

        states_val.set_text(f"{states_count:,}")
        games_val.set_text(f"{games_trained:,}")
        size_val.set_text(size_str)
        speed_val.set_text(f"{games_per_second:.0f}/s")
        wins_val.set_text(f"{proven_wins:,}")
        losses_val.set_text(f"{proven_losses:,}")
        winrate_val.set_text(win_rate_str)
        winrate_sample.set_text(win_rate_sample)

    with ui.card().classes('px-3 py-2 w-full'):
        with ui.row().classes('items-center justify-between w-full mb-1'):
            ui.label('AI Stats').classes('text-sm font-semibold text-gray-700')
            ui.button(icon='refresh', on_click=update_stats).props('flat dense round size=sm').classes('text-gray-500')

        with ui.grid(columns=4).classes('gap-x-4 gap-y-0 text-xs'):
            ui.label('States').classes('text-gray-500')
            ui.label('Games').classes('text-gray-500')
            ui.label('Size').classes('text-gray-500')
            ui.label('Speed').classes('text-gray-500')

            states_val = ui.label('...').classes('font-medium')
            games_val = ui.label('...').classes('font-medium')
            size_val = ui.label('...').classes('font-medium')
            speed_val = ui.label('...').classes('font-medium')

            ui.label('Proven Wins').classes('text-gray-500')
            ui.label('Proven Losses').classes('text-gray-500')
            ui.label('White Win Rate').classes('text-gray-500')
            winrate_sample = ui.label('').classes('text-gray-400 text-xs')

            wins_val = ui.label('...').classes('font-medium text-green-600')
            losses_val = ui.label('...').classes('font-medium text-red-600')
            winrate_val = ui.label('...').classes('font-medium')
            ui.label('')

    ui.timer(0.5, update_stats, once=True)
    ui.timer(5.0, update_stats)


def refresh_all():
    """Refresh all UI components."""
    global _analysis_result, _analysis_state_key, _analysis_elapsed
    # Clear stale analysis when state changes
    current_key = state_to_string(game.state, game.current_player)
    if _analysis_state_key != current_key:
        _analysis_result = None
        _analysis_state_key = None
        _analysis_elapsed = None
    game_board.refresh()
    game_status.refresh()
    game_controls_bottom.refresh()
    history_panel.refresh()


async def animate_ai_turn():
    """Animate a single AI turn, showing each move."""
    import asyncio
    global _ai_turn_in_progress, _animate_next, _ai_move_elapsed

    if _ai_turn_in_progress:
        return False

    _ai_turn_in_progress = True
    refresh_all()

    try:
        query_state = game.get_state_for_ai()
        start = time.monotonic()
        turn = await get_ai_move(query_state)
        _ai_move_elapsed = time.monotonic() - start

        if turn is None:
            game.game_over = True
            game.winner = 1 - game.current_player
            refresh_all()
            return False

        for move in turn:
            if game.game_over:
                break

            new_state = game.state.apply_move(move)
            if new_state is None:
                break

            game.state = new_state
            _animate_next = True
            refresh_all()
            await asyncio.sleep(MOVE_DELAY)

            winner = game.state.get_winner()
            if winner is not None:
                game.game_over = True
                if game.current_player == 0:
                    game.winner = winner
                else:
                    game.winner = 1 - winner
                game._add_to_history()
                refresh_all()
                return False

        if not game.game_over:
            game.finish_ai_turn()
            refresh_all()

        return not game.game_over
    finally:
        _ai_turn_in_progress = False
        refresh_all()


async def do_ai_move():
    """Execute a single AI turn."""
    await animate_ai_turn()


async def do_auto_play():
    """Auto-play: AI plays both sides until game over or stopped."""
    import asyncio
    global _auto_playing, _stop_auto_play

    _auto_playing = True
    _stop_auto_play = False
    refresh_all()

    try:
        while not game.game_over and not _stop_auto_play:
            cont = await animate_ai_turn()
            if not cont:
                break
            await asyncio.sleep(TURN_DELAY)
    finally:
        _auto_playing = False
        _stop_auto_play = False
        refresh_all()


def stop_auto_play_fn():
    """Signal auto-play to stop."""
    global _stop_auto_play
    _stop_auto_play = True


@ui.page('/')
def main_page():
    ui.dark_mode(False)
    ui.add_head_html('''<style>
body, .q-page { overflow: hidden !important; }
@keyframes piece-slide {
    from { transform: translate(var(--slide-x), var(--slide-y)); }
    to { transform: translate(0, 0); }
}
@keyframes piece-spin-cw {
    from { transform: rotate(-90deg); }
    to { transform: rotate(0deg); }
}
@keyframes piece-spin-ccw {
    from { transform: rotate(90deg); }
    to { transform: rotate(0deg); }
}
@keyframes piece-slide-off {
    from { transform: translate(0, 0); opacity: 1; }
    to { transform: translate(var(--off-x), var(--off-y)); opacity: 0; }
}
</style>''')

    # Full-height three-column layout: history | board + rules | logo + controls + ai
    with ui.element('div').classes('w-full').style(
        'display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px;'
        ' align-items: center; height: 100vh; padding: 32px 24px; box-sizing: border-box; overflow: hidden;'
    ):
        # Left column: game states
        with ui.card().classes('p-3').style('height: calc(100vh - 64px); display: flex; flex-direction: column; overflow: hidden;'):
            history_panel()

        # Center column: status + board + rules
        with ui.column().classes('items-center gap-0'):
            game_status()
            game_board()
            with ui.expansion('Game Rules', icon='help').classes('w-96 mt-2'):
                ui.markdown('''
**Duat** - Ancient Egyptian-themed abstract strategy game

- 4x4 board, 3 pieces per player
- Pieces have a direction (shown by arrow)
- On your turn: make exactly **1 or 3 moves** (not 2)
- A move is: slide forward OR rotate 90 degrees
- You can push pieces not facing you
- **Win:** Push an opponent's piece off the board
                ''')

        # Right column: logo, controls, ai stats
        with ui.column().classes('w-full items-center gap-8'):
            ui.image(LOGO_URL).classes('w-64')
            game_controls_bottom()
            ai_stats_panel()


def graceful_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutting_down
    if _shutting_down:
        print("\nForce quitting...")
        sys.exit(1)

    _shutting_down = True
    print("\nShutting down...")
    sys.exit(0)


if __name__ in {"__main__", "__mp_main__"}:
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    ui.run(title='Duat', reload=False, reconnect_timeout=30, show=True)
