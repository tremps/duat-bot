"""Web UI for Duat game using NiceGUI."""

import os
import signal
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
from nicegui import ui, app

from duat import GameState, Direction, Turn

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


class GameMode(Enum):
    PVP = "Player vs Player"
    PVAI = "Player vs AI"
    AIVAI = "AI vs AI"


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
        async with httpx.AsyncClient(timeout=30.0) as client:
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


async def get_ai_stats() -> dict:
    """Get AI server statistics."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
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
        self.mode: GameMode = GameMode.PVP
        self.human_player: int = 0  # Which player the human controls (0=White, 1=Black)
        self.selected_piece: Optional[int] = None
        self.current_turn_moves: list = []  # Moves made so far this turn
        self.game_over: bool = False
        self.winner: Optional[int] = None
        # Whether a game mode is active
        self.game_active: bool = False
        # For undo functionality - store states after each move
        self.undo_stack: list = []  # Stack of (state, selected_piece) for undo
        # Track whose turn it is (since GameState no longer has current_player)
        self.current_player: int = 0

    def reset_game(self):
        """Start a new game."""
        self.state = GameState.initial()
        self.selected_piece = None
        self.current_turn_moves = []
        self.game_over = False
        self.winner = None
        self.undo_stack = []
        self.current_player = 0

    def quit_game(self):
        """Quit current game and return to mode selection."""
        self.reset_game()
        self.game_active = False

    def start_game(self, mode: GameMode, human_player: int = 0):
        """Start a new game with the given mode."""
        self.mode = mode
        self.human_player = human_player
        self.reset_game()
        self.game_active = True

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
        # Display is flipped, so North appears as down, South as up
        arrows = {
            Direction.NORTH: "\u2193",  # Down arrow
            Direction.SOUTH: "\u2191",  # Up arrow
            Direction.EAST: "\u2192",   # Right arrow
            Direction.WEST: "\u2190",   # Left arrow
        }
        return arrows[direction]

    def get_current_player_name(self) -> str:
        """Get display name for current player."""
        return "White" if self.current_player == 0 else "Black"

    def is_human_turn(self) -> bool:
        """Check if it's the human's turn."""
        if self.mode == GameMode.PVP:
            return True
        if self.mode == GameMode.AIVAI:
            return False
        return self.current_player == self.human_player

    def can_select_piece(self, piece_idx: int) -> bool:
        """Check if a piece can be selected."""
        if self.game_over or not self.is_human_turn():
            return False
        piece = self.state.pieces[piece_idx]
        # In the internal state, current player's pieces are always P0
        # So we check if the piece belongs to P0 (which is the current player)
        return piece[3] == 0

    def select_piece(self, piece_idx: int):
        """Select a piece for moving."""
        if self.can_select_piece(piece_idx):
            self.selected_piece = piece_idx

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

        # Save state for undo
        self.undo_stack.append((self.state, self.selected_piece))

        move = (self.selected_piece, action)
        new_state = self.state.apply_move(move)
        if new_state is None:
            self.undo_stack.pop()  # Remove failed undo state
            return False

        self.current_turn_moves.append(move)
        self.state = new_state
        # Keep piece selected for easier multi-move turns

        # Check for win mid-turn - auto submit if won
        winner = self.state.get_winner()
        if winner is not None:
            self.game_over = True
            # winner=0 means P0 in state won, winner=1 means P1 in state won
            # Map to actual player based on current_player perspective
            if self.current_player == 0:
                self.winner = winner
            else:
                self.winner = 1 - winner
            self.selected_piece = None
            self.current_turn_moves = []
            self.undo_stack = []
            return True

        return True

    def undo_move(self) -> bool:
        """Undo the last move. Returns True if successful."""
        if not self.undo_stack or not self.current_turn_moves:
            return False

        # Restore previous state
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

        # Swap players in state and track whose turn it is
        self.state = self.state.swap_players()
        self.current_player = 1 - self.current_player
        self.current_turn_moves = []
        self.selected_piece = None
        self.undo_stack = []

    def can_end_turn(self) -> bool:
        """Check if the current turn can be ended."""
        return self.can_submit_turn()

    def end_turn(self):
        """End the current turn (alias for submit_turn)."""
        self.submit_turn()

    def get_state_for_ai(self) -> GameState:
        """
        Get state in format suitable for AI query.

        AI always plays as P0, so we return the state as-is
        (since current player is already represented as P0).
        """
        return self.state

    def apply_ai_turn(self, turn: Turn) -> bool:
        """
        Apply an AI turn. Returns True if game continues.

        The turn comes from AI which sees the state as P0,
        so we can apply it directly.
        """
        for move in turn:
            new_state = self.state.apply_move(move)
            if new_state is None:
                return False

            self.state = new_state

            # Check for win
            winner = self.state.get_winner()
            if winner is not None:
                self.game_over = True
                if self.current_player == 0:
                    self.winner = winner
                else:
                    self.winner = 1 - winner
                self.selected_piece = None
                return False

        return True

    def finish_ai_turn(self):
        """Switch player after AI turn completes."""
        if not self.game_over:
            self.state = self.state.swap_players()
            self.current_player = 1 - self.current_player
        self.selected_piece = None


# Global game instance
game = DuatGame()


@ui.refreshable
def game_board():
    """Render the game board."""
    # Overlay to gray out board when not active
    board_classes = 'grid grid-cols-4 gap-1 p-4 bg-amber-800 rounded-lg'
    if not game.game_active:
        board_classes += ' opacity-50 pointer-events-none'

    with ui.element('div').classes(board_classes):
        for display_row in range(4):
            for col in range(4):
                piece_info = game.get_piece_at_display(display_row, col)

                # Determine cell styling
                is_selected = False
                piece_idx = None
                if piece_info:
                    piece_idx, piece = piece_info
                    is_selected = (piece_idx == game.selected_piece)

                cell_classes = 'w-16 h-16 rounded flex items-center justify-center text-3xl cursor-pointer '
                if is_selected:
                    cell_classes += 'bg-yellow-300 ring-4 ring-yellow-500'
                else:
                    cell_classes += 'bg-amber-200 hover:bg-amber-300'

                # Create cell with click handler
                idx = piece_idx  # Capture for closure

                def make_click_handler(idx=idx):
                    def handler():
                        if game.game_active and idx is not None and game.can_select_piece(idx):
                            game.select_piece(idx)
                            refresh_all()
                    return handler

                with ui.element('div').classes(cell_classes).on('click', make_click_handler()):
                    if piece_info:
                        _, piece = piece_info
                        row, col_pos, direction, player = piece

                        # In the current state, P0 is always the current player
                        # Map to display colors based on current_player
                        actual_player = player if game.current_player == 0 else (1 - player)

                        # Player 0 = White, Player 1 = Black
                        if actual_player == 0:
                            bg = 'bg-gray-100 border-2 border-gray-400'
                            color = 'text-gray-800'
                        else:
                            bg = 'bg-gray-800'
                            color = 'text-gray-100'
                        arrow = game.get_direction_arrow(direction)

                        with ui.element('div').classes(
                            f'w-12 h-12 rounded-full {bg} flex items-center justify-center {color} pointer-events-none'
                        ):
                            ui.label(arrow).classes('text-2xl font-bold pointer-events-none')


@ui.refreshable
def game_status():
    """Show current game status."""
    if not game.game_active:
        ui.label('Select a game mode to start').classes('text-xl text-gray-500')
    elif game.game_over:
        winner_name = "White" if game.winner == 0 else "Black"
        ui.label(f'{winner_name} wins!').classes('text-2xl font-bold text-green-600')
    else:
        player_name = game.get_current_player_name()
        ui.label(f"{player_name}'s Turn").classes('text-xl font-bold')


@ui.refreshable
def game_controls_top():
    """Forward button above the board."""
    if not game.game_active or game.game_over or not game.is_human_turn():
        # Empty placeholder to maintain layout
        ui.element('div').classes('h-10')
        return

    has_selection = game.selected_piece is not None
    fwd_btn = ui.button('Forward', on_click=lambda: make_move('F')).classes('bg-green-500 w-32')
    if not has_selection or not game.can_make_move('F'):
        fwd_btn.disable()


@ui.refreshable
def game_controls_left():
    """CW button to the left of the board."""
    if not game.game_active or game.game_over or not game.is_human_turn():
        ui.element('div').classes('w-24')
        return

    has_selection = game.selected_piece is not None
    # Note: Actions are swapped because display is vertically flipped
    cw_btn = ui.button('\u21bb', on_click=lambda: make_move('CCW')).classes('bg-blue-500 w-16 h-16 text-3xl')
    if not has_selection or not game.can_make_move('CCW'):
        cw_btn.disable()


@ui.refreshable
def game_controls_right():
    """CCW button to the right of the board."""
    if not game.game_active or game.game_over or not game.is_human_turn():
        ui.element('div').classes('w-24')
        return

    has_selection = game.selected_piece is not None
    # Note: Actions are swapped because display is vertically flipped
    ccw_btn = ui.button('\u21ba', on_click=lambda: make_move('CW')).classes('bg-blue-500 w-16 h-16 text-3xl')
    if not has_selection or not game.can_make_move('CW'):
        ccw_btn.disable()


@ui.refreshable
def game_controls_bottom():
    """Undo/Submit buttons and status below the board."""
    if not game.game_active:
        return

    if game.game_over:
        ui.button('New Game', on_click=lambda: (game.reset_game(), refresh_all())).classes('bg-green-500')
        return

    if not game.is_human_turn():
        if game.mode == GameMode.AIVAI:
            with ui.row().classes('gap-2'):
                ui.button('Step (AI Move)', on_click=animate_ai_turn).classes('bg-blue-500')
                ui.button('Auto Play', on_click=auto_play).classes('bg-purple-500')
        else:
            ui.label('AI is thinking...').classes('text-gray-600')
            # Only create timer if AI turn not already in progress
            if not _ai_turn_in_progress:
                ui.timer(TURN_DELAY, animate_ai_turn, once=True)
        return

    # Human turn controls
    move_count = len(game.current_turn_moves)

    # Undo and Submit buttons
    with ui.row().classes('gap-2'):
        undo_btn = ui.button('Undo', on_click=lambda: (game.undo_move(), refresh_all())).classes('bg-yellow-500')
        if move_count == 0:
            undo_btn.disable()

        can_submit = game.can_submit_turn()
        submit_text = f'Submit ({move_count} move{"s" if move_count != 1 else ""})'
        submit_btn = ui.button(submit_text, on_click=lambda: (game.submit_turn(), refresh_all())).classes('bg-orange-500')
        if not can_submit:
            submit_btn.disable()


@ui.refreshable
def mode_buttons():
    """Game mode selection buttons."""
    pvp = ui.button('Player vs Player', on_click=lambda: set_mode(GameMode.PVP)).classes('bg-green-500')
    white = ui.button('Play as White vs AI', on_click=lambda: set_mode(GameMode.PVAI, 0)).classes('bg-blue-500')
    black = ui.button('Play as Black vs AI', on_click=lambda: set_mode(GameMode.PVAI, 1)).classes('bg-gray-700')
    aivai = ui.button('AI vs AI', on_click=lambda: set_mode(GameMode.AIVAI)).classes('bg-purple-500')
    if game.game_active:
        pvp.disable()
        white.disable()
        black.disable()
        aivai.disable()


@ui.refreshable
def quit_button():
    """Quit game button."""
    qb = ui.button('Quit', on_click=quit_game).classes('bg-red-500')
    if not game.game_active:
        qb.disable()


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

        # Format file size
        if file_size > 1024 * 1024 * 1024:
            size_str = f"{file_size / (1024**3):.1f} GB"
        elif file_size > 1024 * 1024:
            size_str = f"{file_size / (1024**2):.1f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} B"

        # Format win rate
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

    with ui.card().classes('mt-4 px-4 py-3'):
        with ui.row().classes('items-center justify-between w-full mb-2'):
            ui.label('AI Stats').classes('text-sm font-semibold text-gray-700')
            ui.button(icon='refresh', on_click=update_stats).props('flat dense round size=sm').classes('text-gray-500')

        with ui.grid(columns=4).classes('gap-x-6 gap-y-1 text-sm'):
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
            ui.label('')  # empty cell

    # Initial load and auto-refresh every 5 seconds
    ui.timer(0.5, update_stats, once=True)
    ui.timer(5.0, update_stats)


def refresh_all():
    """Refresh all UI components."""
    game_board.refresh()
    game_status.refresh()
    game_controls_top.refresh()
    game_controls_left.refresh()
    game_controls_right.refresh()
    game_controls_bottom.refresh()
    mode_buttons.refresh()
    quit_button.refresh()


def make_move(action: str):
    """Make a move and refresh UI."""
    game.make_move(action)
    refresh_all()


async def animate_ai_turn():
    """Animate a single AI turn, showing each move."""
    import asyncio
    global _ai_turn_in_progress

    if _ai_turn_in_progress:
        return False

    _ai_turn_in_progress = True

    try:
        # Get AI move from server
        query_state = game.get_state_for_ai()
        turn = await get_ai_move(query_state)

        if turn is None:
            # AI has no moves or server error - AI loses
            game.game_over = True
            game.winner = 1 - game.current_player
            refresh_all()
            return False

        # Apply each move with a delay
        for move in turn:
            if game.game_over:
                break

            new_state = game.state.apply_move(move)
            if new_state is None:
                break

            game.state = new_state
            refresh_all()
            await asyncio.sleep(MOVE_DELAY)

            # Check for win
            winner = game.state.get_winner()
            if winner is not None:
                game.game_over = True
                if game.current_player == 0:
                    game.winner = winner
                else:
                    game.winner = 1 - winner
                refresh_all()
                return False

        # Finish the turn (switch player)
        if not game.game_over:
            game.finish_ai_turn()
            refresh_all()

        return not game.game_over
    finally:
        _ai_turn_in_progress = False


async def auto_play():
    """Auto-play AI vs AI game."""
    import asyncio
    while not game.game_over:
        await animate_ai_turn()
        await asyncio.sleep(TURN_DELAY)


def set_mode(mode: GameMode, human_player: int = 0):
    """Change game mode and start new game."""
    game.start_game(mode, human_player)
    refresh_all()


def quit_game():
    """Quit current game and return to mode selection."""
    game.quit_game()
    refresh_all()


@ui.page('/')
def main_page():
    ui.dark_mode(False)

    with ui.column().classes('w-full items-center p-4'):
        ui.image(LOGO_URL).classes('w-80 mb-4')

        # Mode selection
        with ui.card().classes('mb-4'):
            ui.label('Game Mode').classes('text-lg font-bold mb-2')
            with ui.row():
                mode_buttons()
                quit_button()

        # Status
        with ui.element('div').classes('mb-4'):
            game_status()

        # Game board with controls around it
        with ui.column().classes('items-center gap-2'):
            # Forward button on top
            game_controls_top()

            # Board with CCW on left and CW on right
            with ui.row().classes('items-center gap-2'):
                game_controls_left()
                game_board()
                game_controls_right()

            # Undo/Submit below
            game_controls_bottom()

        # AI Stats panel
        ai_stats_panel()

        # Rules
        with ui.expansion('Game Rules', icon='help').classes('mt-4 w-96'):
            ui.markdown('''
**Duat** - Ancient Egyptian-themed abstract strategy game

- 4x4 board, 3 pieces per player
- Pieces have a direction (shown by arrow)
- On your turn: make exactly **1 or 3 moves** (not 2)
- A move is: slide forward OR rotate 90 degrees
- You can push pieces not facing you
- **Win:** Push an opponent's piece off the board
            ''')


def graceful_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutting_down
    if _shutting_down:
        # Second signal - force quit
        print("\nForce quitting...")
        sys.exit(1)

    _shutting_down = True
    print("\nShutting down...")
    sys.exit(0)


if __name__ in {"__main__", "__mp_main__"}:
    # Register signal handlers
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    ui.run(title='Duat', reload=False, reconnect_timeout=30, show=True)
