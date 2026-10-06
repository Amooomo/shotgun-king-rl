from dataclasses import dataclass
from enum import IntEnum

import numpy as np


BOARD_SIZE = 8


class Piece(IntEnum):
    EMPTY = 0
    BLACK_KING = 1
    WHITE_KING = 2
    WHITE_ROOK = 3
    WHITE_PAWN = 4


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7


DIRECTION_VECTOR = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),

    Direction.UP_LEFT: (-1, -1),
    Direction.UP_RIGHT: (-1, 1),
    Direction.DOWN_LEFT: (1, -1),
    Direction.DOWN_RIGHT: (1, 1),
}


MOVE_BASE = 0
SHOOT_BASE = 8
RELOAD = 16

N_ACTIONS = 17

@dataclass
class GameState:
    board: np.ndarray

    player_pos: tuple[int, int]

    ammo: int
    max_ammo: int

    turn: int = 0

    won: bool = False
    lost: bool = False
    timeout: bool = False

class ShotgunKingLite:

    def __init__(
        self,
        shot_range: int = 4,
        max_ammo: int = 2,
        max_turns: int = 80,
        layout_mode: str = "fixed",
    ):
        if layout_mode not in (
            "fixed",
            "random_easy",
            "random_medium",
        ):
            raise ValueError(
                f"Unknown layout_mode: "
                f"{layout_mode}"
            )

        self.shot_range = shot_range
        self.max_ammo = max_ammo
        self.max_turns = max_turns

        self.layout_mode = layout_mode

        self.rng = np.random.default_rng()

        self.state: GameState | None = None

    def reset(self, seed: int | None = None) -> GameState:

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.layout_mode == "fixed":

            board, player_pos = (
                self._create_fixed_layout()
            )
        elif (
            self.layout_mode
            == "random_easy"
        ):

            board, player_pos = (
                self._create_random_easy_layout()
            )

        elif (
            self.layout_mode
            == "random_medium"
        ):

            board, player_pos = (
                self._create_random_medium_layout()
            )

        else:
            raise RuntimeError(
                "Unsupported layout mode."
            )

        self.state = GameState(
            board=board,
            player_pos=player_pos,
            ammo=self.max_ammo,
            max_ammo=self.max_ammo,
        )

        return self._snapshot()

    def _snapshot(self) -> GameState:

        s = self.state

        return GameState(
            board=s.board.copy(),
            player_pos=s.player_pos,
            ammo=s.ammo,
            max_ammo=s.max_ammo,
            turn=s.turn,
            won=s.won,
            lost=s.lost,
            timeout=s.timeout,
        )

    @staticmethod
    def _inside(row: int, col: int) -> bool:

        return (
            0 <= row < BOARD_SIZE
            and
            0 <= col < BOARD_SIZE
        )

    def legal_actions(self) -> list[int]:

        s = self.state

        actions = []

        row, col = s.player_pos

        # MOVE
        for direction in Direction:

            dr, dc = DIRECTION_VECTOR[direction]

            nr = row + dr
            nc = col + dc

            if not self._inside(nr, nc):
                continue

            if s.board[nr, nc] != Piece.EMPTY:
                continue

            actions.append(
                MOVE_BASE + int(direction)
            )

        # SHOOT
        if s.ammo > 0:

            for direction in Direction:

                actions.append(
                    SHOOT_BASE + int(direction)
                )

        # RELOAD
        if s.ammo < s.max_ammo:
            actions.append(RELOAD)

        return actions

    def legal_actions(self) -> list[int]:

        s = self.state

        actions = []

        row, col = s.player_pos

        # MOVE
        for direction in Direction:

            dr, dc = DIRECTION_VECTOR[direction]

            nr = row + dr
            nc = col + dc

            if not self._inside(nr, nc):
                continue

            if s.board[nr, nc] != Piece.EMPTY:
                continue

            actions.append(
                MOVE_BASE + int(direction)
            )

        # SHOOT
        if s.ammo > 0:

            for direction in Direction:

                actions.append(
                    SHOOT_BASE + int(direction)
                )

        # RELOAD
        if s.ammo < s.max_ammo:
            actions.append(RELOAD)

        return actions

    def _move_player(self, direction: Direction,events: list[str],):

        s = self.state

        row, col = s.player_pos

        dr, dc = DIRECTION_VECTOR[direction]

        nr = row + dr
        nc = col + dc

        s.board[row, col] = Piece.EMPTY
        s.board[nr, nc] = Piece.BLACK_KING

        s.player_pos = (nr, nc)

        # Shotgun King 风格：
        # 移动恢复一发弹药
        s.ammo = min(
            s.ammo + 1,
            s.max_ammo,
        )

        events.append(
            f"player_move:{direction.name}"
        )

    def _shoot(self, direction: Direction, events: list[str],):

        s = self.state

        s.ammo -= 1

        row, col = s.player_pos

        dr, dc = DIRECTION_VECTOR[direction]

        for distance in range(
            1,
            self.shot_range + 1,
        ):

            nr = row + dr * distance
            nc = col + dc * distance

            if not self._inside(nr, nc):
                break

            piece = Piece(s.board[nr, nc])

            if piece == Piece.EMPTY:
                continue

            if piece == Piece.BLACK_KING:
                continue

            s.board[nr, nc] = Piece.EMPTY

            events.append(
                f"shot_hit:{piece.name}"
            )

            if piece == Piece.WHITE_KING:

                s.won = True

                events.append(
                    "white_king_killed"
                )

            return

        events.append("shot_missed")

    def _reload(self, events: list[str],):

        self.state.ammo = self.state.max_ammo

        events.append("reload")

    def _pawn_attacks_player(
        self,
        row: int,
        col: int,
        player_pos: tuple[int, int] | None = None,
    ) -> bool:
        """Return True if the white pawn at (row, col) attacks the player.

        ``player_pos`` allows hypothetical queries. When it is ``None`` the
        current ``self.state.player_pos`` is used, which preserves the old
        behaviour.
        """

        if player_pos is None:
            player_pos = self.state.player_pos

        player_row, player_col = player_pos

        targets = [
            (row + 1, col - 1),
            (row + 1, col + 1),
        ]

        return (
            player_row,
            player_col,
        ) in targets

    def _rook_attacks_player(
        self,
        row: int,
        col: int,
        board: np.ndarray | None = None,
        player_pos: tuple[int, int] | None = None,
    ) -> bool:
        """Return True if the white rook at (row, col) attacks the player.

        ``board`` and ``player_pos`` allow hypothetical queries:

        * ``board is None``      -> use ``self.state.board``
        * ``player_pos is None`` -> use ``self.state.player_pos``

        Passing both is required for a correct hypothetical move query,
        because the Black King must be removed from its old cell before the
        rook's line-of-sight is evaluated.
        """

        if board is None:
            board = self.state.board

        if player_pos is None:
            player_pos = self.state.player_pos

        pr, pc = player_pos

        # 不同行也不同列
        if row != pr and col != pc:
            return False

        if row == pr:

            step = 1 if pc > col else -1

            for c in range(
                col + step,
                pc,
                step,
            ):

                if board[row, c] != Piece.EMPTY:
                    return False

            return True

        step = 1 if pr > row else -1

        for r in range(
            row + step,
            pr,
            step,
        ):

            if board[r, col] != Piece.EMPTY:
                return False

        return True

    def _player_is_attacked(
        self,
    ) -> bool:

        return bool(
            self.get_player_attackers()
        )

    def _enemy_candidates(self):

        s = self.state

        candidates = []

        pr, pc = s.player_pos

        for row in range(BOARD_SIZE):

            for col in range(BOARD_SIZE):

                piece = Piece(
                    s.board[row, col]
                )

                if piece == Piece.WHITE_PAWN:

                    nr = row + 1
                    nc = col

                    if (
                        self._inside(nr, nc)
                        and
                        s.board[nr, nc] == Piece.EMPTY
                    ):
                        candidates.append(
                            (
                                (row, col),
                                (nr, nc),
                                piece,
                            )
                        )

                elif piece == Piece.WHITE_ROOK:

                    current_distance = (
                        abs(row - pr)
                        + abs(col - pc)
                    )

                    for dr, dc in [
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                    ]:

                        nr = row + dr
                        nc = col + dc

                        if not self._inside(nr, nc):
                            continue

                        if s.board[nr, nc] != Piece.EMPTY:
                            continue

                        new_distance = (
                            abs(nr - pr)
                            + abs(nc - pc)
                        )

                        if new_distance < current_distance:

                            candidates.append(
                                (
                                    (row, col),
                                    (nr, nc),
                                    piece,
                                )
                            )

        return candidates

    def _enemy_phase(self, events: list[str],):

        s = self.state

        # 玩家移动后已经进入攻击范围
        if self._player_is_attacked():

            s.lost = True

            events.append(
                "player_killed"
            )

            return

        candidates = self._enemy_candidates()

        if candidates:

            index = self.rng.integers(
                len(candidates)
            )

            src, dst, piece = candidates[index]

            sr, sc = src
            dr, dc = dst

            s.board[sr, sc] = Piece.EMPTY
            s.board[dr, dc] = piece

            events.append(
                f"enemy_move:{piece.name}"
            )

        # 敌方移动之后再次判断攻击
        if self._player_is_attacked():

            s.lost = True

            events.append(
                "player_killed"
            )

    def step(self,action: int,) -> tuple[GameState, list[str]]:

        s = self.state

        if s is None:
            raise RuntimeError(
                "Call reset() before step()."
            )

        if s.won or s.lost or s.timeout:
            raise RuntimeError(
                "Episode already finished."
            )

        legal = self.legal_actions()

        if action not in legal:
            raise ValueError(
                f"Illegal action: {action}"
            )

        events = []

        if MOVE_BASE <= action < SHOOT_BASE:

            direction = Direction(
                action - MOVE_BASE
            )

            self._move_player(
                direction,
                events,
            )

        elif SHOOT_BASE <= action < RELOAD:

            direction = Direction(
                action - SHOOT_BASE
            )

            self._shoot(
                direction,
                events,
            )

        elif action == RELOAD:

            self._reload(events)

        # 杀掉白王以后直接结束
        if not s.won:

            self._enemy_phase(events)

        s.turn += 1

        if (
            s.turn >= self.max_turns
            and
            not s.won
            and
            not s.lost
        ):

            s.timeout = True

            events.append("timeout")

        return (
            self._snapshot(),
            events,
        )

    def render(self) -> str:

        symbols = {
            Piece.EMPTY: ".",
            Piece.BLACK_KING: "B",
            Piece.WHITE_KING: "K",
            Piece.WHITE_ROOK: "R",
            Piece.WHITE_PAWN: "P",
        }

        lines = []

        lines.append(
            "    0 1 2 3 4 5 6 7"
        )

        for row in range(BOARD_SIZE):

            cells = []

            for col in range(BOARD_SIZE):

                piece = Piece(
                    self.state.board[row, col]
                )

                cells.append(
                    symbols[piece]
                )

            lines.append(
                f"{row}   "
                + " ".join(cells)
            )

        lines.append(
            f"\nammo={self.state.ammo}"
            f"  turn={self.state.turn}"
            f"  won={self.state.won}"
            f"  lost={self.state.lost}"
        )

        return "\n".join(lines)

    def _create_fixed_layout(
        self,
    ) -> tuple[np.ndarray, tuple[int, int]]:

        board = np.zeros(
            (BOARD_SIZE, BOARD_SIZE),
            dtype=np.int8,
        )

        board[0, 4] = Piece.WHITE_KING

        board[0, 0] = Piece.WHITE_ROOK
        board[0, 7] = Piece.WHITE_ROOK

        board[3, 2] = Piece.WHITE_PAWN
        board[3, 4] = Piece.WHITE_PAWN
        board[3, 6] = Piece.WHITE_PAWN

        player_pos = (7, 4)

        board[player_pos] = (
            Piece.BLACK_KING
        )

        return board, player_pos

    def _create_random_easy_layout(
        self,
    ) -> tuple[np.ndarray, tuple[int, int]]:

        board = np.zeros(
            (BOARD_SIZE, BOARD_SIZE),
            dtype=np.int8,
        )

        # -------------------------
        # 两个 Rook 保持角落
        # -------------------------

        board[0, 0] = Piece.WHITE_ROOK
        board[0, 7] = Piece.WHITE_ROOK

        # -------------------------
        # White King
        # 顶部 1~6 随机
        # -------------------------

        white_king_col = int(
            self.rng.integers(
                1,
                7,
            )
        )

        board[
            0,
            white_king_col,
        ] = Piece.WHITE_KING

        # -------------------------
        # Black King
        # 底部 1~6 随机
        # -------------------------

        black_king_col = int(
            self.rng.integers(
                1,
                7,
            )
        )

        player_pos = (
            7,
            black_king_col,
        )

        board[player_pos] = (
            Piece.BLACK_KING
        )

        # -------------------------
        # 三个 Pawn
        # -------------------------

        pawn_cols = self.rng.choice(
            np.arange(1, 7),
            size=3,
            replace=False,
        )

        for col in pawn_cols:

            board[
                3,
                int(col),
            ] = Piece.WHITE_PAWN

        return board, player_pos

    def _sample_empty_position(
        self,
        board: np.ndarray,
        rows,
        cols,
    ) -> tuple[int, int]:

        candidates = []

        for row in rows:
            for col in cols:

                if board[row, col] == Piece.EMPTY:
                    candidates.append(
                        (row, col)
                    )

        if not candidates:
            raise RuntimeError(
                "No empty position available."
            )

        index = int(
            self.rng.integers(
                len(candidates)
            )
        )

        return candidates[index]

    def _create_random_medium_layout(
        self,
    ) -> tuple[
        np.ndarray,
        tuple[int, int],
    ]:

        board = np.zeros(
            (BOARD_SIZE, BOARD_SIZE),
            dtype=np.int8,
        )

        # =========================
        # Black King
        # =========================

        black_col = int(
            self.rng.integers(1, 7)
        )

        player_pos = (
            7,
            black_col,
        )

        board[player_pos] = (
            Piece.BLACK_KING
        )

        # =========================
        # White King
        # =========================

        king_pos = (
            self._sample_empty_position(
                board,
                rows=range(0, 2),
                cols=range(1, 7),
            )
        )

        board[king_pos] = (
            Piece.WHITE_KING
        )

        # =========================
        # Rooks: 1~2
        # =========================

        n_rooks = int(
            self.rng.integers(
                1,
                3,
            )
        )

        for _ in range(n_rooks):

            pos = (
                self._sample_empty_position(
                    board,
                    rows=range(0, 2),
                    cols=range(0, 8),
                )
            )

            board[pos] = (
                Piece.WHITE_ROOK
            )

        # =========================
        # Pawns: 2~4
        # =========================

        n_pawns = int(
            self.rng.integers(
                2,
                5,
            )
        )

        for _ in range(n_pawns):

            pos = (
                self._sample_empty_position(
                    board,
                    rows=range(2, 5),
                    cols=range(1, 7),
                )
            )

            board[pos] = (
                Piece.WHITE_PAWN
            )

        return board, player_pos


    def get_player_attackers(
        self,
        board: np.ndarray | None = None,
        player_pos: tuple[int, int] | None = None,
    ) -> list[tuple[int, int]]:
        """Return all white pieces currently attacking the player.

        This is the single source of truth for the white attack rules.
        ``board`` and ``player_pos`` may be supplied to query a hypothetical
        situation (for example, the board after a candidate Black King move).
        When they are ``None`` the current ``self.state`` is used, matching the
        original behaviour.
        """

        if board is None:
            board = self.state.board

        if player_pos is None:
            player_pos = self.state.player_pos

        attackers = []

        for row in range(BOARD_SIZE):

            for col in range(BOARD_SIZE):

                piece = Piece(
                    board[row, col]
                )

                if piece == Piece.WHITE_PAWN:

                    if self._pawn_attacks_player(
                        row,
                        col,
                        player_pos=player_pos,
                    ):
                        attackers.append(
                            (row, col)
                        )

                elif piece == Piece.WHITE_ROOK:

                    if self._rook_attacks_player(
                        row,
                        col,
                        board=board,
                        player_pos=player_pos,
                    ):
                        attackers.append(
                            (row, col)
                        )

                elif piece == Piece.WHITE_KING:

                    pr, pc = player_pos

                    dr = abs(pr - row)

                    dc = abs(pc - col)

                    if max(dr, dc) == 1:

                        attackers.append((row,col, ))

        return attackers