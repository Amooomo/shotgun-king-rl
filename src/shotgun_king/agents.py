import numpy as np

from .core import (
    BOARD_SIZE,
    DIRECTION_VECTOR,
    Direction,
    MOVE_BASE,
    Piece,
    RELOAD,
    SHOOT_BASE,
)


class RandomAgent:

    def select_action(
        self,
        obs,
        env,
        rng: np.random.Generator,
    ) -> int:

        mask = env.action_masks()

        legal_actions = np.flatnonzero(
            mask
        )

        return int(
            rng.choice(legal_actions)
        )


class HeuristicAgent:

    def _visible_shots(
        self,
        env,
    ) -> list[tuple[float, int]]:

        state = env.game.state

        if state is None:
            return []

        if state.ammo <= 0:
            return []

        row, col = state.player_pos

        candidates = []

        piece_score = {
            Piece.WHITE_KING: 100.0,
            Piece.WHITE_ROOK: 10.0,
            Piece.WHITE_PAWN: 2.0,
        }

        for direction in Direction:

            dr, dc = DIRECTION_VECTOR[
                direction
            ]

            for distance in range(
                1,
                env.game.shot_range + 1,
            ):

                nr = row + dr * distance
                nc = col + dc * distance

                if not (
                    0 <= nr < BOARD_SIZE
                    and
                    0 <= nc < BOARD_SIZE
                ):
                    break

                piece = Piece(
                    state.board[nr, nc]
                )

                if piece == Piece.EMPTY:
                    continue

                if piece in piece_score:

                    score = (
                        piece_score[piece]
                        - 0.01 * distance
                    )

                    action = (
                        SHOOT_BASE
                        + int(direction)
                    )

                    candidates.append(
                        (
                            score,
                            action,
                        )
                    )

                # 第一枚棋子会挡住射线
                break

        return candidates

    def _find_white_king(
        self,
        env,
    ) -> tuple[int, int]:

        state = env.game.state

        positions = np.argwhere(
            state.board
            == Piece.WHITE_KING
        )

        return tuple(
            int(x)
            for x in positions[0]
        )

    def select_action(
        self,
        obs,
        env,
        rng: np.random.Generator,
    ) -> int:

        mask = env.action_masks()

        # --------------------------------
        # 1. 能射击时优先射击
        # King > Rook > Pawn
        # --------------------------------

        shots = self._visible_shots(
            env
        )

        if shots:

            _, action = max(
                shots,
                key=lambda x: x[0],
            )

            return int(action)

        # --------------------------------
        # 2. 找合法移动
        # 尽量接近 White King
        # --------------------------------

        state = env.game.state

        row, col = state.player_pos

        king_row, king_col = (
            self._find_white_king(env)
        )

        move_candidates = []

        for direction in Direction:

            action = (
                MOVE_BASE
                + int(direction)
            )

            if not mask[action]:
                continue

            dr, dc = DIRECTION_VECTOR[
                direction
            ]

            nr = row + dr
            nc = col + dc

            distance = max(
                abs(nr - king_row),
                abs(nc - king_col),
            )

            move_candidates.append(
                (
                    distance,
                    action,
                )
            )

        if move_candidates:

            best_distance = min(
                distance
                for distance, _
                in move_candidates
            )

            best_actions = [
                action
                for distance, action
                in move_candidates
                if distance == best_distance
            ]

            return int(
                rng.choice(best_actions)
            )

        # --------------------------------
        # 3. 实在没移动，再 RELOAD
        # --------------------------------

        if mask[RELOAD]:
            return RELOAD

        # --------------------------------
        # 4. 最后的保险
        # --------------------------------

        legal_actions = np.flatnonzero(
            mask
        )

        return int(
            rng.choice(legal_actions)
        )