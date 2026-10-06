from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import (
    BOARD_SIZE,
    DIRECTION_VECTOR,
    Direction,
    GameState,
    MOVE_BASE,
    N_ACTIONS,
    Piece,
    SHOOT_BASE,
    ShotgunKingLite,
)

from .reward import (
    RewardCalculator,
    RewardConfig,
)

OBS_PIECES = (
    Piece.BLACK_KING,
    Piece.WHITE_KING,
    Piece.WHITE_ROOK,
    Piece.WHITE_PAWN,
)

GEOMETRY_DIMS = {
    "relative_v1": 25,
    "relative_v2": 47,
    "relative_v3": 56,
    "relative_v4": 55,
}

N_BOARD_CHANNELS = len(OBS_PIECES)


class ShotgunKingEnv(gym.Env):

    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        shot_range: int = 4,
        max_ammo: int = 2,
        max_turns: int = 80,
        reward_config: RewardConfig | None = None,
        action_mode: str = "full",
        layout_mode: str = "fixed",
        geometry_mode: str = "none",
    ):
        super().__init__()

        if render_mode not in (None, "ansi"):
            raise ValueError(
                f"Unsupported render_mode: {render_mode}"
            )

        if action_mode not in (
            "full",
            "pruned_shots",
        ):
            raise ValueError(
                f"Unknown action_mode: "
                f"{action_mode}"
            )

        if geometry_mode not in (
            "none",
            "relative_v1",
            "relative_v2",
            "relative_v3",
            "relative_v4",
        ):
            raise ValueError(
                f"Unknown geometry_mode: "
                f"{geometry_mode}"
            )

        self.geometry_mode = geometry_mode

        self.action_mode = action_mode

        self.render_mode = render_mode

        self.game = ShotgunKingLite(
            shot_range=shot_range,
            max_ammo=max_ammo,
            max_turns=max_turns,
            layout_mode=layout_mode,
        )

        self.layout_mode = layout_mode

        self.reward_config = (
            reward_config
            if reward_config is not None
            else RewardConfig.shaped()
        )

        self.reward_calculator = RewardCalculator(
            self.reward_config
        )

        # 动作空间：
        # 0~7   MOVE
        # 8~15  SHOOT
        # 16    RELOAD
        self.action_space = spaces.Discrete(
            N_ACTIONS
        )

        max_piece_value = max(
            int(piece)
            for piece in Piece
        )

        self.observation_space = spaces.Dict(
            {
                "board": spaces.MultiBinary(
                    (
                        N_BOARD_CHANNELS,
                        BOARD_SIZE,
                        BOARD_SIZE,
                    )
                ),

                "stats": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

        obs_spaces = {
            "board": spaces.MultiBinary(
                (4, 8, 8)
            ),

            "stats": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            ),
        }

        if (
            self.geometry_mode
            != "none"
        ):

            geometry_dim = (
                GEOMETRY_DIMS[
                    self.geometry_mode
                ]
            )

            obs_spaces[
                "geometry"
            ] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(
                    geometry_dim,
                ),
                dtype=np.float32,
            )


        self.observation_space = (
            spaces.Dict(obs_spaces)
        )

        self._state: GameState | None = None

        # Gym 层自己的 step 计数。
        # 与 core.turn 有意区分。
        self._elapsed_steps = 0

        self._episode_done = False

        self._episode_return = 0.0

        self._episode_reward_components: dict[
            str,
            float,
        ] = {}

    def _get_obs(self):

        obs = {

            "board":
                self._encode_board(),

            "stats":
                np.array(
                    [
                        (
                            self._state.ammo
                            /
                            self._state.max_ammo
                        ),

                        (
                            self._state.turn
                            /
                            self.game.max_turns
                        ),
                    ],
                    dtype=np.float32,
                ),
        }

        if self.geometry_mode in (
            "relative_v1",
            "relative_v2",
            "relative_v3",
            "relative_v4",
        ):

            obs["geometry"] = (
                self._get_geometry_obs()
            )

        return obs

    def _get_info(
        self,
        events: list[str] | None = None,
        illegal_action: bool = False,
        reward_components: dict[str, float] | None = None,
    ) -> dict[str, Any]:

        if self._state is None:
            raise RuntimeError(
                "Environment has not been reset."
            )

        return {
            "events": list(
                events or []
            ),

            "legal_actions": [
                int(action)
                for action
                in self.game.legal_actions()
            ],

            "action_mask":
                self.action_masks().copy(),

            "illegal_action":
                bool(illegal_action),

            "core_turn":
                int(self._state.turn),

            "elapsed_steps":
                int(self._elapsed_steps),

            # 当前这一步怎么得到 reward 的
            "reward_components":
                dict(reward_components or {}),

            # 当前 episode 已累计多少 reward
            "episode_return":
                float(self._episode_return),

            # 当前 episode 各项累计值
            "episode_reward_components":
                dict(
                    self._episode_reward_components
                ),

            # Stage 5/6 会直接使用
            "is_success":
                bool(self._state.won),
        }
        
        

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):

        # Gymnasium 要求调用它
        super().reset(seed=seed)

        self._elapsed_steps = 0
        self._episode_done = False

        self._state = self.game.reset(
            seed=seed
        )

        observation = self._get_obs()
        info = self._get_info()
        self._episode_return = 0.0

        self._episode_reward_components = {}

        return observation, info

    def step(self, action):

        if self._state is None:
            raise RuntimeError(
                "Call reset() before step()."
            )

        if self._episode_done:
            raise RuntimeError(
                "Episode finished. Call reset()."
            )

        if not self.action_space.contains(action):
            raise ValueError(
                f"Action {action} is outside "
                f"the action space."
            )

        action = int(action)

        self._elapsed_steps += 1

        legal_actions = (
            self.game.legal_actions()
        )

        is_legal = (
            action in legal_actions
        )

        if is_legal:

            self._state, events = (
                self.game.step(action)
            )

        else:
            # Stage 2：
            # 非法动作暂时定义为 no-op。
            #
            # Stage 3：
            # Action Mask
            #
            # Stage 4：
            # 决定是否给予负奖励。
            events = [
                f"illegal_action:{action}"
            ]

        terminated = bool(
            self._state.won
            or
            self._state.lost
        )

        truncated = bool(
            not terminated
            and
            (
                self._state.timeout
                or
                self._elapsed_steps
                >= self.game.max_turns
            )
        )

        reward, reward_components = (
            self.reward_calculator.calculate(
                events=events,
                won=self._state.won,
                lost=self._state.lost,
                truncated=truncated,
                illegal_action=not is_legal,
            )
        )

        self._episode_return += reward

        for (
            name,
            value,
        ) in reward_components.items():

            self._episode_reward_components[name] = (
                self._episode_reward_components.get(
                    name,
                    0.0,
                )
                + value
            )

        observation = self._get_obs()

        info = self._get_info(
            events=events,
            illegal_action=not is_legal,
            reward_components=reward_components,
        )

        if (
            truncated
            and
            not self._state.timeout
        ):
            info["time_limit_reached"] = True

        self._episode_done = (
            terminated or truncated
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):

        if self._state is None:
            return ""

        return self.game.render()

    def close(self):
        pass

    def _encode_board(self,) -> np.ndarray:

        if self._state is None:
            raise RuntimeError(
                "Environment has not been reset."
            )

        encoded = np.zeros(
            (
                N_BOARD_CHANNELS,
                BOARD_SIZE,
                BOARD_SIZE,
            ),
            dtype=np.int8,
        )

        for channel, piece in enumerate(
            OBS_PIECES
        ):

            encoded[channel] = (
                self._state.board
                == int(piece)
            ).astype(np.int8)

        return encoded

    def action_masks(
        self,
    ) -> np.ndarray:

        if self._state is None:
            raise RuntimeError(
                "Environment has not been reset."
            )

        mask = np.zeros(
            self.action_space.n,
            dtype=np.bool_,
        )

        # -----------------------------
        # 第一层：
        # 游戏规则上的合法动作
        # -----------------------------

        for action in (
            self.game.legal_actions()
        ):
            mask[action] = True

        # -----------------------------
        # 第二层：
        # RL Action Pruning
        # -----------------------------

        if (
            self.action_mode
            == "pruned_shots"
        ):

            for direction in Direction:

                action = (
                    SHOOT_BASE
                    + int(direction)
                )

                if not mask[action]:
                    continue

                if not self._shot_has_target(
                    direction
                ):
                    mask[action] = False

        if not np.any(mask):
            raise RuntimeError(
                "Action pruning produced "
                "an empty action set."
            )

        return mask

    def _shot_has_target(
        self,
        direction: Direction,
    ) -> bool:

        if self._state is None:
            raise RuntimeError(
                "Environment has not been reset."
            )

        if self._state.ammo <= 0:
            return False

        row, col = (
            self._state.player_pos
        )

        dr, dc = DIRECTION_VECTOR[
            direction
        ]

        for distance in range(
            1,
            self.game.shot_range + 1,
        ):

            nr = row + dr * distance
            nc = col + dc * distance

            if not self._inside_board(
                nr,
                nc,
            ):
                break

            piece = Piece(
                self._state.board[
                    nr,
                    nc,
                ]
            )

            if piece == Piece.EMPTY:
                continue

            # 射线碰到第一个棋子后即结束。
            return piece in (
                Piece.WHITE_KING,
                Piece.WHITE_ROOK,
                Piece.WHITE_PAWN,
            )

        return False

    @staticmethod
    def _inside_board(
        row: int,
        col: int,
    ) -> bool:

        return (
            0 <= row < BOARD_SIZE
            and
            0 <= col < BOARD_SIZE
        )

    def _piece_positions(
        self,
        piece: Piece,
    ):

        positions = np.argwhere(
            self._state.board
            == piece
        )

        return [
            (
                int(row),
                int(col),
            )
            for row, col
            in positions
        ]

    def _get_geometry_obs_v1(
        self,
    ) -> np.ndarray:

        br, bc = self._state.player_pos

        king_positions = (
            self._piece_positions(
                Piece.WHITE_KING
            )
        )

        if king_positions:

            kr, kc = king_positions[0]

            dr = kr - br
            dc = kc - bc

            chebyshev = max(
                abs(dr),
                abs(dc),
            )

            manhattan = (
                abs(dr)
                + abs(dc)
            )

            features = [
                dr / 7.0,
                dc / 7.0,
                chebyshev / 7.0,
                manhattan / 14.0,
                float(dr == 0),
                float(dc == 0),
                float(
                    abs(dr)
                    == abs(dc)
                ),
            ]

        else:

            if not self._state.won:
                raise RuntimeError(
                    "WHITE_KING missing "
                    "from non-winning state."
                )

            features = [
                0.0
            ] * 7

        def append_pieces(
            positions,
            max_count,
        ):

            positions = sorted(
                positions,
                key=lambda pos: (
                    max(
                        abs(pos[0] - br),
                        abs(pos[1] - bc),
                    ),
                    pos[0],
                    pos[1],
                ),
            )

            for i in range(
                max_count
            ):

                if i < len(positions):

                    row, col = (
                        positions[i]
                    )

                    features.extend(
                        [
                            1.0,
                            (
                                row - br
                            ) / 7.0,
                            (
                                col - bc
                            ) / 7.0,
                        ]
                    )

                else:

                    features.extend(
                        [
                            0.0,
                            0.0,
                            0.0,
                        ]
                    )

        append_pieces(
            self._piece_positions(
                Piece.WHITE_ROOK
            ),
            max_count=2,
        )

        append_pieces(
            self._piece_positions(
                Piece.WHITE_PAWN
            ),
            max_count=4,
        )

        geometry = np.asarray(
            features,
            dtype=np.float32,
        )

        assert geometry.shape == (
            25,
        )

        assert np.all(
            np.isfinite(geometry)
        )

        return geometry

    def _get_geometry_obs_v2(
        self,
    ) -> np.ndarray:

        br, bc = self._state.player_pos

        king_positions = (
            self._piece_positions(
                Piece.WHITE_KING
            )
        )

        if king_positions:

            kr, kc = (
                king_positions[0]
            )

            dr = kr - br
            dc = kc - bc

            features = [
                dr / 7.0,
                dc / 7.0,

                max(
                    abs(dr),
                    abs(dc),
                ) / 7.0,

                (
                    abs(dr)
                    + abs(dc)
                ) / 14.0,

                float(dr == 0),
                float(dc == 0),

                float(
                    abs(dr)
                    == abs(dc)
                ),
            ]

        else:

            if not self._state.won:
                raise RuntimeError(
                    "WHITE_KING missing "
                    "from non-winning state."
                )

            features = [
                0.0
            ] * 7

        for direction in Direction:

            features.extend(
                self._ray_features(
                    direction
                )
            )

        geometry = np.asarray(
            features,
            dtype=np.float32,
        )

        assert geometry.shape == (
            47,
        )

        assert np.all(
            np.isfinite(geometry)
        )

        return geometry

    def _ray_features(
        self,
        direction: Direction,
    ) -> list[float]:

        br, bc = (
            self._state.player_pos
        )

        dr, dc = (
            DIRECTION_VECTOR[
                direction
            ]
        )

        for distance in range(
            1,
            self.game.shot_range + 1,
        ):

            row = br + dr * distance
            col = bc + dc * distance

            if not self._inside_board(
                row,
                col,
            ):
                break

            piece = Piece(
                self._state.board[
                    row,
                    col,
                ]
            )

            if piece == Piece.EMPTY:
                continue

            return [
                1.0,

                float(
                    piece
                    == Piece.WHITE_KING
                ),

                float(
                    piece
                    == Piece.WHITE_ROOK
                ),

                float(
                    piece
                    == Piece.WHITE_PAWN
                ),

                distance
                / self.game.shot_range,
            ]

        return [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    def _get_threat_features(
        self,
    ) -> np.ndarray:

        attackers = (
            self.game
            .get_player_attackers()
        )

        flags = np.zeros(
            8,
            dtype=np.float32,
        )

        br, bc = (
            self._state.player_pos
        )

        vector_to_direction = {
            vector: direction
            for (
                direction,
                vector,
            )
            in DIRECTION_VECTOR.items()
        }

        for (
            row,
            col,
        ) in attackers:

            dr = row - br
            dc = col - bc

            # 只取方向符号
            sr = (
                -1
                if dr < 0
                else
                1
                if dr > 0
                else
                0
            )

            sc = (
                -1
                if dc < 0
                else
                1
                if dc > 0
                else
                0
            )

            direction = (
                vector_to_direction.get(
                    (
                        sr,
                        sc,
                    )
                )
            )

            if direction is not None:

                flags[
                    int(direction)
                ] = 1.0

        result = np.concatenate(
            [
                np.array(
                    [
                        float(
                            len(attackers)
                            > 0
                        )
                    ],
                    dtype=np.float32,
                ),

                flags,
            ]
        )

        assert result.shape == (
            9,
        )

        return result

    def _get_geometry_obs_v3(
        self,
    ) -> np.ndarray:

        v2 = (
            self._get_geometry_obs_v2()
        )

        threat = (
            self._get_threat_features()
        )

        geometry = np.concatenate(
            [
                v2,
                threat,
            ]
        ).astype(
            np.float32
        )

        assert geometry.shape == (
            56,
        )

        assert np.all(
            np.isfinite(
                geometry
            )
        )

        return geometry

    def _get_geometry_obs_v4(
        self,
    ) -> np.ndarray:

        v2 = (
            self._get_geometry_obs_v2()
        )

        move_danger = (
            self._get_move_danger_features()
        )

        geometry = np.concatenate(
            [
                v2,
                move_danger,
            ]
        ).astype(
            np.float32
        )

        assert geometry.shape == (
            55,
        )

        assert np.all(
            np.isfinite(
                geometry
            )
        )

        return geometry

    def _get_move_danger_features(
        self,
    ) -> np.ndarray:
        """Action-conditioned immediate move danger (8 dims).

        For each MOVE direction, in the same order as ``Direction``:

        * ``0.0`` -> the move is illegal, or the resulting square is not
          attacked by any white piece right now.
        * ``1.0`` -> the move is legal and the resulting square is already
          attacked by a white piece right now.

        Only *immediate* danger is predicted: the white pieces are not allowed
        to move first. Legality itself is not re-encoded here (the action mask
        already covers "can I do it?"); an illegal MOVE is reported as ``0.0``.

        The hypothetical board must both remove the Black King from its old
        square and place it on the new square. Moving only the queried player
        position would leave the old Black King as a phantom blocker on the
        Rook line of sight and could report a false "safe" cell.
        """

        if self._state is None:

            raise RuntimeError(
                "Environment has not been reset."
            )

        board = self._state.board

        row, col = (
            self._state.player_pos
        )

        legal_actions = set(
            self.game.legal_actions()
        )

        features = []

        for direction in Direction:

            action = (
                MOVE_BASE
                + int(direction)
            )

            if action not in legal_actions:

                features.append(0.0)

                continue

            dr, dc = (
                DIRECTION_VECTOR[
                    direction
                ]
            )

            nr = row + dr
            nc = col + dc

            hypothetical_board = (
                board.copy()
            )

            hypothetical_board[
                row, col
            ] = Piece.EMPTY

            hypothetical_board[
                nr, nc
            ] = Piece.BLACK_KING

            attackers = (
                self.game
                .get_player_attackers(
                    board=hypothetical_board,
                    player_pos=(nr, nc),
                )
            )

            features.append(
                float(bool(attackers))
            )

        result = np.asarray(
            features,
            dtype=np.float32,
        )

        assert result.shape == (8,)

        assert np.all(
            np.isfinite(result)
        )

        return result

    def _get_geometry_obs(
        self,
    ) -> np.ndarray:

        if (
            self.geometry_mode
            == "relative_v1"
        ):

            return (
                self._get_geometry_obs_v1()
            )

        if (
            self.geometry_mode
            == "relative_v2"
        ):

            return (
                self._get_geometry_obs_v2()
            )

        if (
            self.geometry_mode
            == "relative_v3"
        ):

            return (
                self._get_geometry_obs_v3()
            )

        if (
            self.geometry_mode
            == "relative_v4"
        ):

            return (
                self._get_geometry_obs_v4()
            )

        raise RuntimeError(
            f"Unsupported geometry mode: "
            f"{self.geometry_mode}"
        )