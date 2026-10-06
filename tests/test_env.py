import numpy as np

import pytest

from shotgun_king.reward import (
    RewardConfig,
)

from stable_baselines3.common.env_checker import (
    check_env,
)

from shotgun_king.core import (
    Direction,
    GameState,
    MOVE_BASE,
    N_ACTIONS,
    RELOAD,
    SHOOT_BASE,
    Piece,
)

from shotgun_king.env import (
    ShotgunKingEnv,
)


def test_env_checker():

    env = ShotgunKingEnv()

    check_env(
        env,
        warn=True,
    )


def test_reset_observation():

    env = ShotgunKingEnv()

    obs, info = env.reset(
        seed=42
    )

    assert env.observation_space.contains(
        obs
    )

    assert obs["board"].shape == (
        4,
        8,
        8,
    )

    assert obs["stats"].shape == (2,)

    assert obs["stats"].dtype == np.float32

    assert obs["stats"][0] == 1.0

    assert obs["stats"][1] == 0.0


def test_board_encoding():

    env = ShotgunKingEnv()

    obs, _ = env.reset(
        seed=42
    )

    board = obs["board"]

    # BLACK KING
    assert board[0].sum() == 1

    # WHITE KING
    assert board[1].sum() == 1

    # WHITE ROOK
    assert board[2].sum() == 2

    # WHITE PAWN
    assert board[3].sum() == 3

    # 总共 7 个棋子
    assert board.sum() == 7


def test_step_returns_gymnasium_format():

    env = ShotgunKingEnv()

    env.reset(seed=42)

    action = (
        MOVE_BASE
        + Direction.UP
    )

    result = env.step(action)

    assert len(result) == 5

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = result

    assert env.observation_space.contains(
        obs
    )

    assert isinstance(
        reward,
        float,
    )

    assert isinstance(
        terminated,
        bool,
    )

    assert isinstance(
        truncated,
        bool,
    )

    assert isinstance(
        info,
        dict,
    )


def test_illegal_action_does_not_crash():

    env = ShotgunKingEnv()

    env.reset(seed=42)

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(RELOAD)

    assert info["illegal_action"]

    assert (
        f"illegal_action:{RELOAD}"
        in info["events"]
    )


def test_seeded_reset_is_deterministic():

    env = ShotgunKingEnv()

    obs1, _ = env.reset(
        seed=42
    )

    board1 = obs1["board"].copy()

    obs2, _ = env.reset(
        seed=42
    )

    board2 = obs2["board"].copy()

    assert np.array_equal(
        board1,
        board2,
    )


def test_initial_action_mask():

    env = ShotgunKingEnv()

    env.reset(seed=42)

    mask = env.action_masks()

    assert mask.shape == (
        N_ACTIONS,
    )

    assert mask.dtype == np.bool_

    # 初始可以向上
    assert mask[
        MOVE_BASE + Direction.UP
    ]

    # 最底下一行不能向下
    assert not mask[
        MOVE_BASE + Direction.DOWN
    ]

    assert not mask[
        MOVE_BASE + Direction.DOWN_LEFT
    ]

    assert not mask[
        MOVE_BASE + Direction.DOWN_RIGHT
    ]

    # 有弹药，因此可以射击
    for direction in Direction:

        assert mask[
            SHOOT_BASE + direction
        ]

    # 初始弹药已满
    assert not mask[RELOAD]


def test_reload_becomes_legal_after_shoot():

    env = ShotgunKingEnv()

    env.reset(seed=42)

    shoot_action = (
        SHOOT_BASE
        + Direction.LEFT
    )

    env.step(
        shoot_action
    )

    mask = env.action_masks()

    assert mask[RELOAD]

def test_default_env_uses_shaped_reward():

    env = ShotgunKingEnv()

    env.reset(seed=42)

    # 初始 ammo 已满，
    # RELOAD 是非法动作
    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(RELOAD)

    assert reward == pytest.approx(
        -0.21
    )

    assert (
        info["reward_components"]
        ["step_cost"]
        == pytest.approx(-0.01)
    )

    assert (
        info["reward_components"]
        ["illegal_action"]
        == pytest.approx(-0.20)
    )


def test_sparse_env_has_no_intermediate_reward():

    env = ShotgunKingEnv(
        reward_config=(
            RewardConfig.sparse()
        )
    )

    env.reset(seed=42)

    action = (
        SHOOT_BASE
        + Direction.UP
    )

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    # 虽然杀掉 Pawn，
    # sparse reward 不奖励中间事件
    assert reward == pytest.approx(
        0.0
    )

def test_full_mode_keeps_empty_shots():

    env = ShotgunKingEnv(
        action_mode="full"
    )

    env.reset(seed=42)

    mask = env.action_masks()

    assert mask[
        SHOOT_BASE
        + Direction.UP
    ]

    # full 模式下，
    # 即使该方向没有目标，
    # 射击仍然属于合法策略动作。
    assert mask[
        SHOOT_BASE
        + Direction.RIGHT
    ]

def test_pruned_mode_removes_empty_shots():

    env = ShotgunKingEnv(
        action_mode="pruned_shots"
    )

    env.reset(seed=42)

    mask = env.action_masks()

    # 正上方四格处有 Pawn
    assert mask[
        SHOOT_BASE
        + Direction.UP
    ]

    # 这些方向初始没有目标
    assert not mask[
        SHOOT_BASE
        + Direction.DOWN
    ]

    assert not mask[
        SHOOT_BASE
        + Direction.LEFT
    ]

    assert not mask[
        SHOOT_BASE
        + Direction.RIGHT
    ]

    assert not mask[
        SHOOT_BASE
        + Direction.UP_RIGHT
    ]

def test_pruning_does_not_change_core_legality():

    env = ShotgunKingEnv(
        action_mode="pruned_shots"
    )

    env.reset(seed=42)

    empty_shot = (
        SHOOT_BASE
        + Direction.RIGHT
    )

    # Core 规则仍允许向空气射击
    assert (
        empty_shot
        in env.game.legal_actions()
    )

    # 但 Policy mask 不允许探索
    assert not env.action_masks()[
        empty_shot
    ]

def test_relative_geometry_shape():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="random_medium",
        geometry_mode="relative_v1",
    )

    obs, _ = env.reset(
        seed=42
    )

    assert (
        obs["geometry"].shape
        == (25,)
    )

    assert np.all(
        obs["geometry"] >= -1.0
    )

    assert np.all(
        obs["geometry"] <= 1.0
    )

def test_relative_geometry_deterministic():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="random_medium",
        geometry_mode="relative_v1",
    )

    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)

    assert np.allclose(
        obs1["geometry"],
        obs2["geometry"],
    )

def test_relative_geometry_changes_observation():

    env = ShotgunKingEnv(
        layout_mode="random_medium",
        geometry_mode="relative_v1",
    )

    obs, _ = env.reset(
        seed=42
    )

    assert (
        "geometry"
        in env.observation_space.spaces
    )

    assert (
        "geometry"
        in obs
    )

    assert (
        obs["geometry"].shape
        == (25,)
    )

def test_geometry_none_keeps_original_observation():

    env = ShotgunKingEnv(
        layout_mode="random_medium",
        geometry_mode="none",
    )

    obs, _ = env.reset(
        seed=42
    )

    assert (
        "geometry"
        not in env.observation_space.spaces
    )

    assert (
        "geometry"
        not in obs
    )

def test_relative_geometry_survives_win_terminal_state():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="fixed",
        geometry_mode="relative_v1",
    )

    obs, _ = env.reset(
        seed=42
    )

    state = env.game.state

    br, bc = state.player_pos

    # -----------------------------
    # 清除所有白棋
    # -----------------------------

    for piece in (
        Piece.WHITE_KING,
        Piece.WHITE_ROOK,
        Piece.WHITE_PAWN,
    ):

        state.board[
            state.board == piece
        ] = Piece.EMPTY

    # -----------------------------
    # 在黑王正上方两格放 White King
    # 保证能够直接射杀
    # -----------------------------

    state.board[
        br - 2,
        bc,
    ] = Piece.WHITE_KING

    action = (
        SHOOT_BASE
        + int(Direction.UP)
    )

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    assert terminated
    assert not truncated

    assert env.game.state.won

    # White King 已经从 board 删除
    assert not np.any(
        env.game.state.board
        == Piece.WHITE_KING
    )

    # 但 terminal observation
    # 必须继续合法
    assert (
        obs["geometry"].shape
        == (25,)
    )

    assert np.all(
        np.isfinite(
            obs["geometry"]
        )
    )

    assert np.all(
        obs["geometry"] >= -1.0
    )

    assert np.all(
        obs["geometry"] <= 1.0
    )

def test_relative_v2_geometry_shape():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="random_medium",
        geometry_mode="relative_v2",
    )

    obs, _ = env.reset(
        seed=42
    )

    assert (
        obs["geometry"].shape
        == (47,)
    )

def test_relative_v2_survives_win_terminal_state():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="fixed",
        geometry_mode="relative_v2",
    )

    env.reset(seed=42)

    state = env.game.state

    br, bc = state.player_pos

    for piece in (
        Piece.WHITE_KING,
        Piece.WHITE_ROOK,
        Piece.WHITE_PAWN,
    ):
        state.board[
            state.board == piece
        ] = Piece.EMPTY

    state.board[
        br - 2,
        bc,
    ] = Piece.WHITE_KING

    action = (
        SHOOT_BASE
        + int(Direction.UP)
    )

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    assert terminated
    assert not truncated

    assert (
        obs["geometry"].shape
        == (47,)
    )

    assert np.all(
        np.isfinite(
            obs["geometry"]
        )
    )

def test_relative_v3_geometry_shape():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="random_medium",
        geometry_mode="relative_v3",
    )

    obs, _ = env.reset(
        seed=42
    )

    assert (
        obs["geometry"].shape
        == (56,)
    )

    assert np.all(
        np.isfinite(
            obs["geometry"]
        )
    )

def test_relative_v3_detects_rook_threat():

    env = ShotgunKingEnv(
        layout_mode="fixed",
        geometry_mode="relative_v3",
    )

    env.reset(
        seed=42
    )

    state = env.game.state

    board = state.board

    br, bc = (
        state.player_pos
    )

    # 清除白棋
    for piece in (
        Piece.WHITE_KING,
        Piece.WHITE_ROOK,
        Piece.WHITE_PAWN,
    ):

        board[
            board == piece
        ] = Piece.EMPTY

    # White King 保留到一个不攻击的位置
    board[
        0,
        0,
    ] = Piece.WHITE_KING

    # Black King 正上方放 Rook
    board[
        br - 3,
        bc,
    ] = Piece.WHITE_ROOK

    geometry = (
        env._get_geometry_obs_v3()
    )

    # 最后9维：
    # [under_attack, 8 directions]
    threat = geometry[-9:]

    assert threat[0] == 1.0

    # Direction.UP = 0
    # threat[1 + UP]
    assert (
        threat[
            1
            + int(
                Direction.UP
            )
        ]
        == 1.0
    )

def test_relative_v3_detects_pawn_threat():

    env = ShotgunKingEnv(
        layout_mode="fixed",
        geometry_mode="relative_v3",
    )

    env.reset(
        seed=42
    )

    state = env.game.state

    board = state.board

    br, bc = (
        state.player_pos
    )

    for piece in (
        Piece.WHITE_KING,
        Piece.WHITE_ROOK,
        Piece.WHITE_PAWN,
    ):

        board[
            board == piece
        ] = Piece.EMPTY

    board[
        0,
        0,
    ] = Piece.WHITE_KING

    # Pawn 在玩家左上方
    board[
        br - 1,
        bc - 1,
    ] = Piece.WHITE_PAWN

    geometry = (
        env._get_geometry_obs_v3()
    )

    threat = geometry[-9:]

    assert threat[0] == 1.0

    assert (
        threat[
            1
            + int(
                Direction.UP_LEFT
            )
        ]
        == 1.0
    )

def test_relative_v3_no_threat():

    env = ShotgunKingEnv(
        layout_mode="fixed",
        geometry_mode="relative_v3",
    )

    env.reset(
        seed=42
    )

    attackers = (
        env.game
        .get_player_attackers()
    )

    if not attackers:

        geometry = (
            env._get_geometry_obs_v3()
        )

        threat = geometry[-9:]

        assert np.all(
            threat == 0.0
        )


# ==================================================
# relative_v4 : v2 + immediate move danger
# ==================================================


def _set_env_state(
    env,
    board,
    player_pos,
    ammo: int = 2,
):

    state = GameState(
        board=np.asarray(
            board,
            dtype=np.int8,
        ),
        player_pos=player_pos,
        ammo=ammo,
        max_ammo=ammo,
    )

    # Keep the Gym layer snapshot and the Core state in sync so that the
    # geometry helpers and game.legal_actions() describe the same board.
    env.game.state = state
    env._state = state

    return state


def test_relative_v4_geometry_shape():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="random_medium",
        geometry_mode="relative_v4",
    )

    assert (
        env.observation_space
        .spaces["geometry"]
        .shape
        == (55,)
    )

    obs, _ = env.reset(
        seed=42
    )

    assert (
        obs["geometry"].shape
        == (55,)
    )

    assert (
        obs["geometry"].dtype
        == np.float32
    )

    assert np.all(
        np.isfinite(
            obs["geometry"]
        )
    )

    assert np.all(
        obs["geometry"] >= -1.0
    )

    assert np.all(
        obs["geometry"] <= 1.0
    )

    assert env.observation_space.contains(
        obs
    )


def test_relative_v4_v2_prefix_is_unchanged():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="random_medium",
        geometry_mode="relative_v4",
    )

    obs, _ = env.reset(
        seed=42
    )

    v2 = env._get_geometry_obs_v2()

    assert np.allclose(
        obs["geometry"][:47],
        v2,
    )


def test_relative_v4_move_danger_shape_and_range():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="random_medium",
        geometry_mode="relative_v4",
    )

    env.reset(seed=42)

    danger = (
        env._get_move_danger_features()
    )

    assert danger.shape == (8,)

    assert danger.dtype == np.float32

    assert np.all(
        (danger == 0.0)
        | (danger == 1.0)
    )


def test_relative_v4_safe_move_has_zero_danger():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="fixed",
        geometry_mode="relative_v4",
    )

    env.reset(seed=42)

    board = np.zeros(
        (8, 8),
        dtype=np.int8,
    )

    player_pos = (4, 4)

    board[player_pos] = Piece.BLACK_KING

    # White King far away, does not attack (4, 4) or (3, 4).
    board[0, 7] = Piece.WHITE_KING

    _set_env_state(
        env,
        board,
        player_pos,
    )

    danger = (
        env._get_move_danger_features()
    )

    # MOVE_UP -> (3, 4) is legal and safe.
    assert (
        MOVE_BASE + int(Direction.UP)
        in env.game.legal_actions()
    )

    assert (
        danger[int(Direction.UP)]
        == 0.0
    )


def test_relative_v4_rook_move_is_marked_dangerous():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="fixed",
        geometry_mode="relative_v4",
    )

    env.reset(seed=42)

    board = np.zeros(
        (8, 8),
        dtype=np.int8,
    )

    player_pos = (1, 1)

    board[player_pos] = Piece.BLACK_KING

    # Rook to the up-left of the player.
    board[0, 0] = Piece.WHITE_ROOK

    board[7, 7] = Piece.WHITE_KING

    _set_env_state(
        env,
        board,
        player_pos,
    )

    danger = (
        env._get_move_danger_features()
    )

    # MOVE_LEFT -> (1, 0) is legal but sits on the rook's file (0, 0).
    assert (
        MOVE_BASE + int(Direction.LEFT)
        in env.game.legal_actions()
    )

    assert (
        danger[int(Direction.LEFT)]
        == 1.0
    )


def test_relative_v4_pawn_move_is_marked_dangerous():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="fixed",
        geometry_mode="relative_v4",
    )

    env.reset(seed=42)

    board = np.zeros(
        (8, 8),
        dtype=np.int8,
    )

    player_pos = (3, 4)

    board[player_pos] = Piece.BLACK_KING

    # White pawn attacks (4, 2) and (4, 4).
    board[3, 3] = Piece.WHITE_PAWN

    board[0, 7] = Piece.WHITE_KING

    _set_env_state(
        env,
        board,
        player_pos,
    )

    danger = (
        env._get_move_danger_features()
    )

    # MOVE_DOWN -> (4, 4) is legal but attacked by the pawn.
    assert (
        MOVE_BASE + int(Direction.DOWN)
        in env.game.legal_actions()
    )

    assert (
        danger[int(Direction.DOWN)]
        == 1.0
    )


def test_relative_v4_illegal_move_has_zero_danger():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="fixed",
        geometry_mode="relative_v4",
    )

    # Fixed layout puts the Black King on the bottom row.
    env.reset(seed=42)

    danger = (
        env._get_move_danger_features()
    )

    # Moving down / down-left / down-right is off the board.
    for direction in (
        Direction.DOWN,
        Direction.DOWN_LEFT,
        Direction.DOWN_RIGHT,
    ):

        action = (
            MOVE_BASE
            + int(direction)
        )

        assert (
            action
            not in env.game.legal_actions()
        )

        assert (
            danger[int(direction)]
            == 0.0
        )


def test_relative_v4_hypothetical_removes_old_position_blocker():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="fixed",
        geometry_mode="relative_v4",
    )

    env.reset(seed=42)

    board = np.zeros(
        (8, 8),
        dtype=np.int8,
    )

    player_pos = (0, 2)

    board[player_pos] = Piece.BLACK_KING

    # Rook on the same rank as both the old and the new player square.
    board[0, 0] = Piece.WHITE_ROOK

    board[7, 7] = Piece.WHITE_KING

    _set_env_state(
        env,
        board,
        player_pos,
    )

    # If the old Black King were left on the board, it would block the
    # rook's line of sight to (0, 3).
    stale_board = board.copy()

    assert (
        env.game.get_player_attackers(
            board=stale_board,
            player_pos=(0, 3),
        )
        == []
    )

    danger = (
        env._get_move_danger_features()
    )

    # Correct hypothetical board removes the old Black King, so the rook
    # attacks the destination.
    assert (
        MOVE_BASE + int(Direction.RIGHT)
        in env.game.legal_actions()
    )

    assert (
        danger[int(Direction.RIGHT)]
        == 1.0
    )


def test_relative_v4_survives_win_terminal_state():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="fixed",
        geometry_mode="relative_v4",
    )

    env.reset(seed=42)

    state = env.game.state

    br, bc = state.player_pos

    for piece in (
        Piece.WHITE_KING,
        Piece.WHITE_ROOK,
        Piece.WHITE_PAWN,
    ):

        state.board[
            state.board == piece
        ] = Piece.EMPTY

    state.board[
        br - 2,
        bc,
    ] = Piece.WHITE_KING

    action = (
        SHOOT_BASE
        + int(Direction.UP)
    )

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    assert terminated
    assert not truncated

    assert (
        obs["geometry"].shape
        == (55,)
    )

    assert np.all(
        np.isfinite(
            obs["geometry"]
        )
    )

    assert np.all(
        obs["geometry"] >= -1.0
    )

    assert np.all(
        obs["geometry"] <= 1.0
    )