import numpy as np

from shotgun_king.core import (
    Direction,
    GameState,
    MOVE_BASE,
    Piece,
)

from shotgun_king.env import (
    ShotgunKingEnv,
)

from shotgun_king.mechanism import (
    MoveDangerDiagnostics,
    MoveDangerState,
    safe_ratio,
)


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

    env.game.state = state
    env._state = state

    return state


def _blank_board():

    return np.zeros(
        (8, 8),
        dtype=np.int8,
    )


def test_safe_legal_move_is_not_counted():

    # relative_v2 env: the diagnostic must not require the v4 observation.
    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="fixed",
        geometry_mode="relative_v2",
    )

    env.reset(seed=42)

    board = _blank_board()

    player_pos = (4, 4)
    board[player_pos] = Piece.BLACK_KING
    board[0, 7] = Piece.WHITE_KING

    _set_env_state(env, board, player_pos)

    diagnostics = MoveDangerDiagnostics()

    state = diagnostics.observe(
        env.action_masks(),
        env._get_move_danger_features(),
    )

    up = MOVE_BASE + int(Direction.UP)

    assert up in state.legal_move_actions
    assert state.dangerous_move_actions == ()
    assert not state.has_dangerous_legal_move

    diagnostics.select(state, up)

    assert diagnostics.decision_states_total == 1
    assert diagnostics.states_with_legal_move == 1
    assert diagnostics.states_with_dangerous_legal_move == 0
    assert diagnostics.selected_move_actions == 1
    assert diagnostics.selected_dangerous_moves == 0
    assert diagnostics.selected_moves_in_danger_states == 0


def test_rook_immediate_danger_move_is_counted():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="fixed",
        geometry_mode="relative_v2",
    )

    env.reset(seed=42)

    board = _blank_board()

    player_pos = (1, 1)
    board[player_pos] = Piece.BLACK_KING
    board[0, 0] = Piece.WHITE_ROOK
    board[7, 7] = Piece.WHITE_KING

    _set_env_state(env, board, player_pos)

    diagnostics = MoveDangerDiagnostics()

    state = diagnostics.observe(
        env.action_masks(),
        env._get_move_danger_features(),
    )

    left = MOVE_BASE + int(Direction.LEFT)

    assert left in state.legal_move_actions
    assert left in state.dangerous_move_actions
    assert (
        diagnostics.dangerous_legal_moves_total
        == len(state.dangerous_move_actions)
    )
    assert (
        diagnostics.dangerous_legal_moves_total
        <= diagnostics.legal_moves_total
    )

    diagnostics.select(state, left)

    assert diagnostics.states_with_dangerous_legal_move == 1
    assert diagnostics.selected_move_actions == 1
    assert diagnostics.selected_dangerous_moves == 1
    assert diagnostics.selected_moves_in_danger_states == 1


def test_pawn_immediate_danger_move_is_counted():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="fixed",
        geometry_mode="relative_v2",
    )

    env.reset(seed=42)

    board = _blank_board()

    player_pos = (3, 4)
    board[player_pos] = Piece.BLACK_KING
    board[3, 3] = Piece.WHITE_PAWN
    board[0, 7] = Piece.WHITE_KING

    _set_env_state(env, board, player_pos)

    diagnostics = MoveDangerDiagnostics()

    state = diagnostics.observe(
        env.action_masks(),
        env._get_move_danger_features(),
    )

    down = MOVE_BASE + int(Direction.DOWN)

    assert down in state.legal_move_actions
    assert down in state.dangerous_move_actions

    diagnostics.select(state, down)

    assert diagnostics.selected_dangerous_moves == 1


def test_illegal_move_is_not_counted_even_if_blocked():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="fixed",
        geometry_mode="relative_v2",
    )

    env.reset(seed=42)

    board = _blank_board()

    player_pos = (4, 4)
    board[player_pos] = Piece.BLACK_KING

    # A white pawn occupies the cell directly above the Black King, so
    # MOVE_UP is illegal (destination not empty).
    board[3, 4] = Piece.WHITE_PAWN
    board[0, 7] = Piece.WHITE_KING

    _set_env_state(env, board, player_pos)

    diagnostics = MoveDangerDiagnostics()

    danger = env._get_move_danger_features()

    state = diagnostics.observe(
        env.action_masks(),
        danger,
    )

    up = MOVE_BASE + int(Direction.UP)

    assert up not in state.legal_move_actions
    assert up not in state.dangerous_move_actions

    # Illegal MOVE danger is always 0.0.
    assert float(danger[int(Direction.UP)]) == 0.0


def test_v4_danger_matches_observation_last_eight_dims():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="fixed",
        geometry_mode="relative_v4",
    )

    env.reset(seed=42)

    board = _blank_board()

    player_pos = (0, 2)
    board[player_pos] = Piece.BLACK_KING
    board[0, 0] = Piece.WHITE_ROOK
    board[7, 7] = Piece.WHITE_KING

    _set_env_state(env, board, player_pos)

    geometry = env._get_geometry_obs_v4()

    danger = env._get_move_danger_features()

    assert geometry.shape == (55,)
    assert danger.shape == (8,)

    assert np.allclose(
        geometry[-8:],
        danger,
    )

    right = MOVE_BASE + int(Direction.RIGHT)

    assert float(danger[int(Direction.RIGHT)]) == 1.0

    diagnostics = MoveDangerDiagnostics()

    state = diagnostics.observe(
        env.action_masks(),
        danger,
    )

    assert right in state.dangerous_move_actions


def test_old_position_blocker_regression_for_mechanism():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="fixed",
        geometry_mode="relative_v4",
    )

    env.reset(seed=42)

    board = _blank_board()

    player_pos = (0, 2)
    board[player_pos] = Piece.BLACK_KING
    board[0, 0] = Piece.WHITE_ROOK
    board[7, 7] = Piece.WHITE_KING

    _set_env_state(env, board, player_pos)

    # If the old Black King stayed on the board it would block the rook.
    stale_board = board.copy()

    assert (
        env.game.get_player_attackers(
            board=stale_board,
            player_pos=(0, 3),
        )
        == []
    )

    # The diagnostic must therefore report the destination as dangerous.
    danger = env._get_move_danger_features()

    assert float(danger[int(Direction.RIGHT)]) == 1.0


def test_zero_denominator_is_safe():

    diagnostics = MoveDangerDiagnostics()

    rates = diagnostics.rates()

    for value in rates.values():
        assert value == 0.0

    assert safe_ratio(3, 0) == 0.0
    assert safe_ratio(0, 5) == 0.0
    assert safe_ratio(2, 4) == 0.5

    summary = diagnostics.summary()

    assert (
        summary["dangerous_selection_rate"]
        == 0.0
    )

    assert (
        summary["dangerous_move_share"]
        == 0.0
    )


def test_rates_formula():

    diagnostics = MoveDangerDiagnostics()

    diagnostics.decision_states_total = 10
    diagnostics.states_with_legal_move = 8
    diagnostics.states_with_dangerous_legal_move = 4

    diagnostics.legal_moves_total = 40
    diagnostics.dangerous_legal_moves_total = 8

    diagnostics.selected_move_actions = 50
    diagnostics.selected_dangerous_moves = 2
    diagnostics.selected_moves_in_danger_states = 5

    rates = diagnostics.rates()

    assert rates[
        "dangerous_state_exposure_rate"
    ] == 0.4

    assert rates[
        "dangerous_move_share"
    ] == 0.2

    assert rates[
        "dangerous_selection_rate"
    ] == 0.5

    assert rates[
        "dangerous_rate_given_move_in_danger_state"
    ] == 0.4

    assert rates[
        "overall_dangerous_action_rate"
    ] == 0.2


def test_dangerous_moves_are_subset_of_legal_moves():

    assert MoveDangerState(
        legal_move_actions=(1, 2, 3),
        dangerous_move_actions=(2,),
    ).has_dangerous_legal_move

    assert not MoveDangerState(
        legal_move_actions=(),
        dangerous_move_actions=(),
    ).has_legal_move


def test_select_non_move_does_not_count():

    diagnostics = MoveDangerDiagnostics()

    state = MoveDangerState(
        legal_move_actions=(0,),
        dangerous_move_actions=(0,),
    )

    diagnostics.select(
        state,
        MOVE_BASE + len(Direction),
    )

    assert diagnostics.selected_move_actions == 0
    assert diagnostics.selected_dangerous_moves == 0
    assert diagnostics.selected_moves_in_danger_states == 0
