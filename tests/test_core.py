import pytest
import numpy as np

from shotgun_king.core import (
    ShotgunKingLite,
    Piece,
    Direction,
    MOVE_BASE,
    SHOOT_BASE,
)


def test_reset():

    game = ShotgunKingLite()

    state = game.reset(seed=42)

    assert state.player_pos == (7, 4)

    assert state.ammo == 2

    assert (
        state.board[7, 4]
        == Piece.BLACK_KING
    )

    assert (
        state.board[0, 4]
        == Piece.WHITE_KING
    )


def test_move_up_is_legal():

    game = ShotgunKingLite()

    game.reset(seed=42)

    action = (
        MOVE_BASE
        + Direction.UP
    )

    assert action in game.legal_actions()


def test_move_down_is_illegal():

    game = ShotgunKingLite()

    game.reset(seed=42)

    action = (
        MOVE_BASE
        + Direction.DOWN
    )

    assert action not in game.legal_actions()


def test_shoot_consumes_ammo():

    game = ShotgunKingLite()

    game.reset(seed=42)

    action = (
        SHOOT_BASE
        + Direction.LEFT
    )

    state, events = game.step(
        action
    )

    assert state.ammo == 1


def test_shoot_up_hits_pawn():

    game = ShotgunKingLite()

    game.reset(seed=42)

    action = (
        SHOOT_BASE
        + Direction.UP
    )

    state, events = game.step(
        action
    )

    assert (
        state.board[3, 4]
        == Piece.EMPTY
    )

    assert "shot_hit:WHITE_PAWN" in events


def test_illegal_action():

    game = ShotgunKingLite()

    game.reset(seed=42)

    with pytest.raises(ValueError):

        game.step(100)

def test_random_easy_same_seed_same_layout():

    game = ShotgunKingLite(
        layout_mode="random_easy"
    )

    state1 = game.reset(
        seed=123
    )

    board1 = state1.board.copy()

    state2 = game.reset(
        seed=123
    )

    board2 = state2.board.copy()

    assert np.array_equal(
        board1,
        board2,
    )

def test_random_easy_piece_counts():

    game = ShotgunKingLite(
        layout_mode="random_easy"
    )

    state = game.reset(
        seed=123
    )

    board = state.board

    assert (
        np.sum(
            board
            == Piece.BLACK_KING
        )
        == 1
    )

    assert (
        np.sum(
            board
            == Piece.WHITE_KING
        )
        == 1
    )

    assert (
        np.sum(
            board
            == Piece.WHITE_ROOK
        )
        == 2
    )

    assert (
        np.sum(
            board
            == Piece.WHITE_PAWN
        )
        == 3
    )

def test_random_easy_position_ranges():

    game = ShotgunKingLite(
        layout_mode="random_easy"
    )

    state = game.reset(
        seed=123
    )

    br, bc = state.player_pos

    assert br == 7
    assert 1 <= bc <= 6

    king_pos = np.argwhere(
        state.board
        == Piece.WHITE_KING
    )[0]

    assert king_pos[0] == 0
    assert 1 <= king_pos[1] <= 6

    pawn_positions = np.argwhere(
        state.board
        == Piece.WHITE_PAWN
    )

    assert np.all(
        pawn_positions[:, 0] == 3
    )

def test_random_easy_has_layout_diversity():

    game = ShotgunKingLite(
        layout_mode="random_easy"
    )

    layouts = set()

    for seed in range(20):

        state = game.reset(
            seed=seed
        )

        layouts.add(
            state.board.tobytes()
        )

    assert len(layouts) >= 10

def test_fixed_layout_unchanged():

    game = ShotgunKingLite(
        layout_mode="fixed"
    )

    state = game.reset(
        seed=42
    )

    assert (
        state.player_pos
        == (7, 4)
    )

    assert (
        state.board[0, 4]
        == Piece.WHITE_KING
    )

    assert (
        state.board[3, 2]
        == Piece.WHITE_PAWN
    )

    assert (
        state.board[3, 4]
        == Piece.WHITE_PAWN
    )

    assert (
        state.board[3, 6]
        == Piece.WHITE_PAWN
    )

def test_random_medium_reproducible():

    game = ShotgunKingLite(
        layout_mode="random_medium"
    )

    state1 = game.reset(seed=123)
    board1 = state1.board.copy()

    state2 = game.reset(seed=123)
    board2 = state2.board.copy()

    assert np.array_equal(
        board1,
        board2,
    )

def test_random_medium_piece_ranges():

    game = ShotgunKingLite(
        layout_mode="random_medium"
    )

    for seed in range(50):

        state = game.reset(
            seed=seed
        )

        board = state.board

        assert np.sum(
            board
            == Piece.BLACK_KING
        ) == 1

        assert np.sum(
            board
            == Piece.WHITE_KING
        ) == 1

        n_rooks = np.sum(
            board
            == Piece.WHITE_ROOK
        )

        n_pawns = np.sum(
            board
            == Piece.WHITE_PAWN
        )

        assert 1 <= n_rooks <= 2
        assert 2 <= n_pawns <= 4

def test_random_medium_no_overlap():

    game = ShotgunKingLite(
        layout_mode="random_medium"
    )

    for seed in range(50):

        state = game.reset(
            seed=seed
        )

        board = state.board

        occupied = np.sum(
            board != Piece.EMPTY
        )

        n_rooks = np.sum(
            board == Piece.WHITE_ROOK
        )

        n_pawns = np.sum(
            board == Piece.WHITE_PAWN
        )

        expected = (
            1       # black king
            + 1     # white king
            + n_rooks
            + n_pawns
        )

        assert occupied == expected

