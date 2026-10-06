import pytest

from shotgun_king.reward import (
    RewardCalculator,
    RewardConfig,
)


def test_sparse_win_reward():

    calculator = RewardCalculator(
        RewardConfig.sparse()
    )

    reward, components = (
        calculator.calculate(
            events=[
                "shot_hit:WHITE_KING",
                "white_king_killed",
            ],
            won=True,
            lost=False,
            truncated=False,
            illegal_action=False,
        )
    )

    assert reward == pytest.approx(
        1.0
    )

    assert components["win"] == 1.0


def test_sparse_normal_step_is_zero():

    calculator = RewardCalculator(
        RewardConfig.sparse()
    )

    reward, components = (
        calculator.calculate(
            events=[
                "player_move:UP"
            ],
            won=False,
            lost=False,
            truncated=False,
            illegal_action=False,
        )
    )

    assert reward == pytest.approx(
        0.0
    )


def test_shaped_pawn_kill():

    calculator = RewardCalculator(
        RewardConfig.shaped()
    )

    reward, components = (
        calculator.calculate(
            events=[
                "shot_hit:WHITE_PAWN"
            ],
            won=False,
            lost=False,
            truncated=False,
            illegal_action=False,
        )
    )

    assert reward == pytest.approx(
        0.24
    )

    assert components["kill_pawn"] == (
        pytest.approx(0.25)
    )

    assert components["step_cost"] == (
        pytest.approx(-0.01)
    )


def test_shaped_miss():

    calculator = RewardCalculator(
        RewardConfig.shaped()
    )

    reward, _ = calculator.calculate(
        events=[
            "shot_missed"
        ],
        won=False,
        lost=False,
        truncated=False,
        illegal_action=False,
    )

    assert reward == pytest.approx(
        -0.06
    )


def test_shaped_illegal_action():

    calculator = RewardCalculator(
        RewardConfig.shaped()
    )

    reward, components = (
        calculator.calculate(
            events=[],
            won=False,
            lost=False,
            truncated=False,
            illegal_action=True,
        )
    )

    assert reward == pytest.approx(
        -0.21
    )

    assert (
        components["illegal_action"]
        == pytest.approx(-0.20)
    )


def test_shaped_win_dominates_piece_rewards():

    config = RewardConfig.shaped()

    max_piece_reward = (
        2 * config.kill_rook
        + 3 * config.kill_pawn
    )

    assert config.win > max_piece_reward

def test_shaped_v2_reload_cost():

    calculator = RewardCalculator(
        RewardConfig.shaped_v2()
    )

    reward, components = (
        calculator.calculate(
            events=["reload"],
            won=False,
            lost=False,
            truncated=False,
            illegal_action=False,
        )
    )

    assert reward == pytest.approx(
        -0.02
    )

    assert components[
        "reload_cost"
    ] == pytest.approx(-0.01)

    assert components[
        "step_cost"
    ] == pytest.approx(-0.01)

def test_shaped_v2_reward_relationship():

    config = RewardConfig.shaped_v2()

    assert config.win > 0

    assert config.timeout < 0

    assert config.death < config.timeout

    max_piece_reward = (
        2 * config.kill_rook
        + 3 * config.kill_pawn
    )

    assert config.win > max_piece_reward

def test_shaped_v2_timeout_loop_not_better_by_large_margin():

    config = RewardConfig.shaped_v2()

    bad_timeout_return = (
        config.timeout
        + 80 * config.step_cost
        + 40 * config.shot_miss
        + 40 * config.reload_cost
    )

    early_death_return = (
        config.death
        + 10 * config.step_cost
    )

    # Timeout 仍应略好于死亡，
    # 但两者不能像 V1 那样差距巨大。
    assert bad_timeout_return > early_death_return

    assert (
        bad_timeout_return
        - early_death_return
        < 2.0
    )