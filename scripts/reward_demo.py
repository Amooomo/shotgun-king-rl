from shotgun_king.core import (
    Direction,
    RELOAD,
    SHOOT_BASE,
)

from shotgun_king.env import (
    ShotgunKingEnv,
)

from shotgun_king.reward import (
    RewardConfig,
)


def show_step(
    env,
    action,
):

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    print(
        "events:",
        info["events"],
    )

    print(
        "reward:",
        reward,
    )

    print(
        "components:",
        info["reward_components"],
    )

    print(
        "episode return:",
        info["episode_return"],
    )

    print()


def main():

    print(
        "========== SHAPED =========="
    )

    env = ShotgunKingEnv(
        reward_config=(
            RewardConfig.shaped()
        )
    )

    env.reset(seed=42)

    # 初始向上开枪：
    # 应该打中 WHITE_PAWN
    show_step(
        env,
        SHOOT_BASE
        + Direction.UP,
    )

    print(
        "======= ILLEGAL ACTION ======="
    )

    env.reset(seed=42)

    # 初始满弹，因此 RELOAD 非法
    show_step(
        env,
        RELOAD,
    )

    print(
        "========== SPARSE =========="
    )

    sparse_env = ShotgunKingEnv(
        reward_config=(
            RewardConfig.sparse()
        )
    )

    sparse_env.reset(seed=42)

    show_step(
        sparse_env,
        SHOOT_BASE
        + Direction.UP,
    )


if __name__ == "__main__":
    main()