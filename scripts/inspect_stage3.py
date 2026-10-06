import numpy as np

from sb3_contrib.common.maskable.utils import (
    get_action_masks,
)

from shotgun_king.core import (
    Direction,
    MOVE_BASE,
    SHOOT_BASE,
    RELOAD,
)

from shotgun_king.env import (
    ShotgunKingEnv,
)


def action_name(
    action: int,
) -> str:

    if MOVE_BASE <= action < SHOOT_BASE:

        direction = Direction(
            action - MOVE_BASE
        )

        return (
            f"MOVE_{direction.name}"
        )

    if SHOOT_BASE <= action < RELOAD:

        direction = Direction(
            action - SHOOT_BASE
        )

        return (
            f"SHOOT_{direction.name}"
        )

    if action == RELOAD:

        return "RELOAD"

    return "UNKNOWN"


def main():

    env = ShotgunKingEnv()

    obs, info = env.reset(
        seed=42
    )

    print(
        "Observation space:"
    )

    print(
        env.observation_space
    )

    print(
        "\nboard shape:",
        obs["board"].shape,
    )

    print(
        "stats:",
        obs["stats"],
    )

    print(
        "\nGame:"
    )

    print(
        env.render()
    )

    print(
        "\nBoard channels:"
    )

    for channel in range(
        obs["board"].shape[0]
    ):

        print(
            f"\nchannel {channel}"
        )

        print(
            obs["board"][channel]
        )

    mask = get_action_masks(
        env
    )

    print(
        "\nAction Mask:"
    )

    for action, valid in enumerate(
        mask
    ):

        print(
            f"{action:2d}",
            f"{action_name(action):20s}",
            bool(valid),
        )

    valid_actions = np.flatnonzero(
        mask
    )

    print(
        "\nValid actions:",
        valid_actions,
    )


if __name__ == "__main__":
    main()