import argparse

import numpy as np

from sb3_contrib import (
    MaskablePPO,
)

from sb3_contrib.common.maskable.utils import (
    get_action_masks,
)

from shotgun_king.env import (
    ShotgunKingEnv,
)

from shotgun_king.reward import (
    RewardConfig,
)


def get_reward_config(
    name,
):

    if name == "sparse":
        return RewardConfig.sparse()

    if name == "shaped":
        return RewardConfig.shaped()

    if name == "shaped_v2":
        return RewardConfig.shaped_v2()

    raise ValueError(name)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--reward",
        choices=[
            "sparse",
            "shaped",
            "shaped_v2",
        ],
        default="shaped",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--action-mode",
        choices=[
            "full",
            "pruned_shots",
        ],
        default="full",
    )

    parser.add_argument(
        "--layout-mode",
        choices=[
            "fixed",
            "random_easy",
            "random_medium",
        ],
        default="fixed",
    )

    parser.add_argument(
        "--geometry-mode",
        choices=[
            "none",
            "relative_v1",
            "relative_v2",
            "relative_v3",
            "relative_v4",
        ],
        default="none",
    )

    args = parser.parse_args()

    env = ShotgunKingEnv(
        reward_config=(
            get_reward_config(
                args.reward
            )
        ),
        action_mode=args.action_mode,
        layout_mode=args.layout_mode,
        geometry_mode=args.geometry_mode,
    )

    model = MaskablePPO.load(
        args.model,
        env=env,
    )

    wins = 0
    deaths = 0
    timeouts = 0

    returns = []
    lengths = []

    for episode in range(
        args.episodes
    ):

        obs, info = env.reset(
            seed=args.seed + episode
        )

        terminated = False
        truncated = False

        episode_return = 0.0
        episode_length = 0

        while not (
            terminated
            or truncated
        ):

            masks = get_action_masks(
                env
            )

            action, _ = model.predict(
                obs,
                action_masks=masks,
                deterministic=True,
            )

            action = int(action)

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            episode_return += reward
            episode_length += 1

        if env.game.state.won:
            wins += 1

        elif env.game.state.lost:
            deaths += 1

        else:
            timeouts += 1

        returns.append(
            episode_return
        )

        lengths.append(
            episode_length
        )

    print(
        "\n===== Evaluation ====="
    )

    print(
        "Model:",
        args.model,
    )

    print(
        f"Episodes: {args.episodes}"
    )

    print(
        f"Win rate: "
        f"{wins / args.episodes:.2%}"
    )

    print(
        f"Death rate: "
        f"{deaths / args.episodes:.2%}"
    )

    print(
        f"Timeout rate: "
        f"{timeouts / args.episodes:.2%}"
    )

    print(
        f"Mean return: "
        f"{np.mean(returns):.3f}"
    )

    print(
        f"Mean episode length: "
        f"{np.mean(lengths):.2f}"
    )

    env.close()


if __name__ == "__main__":
    main()