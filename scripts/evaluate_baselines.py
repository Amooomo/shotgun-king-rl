import argparse

import numpy as np

from shotgun_king.agents import (
    HeuristicAgent,
    RandomAgent,
)

from shotgun_king.env import (
    ShotgunKingEnv,
)

from shotgun_king.reward import (
    RewardConfig,
)


def evaluate_agent(
    agent,
    episodes: int,
    seed: int,
    action_mode: str,
    layout_mode: str,
    geometry_mode: str,
):

    env = ShotgunKingEnv(
        reward_config=(
            RewardConfig.shaped_v2()
        ),
        action_mode=action_mode,
        layout_mode=layout_mode,
        geometry_mode=geometry_mode,
    )

    rng = np.random.default_rng(
        seed
    )

    wins = 0
    deaths = 0
    timeouts = 0

    returns = []
    lengths = []

    for episode in range(
        episodes
    ):

        obs, info = env.reset(
            seed=seed + episode
        )

        terminated = False
        truncated = False

        episode_return = 0.0
        episode_length = 0

        while not (
            terminated
            or truncated
        ):

            action = agent.select_action(
                obs,
                env,
                rng,
            )

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

    env.close()

    print(
        f"\nAgent: "
        f"{agent.__class__.__name__}"
    )

    print(
        f"Episodes: {episodes}"
    )

    print(
        f"Win rate: "
        f"{wins / episodes:.2%}"
    )

    print(
        f"Death rate: "
        f"{deaths / episodes:.2%}"
    )

    print(
        f"Timeout rate: "
        f"{timeouts / episodes:.2%}"
    )

    print(
        f"Mean return: "
        f"{np.mean(returns):.3f}"
    )

    print(
        f"Mean length: "
        f"{np.mean(lengths):.2f}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        ],
        default="none",
    )

    args = parser.parse_args()

    evaluate_agent(
        RandomAgent(),
        episodes=args.episodes,
        seed=args.seed,
        action_mode=args.action_mode,
        layout_mode=args.layout_mode
    )

    evaluate_agent(
        HeuristicAgent(),
        episodes=args.episodes,
        seed=args.seed,
        action_mode=args.action_mode,
        layout_mode=args.layout_mode
    )


if __name__ == "__main__":
    main()