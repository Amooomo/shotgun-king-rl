import argparse
from collections import Counter, defaultdict

import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import (
    get_action_masks,
)

from shotgun_king.agents import (
    HeuristicAgent,
    RandomAgent,
)

from shotgun_king.core import (
    Direction,
    MOVE_BASE,
    RELOAD,
    SHOOT_BASE,
)

from shotgun_king.env import (
    ShotgunKingEnv,
)

from shotgun_king.reward import (
    RewardConfig,
)


def get_reward_config(name: str):

    if name == "sparse":
        return RewardConfig.sparse()

    if name == "shaped":
        return RewardConfig.shaped()

    if name == "shaped_v2":
        return RewardConfig.shaped_v2()

    raise ValueError(
        f"Unknown reward config: {name}"
    )


def action_group(action: int) -> str:

    if MOVE_BASE <= action < SHOOT_BASE:
        return "MOVE"

    if SHOOT_BASE <= action < RELOAD:
        return "SHOOT"

    if action == RELOAD:
        return "RELOAD"

    return "UNKNOWN"


def action_name(action: int) -> str:

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

    return f"UNKNOWN_{action}"


def choose_action(
    *,
    agent_type,
    model,
    baseline_agent,
    obs,
    env,
    rng,
    stochastic,
):

    if agent_type == "ppo":

        mask = get_action_masks(
            env
        )

        action, _ = model.predict(
            obs,
            action_masks=mask,
            deterministic=not stochastic,
        )

        return int(action)

    return baseline_agent.select_action(
        obs,
        env,
        rng,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        choices=[
            "random",
            "heuristic",
            "ppo",
        ],
        required=True,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
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
        default=10000,
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
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

    if (
        args.agent == "ppo"
        and args.model is None
    ):
        raise ValueError(
            "--model is required "
            "when --agent ppo"
        )

    # ----------------------------------
    # Environment
    # ----------------------------------

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

    # ----------------------------------
    # Agent
    # ----------------------------------

    model = None
    baseline_agent = None

    if args.agent == "ppo":

        # 注意：
        # 这里故意不把 env 传给 load()
        # 我们只是做 prediction
        model = MaskablePPO.load(
            args.model
        )

    elif args.agent == "random":

        baseline_agent = RandomAgent()

    elif args.agent == "heuristic":

        baseline_agent = HeuristicAgent()

    rng = np.random.default_rng(
        args.seed
    )

    # ==================================
    # Global statistics
    # ==================================

    outcomes = Counter()

    action_groups = Counter()

    exact_actions = Counter()

    events = Counter()

    reward_components = Counter()

    episode_returns = []

    episode_lengths = []

    # ----------------------------------
    # 按结局分类统计
    # ----------------------------------

    outcome_action_groups = defaultdict(
        Counter
    )

    outcome_lengths = defaultdict(
        list
    )

    outcome_returns = defaultdict(
        list
    )

    # ==================================
    # Evaluation
    # ==================================

    for episode in range(
        args.episodes
    ):

        obs, info = env.reset(
            seed=args.seed + episode
        )

        terminated = False
        truncated = False

        ep_return = 0.0
        ep_length = 0

        ep_action_groups = Counter()

        while not (
            terminated
            or truncated
        ):

            action = choose_action(
                agent_type=args.agent,
                model=model,
                baseline_agent=baseline_agent,
                obs=obs,
                env=env,
                rng=rng,
                stochastic=args.stochastic,
            )

            group = action_group(
                action
            )

            name = action_name(
                action
            )

            action_groups[group] += 1
            exact_actions[name] += 1

            ep_action_groups[
                group
            ] += 1

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            ep_return += reward
            ep_length += 1

            # --------------------------
            # Events
            # --------------------------

            for event in info[
                "events"
            ]:
                events[event] += 1

            # --------------------------
            # Reward components
            # --------------------------

            for (
                component,
                value,
            ) in info[
                "reward_components"
            ].items():

                reward_components[
                    component
                ] += value

        # ==================================
        # Episode outcome
        # ==================================

        if env.game.state.won:

            outcome = "WIN"

        elif env.game.state.lost:

            outcome = "DEATH"

        else:

            outcome = "TIMEOUT"

        outcomes[outcome] += 1

        episode_returns.append(
            ep_return
        )

        episode_lengths.append(
            ep_length
        )

        outcome_action_groups[
            outcome
        ].update(
            ep_action_groups
        )

        outcome_lengths[
            outcome
        ].append(
            ep_length
        )

        outcome_returns[
            outcome
        ].append(
            ep_return
        )

    # ==================================
    # Derived statistics
    # ==================================

    total_steps = sum(
        action_groups.values()
    )

    total_shots = action_groups[
        "SHOOT"
    ]

    hit_count = sum(
        count
        for event, count
        in events.items()
        if event.startswith(
            "shot_hit:"
        )
    )

    miss_count = events[
        "shot_missed"
    ]

    pawn_kills = events[
        "shot_hit:WHITE_PAWN"
    ]

    rook_kills = events[
        "shot_hit:WHITE_ROOK"
    ]

    king_kills = events[
        "shot_hit:WHITE_KING"
    ]

    # ==================================
    # Print
    # ==================================

    print()
    print("=" * 60)
    print("AGENT BEHAVIOR ANALYSIS")
    print("=" * 60)

    print(
        f"Agent:        {args.agent}"
    )

    if args.model is not None:
        print(
            f"Model:        {args.model}"
        )

    print(
        f"Reward:       {args.reward}"
    )

    print(
        f"Episodes:     {args.episodes}"
    )

    print(
        f"Eval seed:    {args.seed}"
    )

    if args.agent == "ppo":
        print(
            "Policy mode:  "
            + (
                "stochastic"
                if args.stochastic
                else "deterministic"
            )
        )

    # ----------------------------------
    # Outcome
    # ----------------------------------

    print()
    print("----- Outcomes -----")

    for outcome in [
        "WIN",
        "DEATH",
        "TIMEOUT",
    ]:

        count = outcomes[
            outcome
        ]

        rate = (
            count
            / args.episodes
            * 100
        )

        print(
            f"{outcome:8s}: "
            f"{count:5d} "
            f"({rate:6.2f}%)"
        )

    print()
    print(
        "Mean episode return:",
        f"{np.mean(episode_returns):.3f}",
    )

    print(
        "Mean episode length:",
        f"{np.mean(episode_lengths):.2f}",
    )

    # ----------------------------------
    # Action group
    # ----------------------------------

    print()
    print("----- Action Distribution -----")

    for group in [
        "MOVE",
        "SHOOT",
        "RELOAD",
    ]:

        count = action_groups[
            group
        ]

        pct = (
            count
            / total_steps
            * 100
            if total_steps
            else 0.0
        )

        print(
            f"{group:8s}: "
            f"{count:7d} "
            f"({pct:6.2f}%)"
        )

    # ----------------------------------
    # Individual actions
    # ----------------------------------

    print()
    print("----- Exact Action Distribution -----")

    for action in range(17):

        name = action_name(
            action
        )

        count = exact_actions[
            name
        ]

        pct = (
            count
            / total_steps
            * 100
            if total_steps
            else 0.0
        )

        print(
            f"{action:2d} "
            f"{name:22s}"
            f"{count:7d} "
            f"{pct:6.2f}%"
        )

    # ----------------------------------
    # Shooting
    # ----------------------------------

    print()
    print("----- Shooting -----")

    print(
        f"Shots:          {total_shots}"
    )

    print(
        f"Hits:           {hit_count}"
    )

    print(
        f"Misses:         {miss_count}"
    )

    hit_rate = (
        hit_count / total_shots
        if total_shots
        else 0.0
    )

    print(
        f"Hit rate:       "
        f"{hit_rate:.2%}"
    )

    print(
        f"Pawn kills:     {pawn_kills}"
    )

    print(
        f"Rook kills:     {rook_kills}"
    )

    print(
        f"King kills:     {king_kills}"
    )

    # ----------------------------------
    # Reward composition
    # ----------------------------------

    print()
    print("----- Reward Components -----")

    if not reward_components:

        print(
            "(no non-zero reward components)"
        )

    else:

        for (
            component,
            total,
        ) in sorted(
            reward_components.items()
        ):

            mean_per_episode = (
                total
                / args.episodes
            )

            print(
                f"{component:20s}"
                f" total="
                f"{total:10.3f}"
                f"  per_episode="
                f"{mean_per_episode:8.3f}"
            )

    # ----------------------------------
    # Outcome-specific behavior
    # ----------------------------------

    print()
    print(
        "----- Behavior by Outcome -----"
    )

    for outcome in [
        "WIN",
        "DEATH",
        "TIMEOUT",
    ]:

        n = outcomes[
            outcome
        ]

        if n == 0:
            continue

        print()
        print(
            f"[{outcome}] "
            f"episodes={n}"
        )

        print(
            "mean_length =",
            f"{np.mean(outcome_lengths[outcome]):.2f}",
        )

        print(
            "mean_return =",
            f"{np.mean(outcome_returns[outcome]):.3f}",
        )

        for group in [
            "MOVE",
            "SHOOT",
            "RELOAD",
        ]:

            avg = (
                outcome_action_groups[
                    outcome
                ][group]
                / n
            )

            print(
                f"mean_{group.lower():6s}"
                f" = {avg:.2f}"
            )

    env.close()


if __name__ == "__main__":
    main()