from shotgun_king.env import (
    ShotgunKingEnv,
)


def main():

    env = ShotgunKingEnv(
        render_mode="ansi",
    )

    obs, info = env.reset(
        seed=42
    )

    print(env.render())

    terminated = False
    truncated = False

    while not (
        terminated
        or truncated
    ):

        legal_actions = (
            info["legal_actions"]
        )

        action = int(
            env.np_random.choice(
                legal_actions
            )
        )

        print(
            "\naction =",
            action,
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        print(
            "reward =",
            reward,
        )

        print(
            "events =",
            info["events"],
        )

        print(
            env.render()
        )

    print()

    if terminated:
        print(
            "Environment terminated."
        )

    if truncated:
        print(
            "Environment truncated."
        )

    env.close()


if __name__ == "__main__":
    main()