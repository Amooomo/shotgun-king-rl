from sb3_contrib import (
    MaskablePPO,
)

from sb3_contrib.common.maskable.utils import (
    get_action_masks,
)

from shotgun_king.env import (
    ShotgunKingEnv,
)


def main():

    env = ShotgunKingEnv()

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        n_steps=32,
        batch_size=32,
        seed=42,
        verbose=0,
    )

    obs, info = env.reset(
        seed=42
    )

    mask = get_action_masks(
        env
    )

    for _ in range(100):

        action, _ = model.predict(
            obs,
            action_masks=mask,
            deterministic=False,
        )

        action = int(action)

        assert mask[action], (
            f"Invalid action selected: "
            f"{action}"
        )

    print(
        "100 masked predictions passed."
    )


if __name__ == "__main__":
    main()