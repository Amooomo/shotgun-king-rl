from sb3_contrib import MaskablePPO

from shotgun_king.env import (
    ShotgunKingEnv,
)

from shotgun_king.features import (
    ShotgunCoordCNNExtractor,
)


def test_coord_cnn_model_save_load(
    tmp_path,
):

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="random_medium",
    )

    policy_kwargs = {
        "features_extractor_class":
            ShotgunCoordCNNExtractor,

        "features_extractor_kwargs": {
            "features_dim": 128,
        },
    }

    model = MaskablePPO(
        "MultiInputPolicy",
        env,

        n_steps=32,
        batch_size=32,

        policy_kwargs=policy_kwargs,

        verbose=0,
    )

    path = (
        tmp_path
        / "coord_cnn_test"
    )

    model.save(
        str(path)
    )

    loaded_model = (
        MaskablePPO.load(
            str(path),
            env=env,
        )
    )

    assert loaded_model is not None