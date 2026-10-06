import torch

from shotgun_king.env import (
    ShotgunKingEnv,
)

from sb3_contrib import MaskablePPO

from shotgun_king.features import (
    ShotgunCNNExtractor,
    ShotgunSmallCNNExtractor,
    ShotgunUNetLiteExtractor,
    ShotgunCoordCNNExtractor,
    ShotgunTransformerExtractor,
    ShotgunPieceTransformerExtractor,
)


def test_cnn_feature_shape():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="random_medium",
    )

    obs, _ = env.reset(
        seed=42
    )

    extractor = (
        ShotgunCNNExtractor(
            env.observation_space,
            features_dim=128,
        )
    )

    batched_obs = {
        key: torch.as_tensor(
            value
        ).unsqueeze(0)
        for key, value in obs.items()
    }

    features = extractor(
        batched_obs
    )

    assert features.shape == (
        1,
        128,
    )

def test_unet_lite_feature_shape():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="random_medium",
    )

    obs, _ = env.reset(
        seed=42
    )

    extractor = (
        ShotgunUNetLiteExtractor(
            env.observation_space,
            features_dim=128,
        )
    )

    batched_obs = {

        key: torch.as_tensor(
            value
        ).unsqueeze(0)

        for key, value
        in obs.items()
    }

    features = extractor(
        batched_obs
    )

    assert features.shape == (
        1,
        128,
    )

    assert torch.isfinite(
        features
    ).all()

def test_coord_cnn_feature_shape():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="random_medium",
    )

    obs, _ = env.reset(
        seed=42
    )

    extractor = (
        ShotgunCoordCNNExtractor(
            env.observation_space,
            features_dim=128,
        )
    )

    batched_obs = {

        key: torch.as_tensor(
            value
        ).unsqueeze(0)

        for key, value
        in obs.items()
    }

    features = extractor(
        batched_obs
    )

    assert features.shape == (
        1,
        128,
    )

    assert torch.isfinite(
        features
    ).all()

def test_transformer_feature_shape():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="random_medium",
    )

    obs, _ = env.reset(
        seed=42
    )

    extractor = (
        ShotgunTransformerExtractor(
            env.observation_space,
            features_dim=128,
        )
    )

    batched_obs = {

        key: torch.as_tensor(
            value
        ).unsqueeze(0)

        for key, value in obs.items()
    }

    features = extractor(
        batched_obs
    )

    assert features.shape == (
        1,
        128,
    )

    assert torch.isfinite(
        features
    ).all()

def test_transformer_model_save_load(
    tmp_path,
):

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="random_medium",
    )

    policy_kwargs = {

        "features_extractor_class":
            ShotgunTransformerExtractor,

        "features_extractor_kwargs": {

            "features_dim": 128,

            "d_model": 64,

            "nhead": 4,

            "num_layers": 2,

            "dim_feedforward": 128,
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
        / "transformer_test"
    )

    model.save(
        str(path)
    )

    loaded = MaskablePPO.load(
        str(path),
        env=env,
    )

    assert loaded is not None

def test_piece_transformer_token_counts():

    env = ShotgunKingEnv(
        action_mode="pruned_shots",
        layout_mode="random_medium",
    )

    obs, _ = env.reset(
        seed=42
    )

    extractor = (
        ShotgunPieceTransformerExtractor(
            env.observation_space
        )
    )

    board = torch.as_tensor(
        obs["board"]
    ).unsqueeze(0)

    (
        piece_ids,
        rows,
        cols,
        valid,
    ) = extractor._extract_piece_tokens(
        board.float()
    )

    assert valid.shape == (
        1,
        8,
    )

    # BK / WK 必须存在
    assert valid[0, 0]
    assert valid[0, 1]

    # 非 PAD 数量应该等于棋盘棋子数量
    expected = int(
        obs["board"].sum()
    )

    assert int(
        valid.sum()
    ) == expected