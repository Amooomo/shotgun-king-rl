import argparse
from pathlib import Path

from sb3_contrib import (
    MaskablePPO,
)

from shotgun_king.callbacks import (
    WinRateEvalCallback,
)

from stable_baselines3.common.monitor import (
    Monitor,
)

from shotgun_king.env import (
    ShotgunKingEnv,
)

from shotgun_king.reward import (
    RewardConfig,
)

from shotgun_king.features import (
    ShotgunCNNExtractor,
    ShotgunSmallCNNExtractor,
    ShotgunUNetLiteExtractor,
    ShotgunCoordCNNExtractor,
    ShotgunTransformerExtractor,
    ShotgunPieceTransformerExtractor,
)


def get_reward_config(
    name: str,
):

    if name == "sparse":
        return RewardConfig.sparse()

    if name == "shaped":
        return RewardConfig.shaped()

    if name == "shaped_v2":
        return RewardConfig.shaped_v2()

    raise ValueError(
        f"Unknown reward config: {name}"
    )


def make_env(
    reward_config,
    monitor_path=None,
    action_mode="full",
    layout_mode="fixed",
    geometry_mode="none",
):

    env = ShotgunKingEnv(
        reward_config=reward_config,
        action_mode=action_mode,
        layout_mode=layout_mode,
        geometry_mode=geometry_mode,
    )

    env = Monitor(
        env,
        filename=monitor_path,
        info_keywords=(
            "is_success",
        ),
    )

    if monitor_path is not None:

        env = Monitor(
            env,
            filename=monitor_path,
            info_keywords=(
                "is_success",
            ),
        )

    return env


def main():

    parser = argparse.ArgumentParser()

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
        "--timesteps",
        type=int,
        default=100_000,
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
        "--init-model",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--init-mode",
        choices=[
            "continue",
            "weights_only",
        ],
        default="continue",
    )

    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
    )

    parser.add_argument(
        "--feature-mode",
        choices=[
            "flat",
            "cnn",
            "cnn_small",
            "unet_lite",
            "coord_cnn",
            "transformer",
            "piece_transformer",
        ],
        default="flat",
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

    reward_name = args.reward

    reward_config = (
        get_reward_config(
            reward_name
        )
    )

    # -----------------------------
    # 目录
    # -----------------------------

    run_name = (
        f"{reward_name}_"
        f"{args.action_mode}"
        f"_{args.layout_mode}"
        f"_{args.feature_mode}"
        f"_{args.geometry_mode}"
        f"_seed{args.seed}"
    )

    if args.run_tag:
        run_name += (
            f"_{args.run_tag}"
        )
    
    model_dir = Path(
        "models"
    ) / run_name

    log_dir = Path(
        "logs"
    ) / run_name

    tensorboard_dir = (
        Path("logs")
        / "tensorboard"
        / run_name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tensorboard_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------
    # 训练环境
    # -----------------------------

    train_env = make_env(
        reward_config,
        monitor_path=str(
            log_dir / "train"
        ),
        action_mode=args.action_mode,
        layout_mode=args.layout_mode,
        geometry_mode=args.geometry_mode,
    )

    # -----------------------------
    # 独立 evaluation env
    # -----------------------------

    eval_env = ShotgunKingEnv(
        reward_config=reward_config,
        action_mode=args.action_mode,
        layout_mode=args.layout_mode,
        geometry_mode=args.geometry_mode,
    )

    # -----------------------------
    # Evaluation callback
    # -----------------------------

    eval_callback = (
        WinRateEvalCallback(
            eval_env=eval_env,

            eval_freq=5000,

            n_eval_episodes=100,

            best_model_save_path=str(
                model_dir
            ),

            eval_seed=20000,

            verbose=1,
        )
    )

    # -----------------------------
    # feature extractor
    # -----------------------------
    
    policy_kwargs = None

    if args.feature_mode == "cnn":

        policy_kwargs = {
            "features_extractor_class":
                ShotgunCNNExtractor,

            "features_extractor_kwargs": {
                "features_dim": 128,
            },
        }

    elif args.feature_mode == "cnn_small":

        policy_kwargs = {
            "features_extractor_class":
                ShotgunSmallCNNExtractor,

            "features_extractor_kwargs": {
                "features_dim": 128,
            },
        }

    elif args.feature_mode == "unet_lite":

        policy_kwargs = {

            "features_extractor_class":
                ShotgunUNetLiteExtractor,

            "features_extractor_kwargs": {
                "features_dim": 128,
            },
        }

    elif (
        args.feature_mode
        == "coord_cnn"
    ):

        policy_kwargs = {

            "features_extractor_class":
                ShotgunCoordCNNExtractor,

            "features_extractor_kwargs": {
                "features_dim": 128,
            },
        }

    elif (
        args.feature_mode
        == "transformer"
    ):

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

    elif (
        args.feature_mode
        == "piece_transformer"
    ):

        policy_kwargs = {

            "features_extractor_class":
                ShotgunPieceTransformerExtractor,

            "features_extractor_kwargs": {

                "features_dim": 128,

                "d_model": 64,

                "nhead": 4,

                "num_layers": 2,

                "dim_feedforward": 128,
            },
        }

    # -----------------------------
    # PPO
    # -----------------------------

    if args.init_model is None:

        model = MaskablePPO(
            policy="MultiInputPolicy",

            env=train_env,

            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,

            gamma=0.99,
            gae_lambda=0.95,

            clip_range=0.2,

            ent_coef=0.01,

            policy_kwargs=policy_kwargs,

            seed=args.seed,

            verbose=1,

            tensorboard_log=str(
                tensorboard_dir
            ),
        )

    elif(args.init_model is not None and args.init_mode == "continue"):

        print(
            "Loading initial model:",
            args.init_model,
        )

        model = MaskablePPO.load(
            args.init_model,
            env=train_env,
            tensorboard_log=str(
                tensorboard_dir
            ),
        )

    elif (args.init_model is not None and args.init_mode == "weights_only"):

        print(
            "Warm-starting policy weights from:",
            args.init_model,
        )

        # ① 全新 PPO
        model = MaskablePPO(
            policy="MultiInputPolicy",

            env=train_env,

            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,

            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,

            seed=args.seed,

            verbose=1,

            tensorboard_log=str(
                tensorboard_dir
            ),
        )

        # ② 加载旧模型
        source_model = (
            MaskablePPO.load(
                args.init_model
            )
        )

        # ③ 只复制网络参数
        model.policy.load_state_dict(
            source_model.policy.state_dict()
        )

        print(
            "Policy weights loaded."
        )

    model.learn(
        total_timesteps=args.timesteps,

        callback=eval_callback,

        tb_log_name=(
            f"ppo_{reward_name}"
        ),
    )

    final_path = (
        model_dir
        / "final_model"
    )

    model.save(
        str(final_path)
    )

    train_env.close()
    eval_env.close()

    print(
        "\nTraining finished."
    )

    print(
        "Final model:",
        final_path,
    )


if __name__ == "__main__":
    main()