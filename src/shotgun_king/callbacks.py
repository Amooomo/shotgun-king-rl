from pathlib import Path

import numpy as np

from stable_baselines3.common.callbacks import (
    BaseCallback,
)

from sb3_contrib.common.maskable.utils import (
    get_action_masks,
)


class WinRateEvalCallback(
    BaseCallback
):

    def __init__(
        self,
        eval_env,
        eval_freq: int = 5000,
        n_eval_episodes: int = 100,
        best_model_save_path: str = "models",
        eval_seed: int = 20000,
        verbose: int = 1,
    ):

        super().__init__(verbose)

        self.eval_env = eval_env

        self.eval_freq = eval_freq

        self.n_eval_episodes = (
            n_eval_episodes
        )

        self.best_model_save_path = (
            Path(best_model_save_path)
        )

        self.eval_seed = eval_seed

        self.best_win_rate = -1.0

        self.best_mean_return = (
            -float("inf")
        )

        self.best_model_save_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _evaluate(self):

        wins = 0
        deaths = 0
        timeouts = 0

        returns = []
        lengths = []

        for episode in range(
            self.n_eval_episodes
        ):

            obs, info = (
                self.eval_env.reset(
                    seed=(
                        self.eval_seed
                        + episode
                    )
                )
            )

            terminated = False
            truncated = False

            episode_return = 0.0
            episode_length = 0

            while not (
                terminated
                or truncated
            ):

                mask = (
                    get_action_masks(
                        self.eval_env
                    )
                )

                action, _ = (
                    self.model.predict(
                        obs,
                        action_masks=mask,
                        deterministic=True,
                    )
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.eval_env.step(
                    int(action)
                )

                episode_return += (
                    reward
                )

                episode_length += 1

            if info.get(
                "is_success",
                False,
            ):
                wins += 1

            elif terminated:
                deaths += 1

            elif truncated:
                timeouts += 1

            else:
                raise RuntimeError(
                    "Episode ended without "
                    "terminated or truncated."
                )

            returns.append(
                episode_return
            )

            lengths.append(
                episode_length
            )

        win_rate = (
            wins
            / self.n_eval_episodes
        )

        death_rate = (
            deaths
            / self.n_eval_episodes
        )

        timeout_rate = (
            timeouts
            / self.n_eval_episodes
        )

        mean_return = float(
            np.mean(returns)
        )

        mean_length = float(
            np.mean(lengths)
        )

        return {
            "win_rate": win_rate,
            "death_rate": death_rate,
            "timeout_rate": timeout_rate,
            "mean_return": mean_return,
            "mean_length": mean_length,
        }

    def _on_step(self) -> bool:

        if (
            self.n_calls
            % self.eval_freq
            != 0
        ):
            return True

        result = self._evaluate()

        win_rate = result[
            "win_rate"
        ]

        mean_return = result[
            "mean_return"
        ]

        # -----------------------------
        # TensorBoard
        # -----------------------------

        self.logger.record(
            "eval/win_rate",
            win_rate,
        )

        self.logger.record(
            "eval/death_rate",
            result["death_rate"],
        )

        self.logger.record(
            "eval/timeout_rate",
            result["timeout_rate"],
        )

        self.logger.record(
            "eval/mean_return",
            mean_return,
        )

        self.logger.record(
            "eval/mean_length",
            result["mean_length"],
        )

        # -----------------------------
        # Best model
        # -----------------------------

        better = False

        if (
            win_rate
            > self.best_win_rate
        ):

            better = True

        elif (
            np.isclose(
                win_rate,
                self.best_win_rate,
            )
            and
            mean_return
            > self.best_mean_return
        ):

            better = True

        if better:

            self.best_win_rate = (
                win_rate
            )

            self.best_mean_return = (
                mean_return
            )

            path = (
                self.best_model_save_path
                / "best_win_model"
            )

            self.model.save(
                str(path)
            )

            if self.verbose:

                print(
                    "\n"
                    "New best policy:"
                )

                print(
                    f"Win rate: "
                    f"{win_rate:.2%}"
                )

                print(
                    f"Mean return: "
                    f"{mean_return:.3f}"
                )

        if self.verbose:

            print(
                "\n"
                "===== WinRate Evaluation ====="
            )

            print(
                f"Timesteps: "
                f"{self.num_timesteps}"
            )

            print(
                f"Win: "
                f"{win_rate:.2%}"
            )

            print(
                f"Death: "
                f"{result['death_rate']:.2%}"
            )

            print(
                f"Timeout: "
                f"{result['timeout_rate']:.2%}"
            )

            print(
                f"Mean return: "
                f"{mean_return:.3f}"
            )

        return True