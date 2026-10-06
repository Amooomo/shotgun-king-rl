from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:

    win: float = 0.0
    death: float = 0.0

    kill_rook: float = 0.0
    kill_pawn: float = 0.0

    shot_miss: float = 0.0

    step_cost: float = 0.0

    reload_cost: float = 0.0

    illegal_action: float = 0.0

    timeout: float = 0.0

    # 故意保留，用于后面演示 reward hacking
    survival_bonus: float = 0.0

    @classmethod
    def sparse(cls) -> "RewardConfig":

        return cls(
            win=1.0,
            death=-1.0,
        )

    @classmethod
    def shaped(cls) -> "RewardConfig":

        return cls(
            win=10.0,
            death=-10.0,

            kill_rook=1.0,
            kill_pawn=0.25,

            shot_miss=-0.05,

            step_cost=-0.01,

            illegal_action=-0.20,

            timeout=-1.0,
        )

    @classmethod
    def shaped_v2(cls) -> "RewardConfig":

        return cls(
            win=20.0,

            death=-12.0,

            kill_rook=1.0,
            kill_pawn=0.25,

            shot_miss=-0.05,

            reload_cost=-0.01,

            step_cost=-0.01,

            illegal_action=-0.20,

            timeout=-8.0,
        )

    @classmethod
    def intentionally_bad(cls) -> "RewardConfig":
        """
        故意有漏洞的 Reward。

        后面专门用于演示：
        Agent 为什么可能为了刷生存奖励而不通关。
        """

        return cls(
            win=1.0,
            death=-1.0,

            survival_bonus=0.05,
        )


class RewardCalculator:

    def __init__(
        self,
        config: RewardConfig,
    ):
        self.config = config

    def calculate(
        self,
        *,
        events: list[str],
        won: bool,
        lost: bool,
        truncated: bool,
        illegal_action: bool,
    ) -> tuple[float, dict[str, float]]:

        components: dict[str, float] = {}

        def add(
            name: str,
            value: float,
        ) -> None:

            if value == 0.0:
                return

            components[name] = (
                components.get(name, 0.0)
                + float(value)
            )

        # -----------------------------
        # 每一步都发生的奖励
        # -----------------------------

        add(
            "step_cost",
            self.config.step_cost,
        )

        if illegal_action:

            add(
                "illegal_action",
                self.config.illegal_action,
            )

        # -----------------------------
        # 根据游戏事件奖励
        # -----------------------------

        for event in events:

            if event == "shot_hit:WHITE_ROOK":

                add(
                    "kill_rook",
                    self.config.kill_rook,
                )

            elif event == "shot_hit:WHITE_PAWN":

                add(
                    "kill_pawn",
                    self.config.kill_pawn,
                )

            elif event == "shot_missed":

                add(
                    "shot_miss",
                    self.config.shot_miss,
                )

            elif event == "reload":

                add(
                    "reload_cost",
                    self.config.reload_cost,
                )

        # -----------------------------
        # Episode 结果
        # -----------------------------

        if won:

            add(
                "win",
                self.config.win,
            )

        elif lost:

            add(
                "death",
                self.config.death,
            )

        elif truncated:

            add(
                "timeout",
                self.config.timeout,
            )

        # -----------------------------
        # 故意留下的实验项
        # -----------------------------

        if (
            not won
            and
            not lost
        ):

            add(
                "survival_bonus",
                self.config.survival_bonus,
            )

        total = float(
            sum(components.values())
        )

        return total, components