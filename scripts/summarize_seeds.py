import numpy as np


win_rates = [
    # 后面填你的真实结果
    0.989,
    0.992,
    0.993,
    0.993,
]

mean = np.mean(win_rates)

std = np.std(
    win_rates,
    ddof=1,
)

print(
    f"Win rate: "
    f"{mean:.2%} ± {std:.2%}"
)

print(
    f"Min: {np.min(win_rates):.2%}"
)

print(
    f"Max: {np.max(win_rates):.2%}"
)