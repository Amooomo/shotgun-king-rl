from sb3_contrib import MaskablePPO

from shotgun_king.env import (
    ShotgunKingEnv,
)

from shotgun_king.features import (
    ShotgunSmallCNNExtractor,
)


env = ShotgunKingEnv(
    action_mode="pruned_shots",
    layout_mode="random_medium",
)

model = MaskablePPO(
    "MultiInputPolicy",
    env,
    policy_kwargs={
        "features_extractor_class":
            ShotgunSmallCNNExtractor,

        "features_extractor_kwargs": {
            "features_dim": 128,
        },
    },
)

total = sum(
    p.numel()
    for p in model.policy.parameters()
)

trainable = sum(
    p.numel()
    for p in model.policy.parameters()
    if p.requires_grad
)

print(
    "Total parameters:",
    total,
)

print(
    "Trainable parameters:",
    trainable,
)

print(
    model.policy
)