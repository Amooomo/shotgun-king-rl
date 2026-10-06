# relative_v4 Multi-seed + Mechanism Analysis

> 任务：`relative_v4` seed 1/2/3 正式 300k 复现 + 4-seed 汇总 + immediate-danger MOVE 机制统计 + matched-seed v2 对照。
> 不修改 reward / PPO / action mask / layout / observation 定义 / core attack rules。

---

## 1. 配置

```text
reward        = shaped_v2
action_mode   = pruned_shots
layout_mode   = random_medium
feature_mode  = flat
geometry_mode = relative_v4
timesteps     = 300000
seeds         = 1 / 2 / 3 / 42 (seed42 复用已有正式 run)
```

Independent evaluation / mechanism analysis：

```text
episodes = 1000
eval seed = 10000
deterministic
```

---

## 2. 表 1：relative_v4 4-seed performance

独立 1000 episodes（best_win_model，eval seed 10000）：

| Seed | Win | Death | Timeout | Mean Return | Mean Length | Callback best | Callback 300k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 72.0% | 27.8% | 0.2% | 11.415 | 6.18 | 80% | 77% |
| 2 | 72.9% | 26.9% | 0.2% | 11.738 | 6.07 | 78% | 73% |
| 3 | 71.2% | 28.5% | 0.3% | 11.243 | 6.21 | 78% | 74% |
| 42 | 74.5% | 25.3% | 0.2% | 12.266 | 6.60 | 77% | 76% |
| **Mean ± SD(ddof=1)** | **72.65% ± 1.42pp** | **27.13% ± 1.38pp** | **0.225% ± 0.05pp** | **11.665 ± 0.450** | **6.263 ± 0.233** | — | — |

- 4 个 seed 全部 ≥ 71.2%，seed42 (74.5%) 略高但在 1.3 SD 内，**不是孤立高点**。
- Callback best 78–80%；independent 与 callback 差 2–5pp，方向一致（seed42 已核对 `best_win_model`）。

---

## 3. 表 2：relative_v4 4-seed mechanism

（分母均为 decision states；v4 各 seed 约 6100–6600 个决策状态）

| Seed | Danger-state exposure | Dangerous-move share | **Dangerous selection rate** | Danger rate given MOVE in danger state | Overall dangerous action rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.4981 | 0.1607 | **0.0075** | 0.0106 | 0.0037 |
| 2 | 0.5139 | 0.1716 | **0.0022** | 0.0033 | 0.0012 |
| 3 | 0.4769 | 0.1524 | **0.0054** | 0.0077 | 0.0026 |
| 42 | 0.4646 | 0.1524 | **0.0033** | 0.0047 | 0.0015 |
| **Mean ± SD** | **0.4884 ± 0.0219** | **0.1593 ± 0.0091** | **0.004596 ± 0.002327** | **0.006577 ± 0.003231** | **0.002242 ± 0.001157** |

原始计数 sanity（以 seed1 为例）：

```text
legal_moves_total                 40373
dangerous_legal_moves_total        6489   <= legal_moves_total
selected_dangerous_moves             23   <= states_with_dangerous_legal_move (3077)
selected_moves_in_danger_states    2178   <= states_with_dangerous_legal_move (3077)
```

---

## 4. 表 3：v2 vs v4 matched-seed mechanism comparison

matched seeds = {1, 2, 3, 42}（v2 使用同一新版 analyzer、同 geometry/env、同 eval seed）。

### 4.1 性能（independent 1000）

| Seed | v2 Win | v4 Win | ΔWin | v2 Death | v4 Death | ΔDeath |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 59.0% | 72.0% | +13.0pp | 40.9% | 27.8% | −13.1pp |
| 2 | 57.3% | 72.9% | +15.6pp | 41.7% | 26.9% | −14.8pp |
| 3 | 58.6% | 71.2% | +12.6pp | 41.3% | 28.5% | −12.8pp |
| 42 | 58.5% | 74.5% | +16.0pp | 41.0% | 25.3% | −15.7pp |
| **mean paired** | **58.35%** | **72.65%** | **+14.30pp** | **41.23%** | **27.13%** | **−14.10pp** |

v2 4-seed：Win 58.35% ± 0.73pp；v4 4-seed：Win 72.65% ± 1.42pp → **均值大幅提升，且 SD 仍很小**。

### 4.2 机制

| Metric | v2 Mean ± SD | v4 Mean ± SD | Δ mean paired |
| --- | --- | --- | --- |
| dangerous_state_exposure_rate | 0.3482 ± 0.0223 | 0.4884 ± 0.0219 | +0.1402 |
| dangerous_move_share | 0.1079 ± 0.0078 | 0.1593 ± 0.0091 | +0.0514 |
| **dangerous_selection_rate** | 0.08261 ± 0.00452 | **0.004596 ± 0.00233** | **−0.0780** |
| dangerous_rate_given_move_in_danger_state | 0.14531 ± 0.00496 | 0.006577 ± 0.003231 | −0.1387 |
| overall_dangerous_action_rate | 0.028745 ± 0.002001 | 0.002242 ± 0.001157 | −0.0265 |

逐 seed 的 `dangerous_selection_rate` Δ（v4−v2）：seed1 −0.0794、seed2 −0.0748、seed3 −0.0754、seed42 −0.0824 —— **四个 seed 全部大幅下降且方向一致**。

### 4.3 解读注意点

- v4 的 exposure / share **更高**：v4 策略会进入/面对更多 immediate-danger 局面（约 49% 决策状态存在危险合法 MOVE，v2 约 35%）。
- 但 v4 在这些局面下实际选择 dangerous MOVE 的概率从 ~8.3% 降到 ~0.46%（约 18×），且一旦决定 MOVE，踩坑率从 ~14.5% 降到 ~0.66%。
- 因此机制证据支持的是「**在同样甚至更危险的决策环境中，v4 更少实际选择 immediate-danger MOVE**」，而不是「v4 让局面更安全」。

---

## 5. 审阅结论（review）

**PASS WITH ISSUES（均为 Minor，无 Blocker / Major）**

- 代码正确性：统计发生在 `env.step` 之前的 decision state；danger 复用 `env._get_move_danger_features()`（与 v4 observation 同源，无重复攻击逻辑）；`legal ∩ danger`；action index 与 `Direction` 顺序一致（有测试保护）。
- 测试：`pytest -q` → **77 passed**（+10 mechanism tests）。
- 复现：seed1/2/3 均严格使用规定配置，`best_win_model` 做正式比较，callback eval seed=20000，independent/mechanism eval seed=10000、deterministic。
- 机制 sanity：所有 rate ∈ [0,1]；全部计数不变量成立。
- 统计：以 4 个 training seed 为单位，SD 用 ddof=1。

### Minor issues

1. v2 seed42 的 checkpoint 目录名为 `..._relative_v2_seed42_relv2_smoke`，命名有误导性。实测 200 episodes Win 59.5%、机制/性能与 seed1/2/3 的 v2 家族一致，判定可用；但严格说 seed42 的 v2 训练口径缺少命名层面的明确证据。
2. Callback 与 independent 最多相差 ~5pp（seed3：74% vs 71.2%），源于 eval seed 不同（20000 vs 10000）与 eval episode 数不同；不构成矛盾。
3. `selected_move_actions` 对非 PPO agent 会把任何 MOVE action 计入（本任务机制分析全部用 PPO + action mask，不影响结论）。

---

## 6. 机器可读文件

```text
results/relative_v4_mechanism/
├── relative_v4_seed1_eval10000.json
├── relative_v4_seed2_eval10000.json
├── relative_v4_seed3_eval10000.json
├── relative_v4_seed42_eval10000.json
├── relative_v2_seed1_eval10000.json
├── relative_v2_seed2_eval10000.json
├── relative_v2_seed3_eval10000.json
├── relative_v2_seed42_eval10000.json
├── seed_metrics.csv
├── summary.csv
├── matched_delta.csv
└── summary.json
```

生成方式：`python scripts/analyze_agent.py ... --output-json <path>`，再用
`python scripts/summarize_mechanism.py --input-dir results/relative_v4_mechanism` 汇总。

---

## 7. 结论与下一步

1. seed42 74.5% **不是偶然高点**：4 seed 均 71–75%，mean 72.65% ± 1.42pp。
2. v4 **跨 seed 稳定优于 v2**：逐 seed ΔWin +12.6 ~ +16.0pp，全部为正。
3. dangerous move selection **显著下降**：8.26% → 0.46%，四个 seed 一致。
4. 机制证据与「immediate-safety」假设**一致**（注意措辞：行为证据一致，不等于证明唯一因果机制）。
5. 是否进入 relative_v5：剩余 Death 约 27%，值得进一步分解；enemy reply risk 是下一候选机制，但需等本审阅通过后再决定，本任务不实现 v5。
