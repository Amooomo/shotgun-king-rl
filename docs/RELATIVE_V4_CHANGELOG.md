# Relative-v4 开发变更记录（RELATIVE_V4_CHANGELOG）

> 项目：`shotgun-king-rl`
> 任务：`relative_v4 = relative_v2 + action-conditioned immediate move danger`
> 本文件用于下一个 ChatGPT 会话快速同步。

---

## 1. 修改了哪些文件

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `src/shotgun_king/core.py` | 修改 | 攻击查询统一入口支持 hypothetical board / player_pos |
| `src/shotgun_king/env.py` | 修改 | 新增 move danger 特征、`relative_v4` geometry、dispatcher 与维度 |
| `scripts/train_ppo.py` | 修改 | `--geometry-mode` 增加 `relative_v4` |
| `scripts/evaluate_ppo.py` | 修改 | `--geometry-mode` 增加 `relative_v4` |
| `scripts/analyze_agent.py` | 修改 | `--geometry-mode` 增加 `relative_v4` |
| `scripts/merge_tensorboard_csv.py` | 修改 | run name 解析识别 `relative_v4` |
| `tests/test_core.py` | 修改 | 新增 hypothetical 攻击查询测试 |
| `tests/test_env.py` | 修改 | 新增 relative_v4 行为测试（shape / safe / death / illegal / blocker / terminal） |
| `.gitignore` | 新增 | 忽略 `models/`、`logs/`、`__pycache__` 等大文件 |
| `docs/RELATIVE_V4_CHANGELOG.md` | 新增 | 本文件 |

---

## 2. 每个文件主要改动

### `src/shotgun_king/core.py`

- `get_player_attackers(board=None, player_pos=None)`：
  - `board is None` → 使用 `self.state.board`
  - `player_pos is None` → 使用 `self.state.player_pos`
  - 成为白棋攻击规则的**唯一 source of truth**。
- `_pawn_attacks_player(row, col, player_pos=None)`：允许 hypothetical player 位置。
- `_rook_attacks_player(row, col, board=None, player_pos=None)`：允许 hypothetical board / player 位置。
- `_player_is_attacked()` 接口保留不变，仍然调用 `get_player_attackers()`。
- White King 攻击规则保持标准化判定 `max(dr, dc) == 1`。
- 默认参数为空时行为与旧版本完全一致（由测试覆盖）。

### `src/shotgun_king/env.py`

- `GEOMETRY_DIMS` 增加 `"relative_v4": 55`。
- 构造函数 `geometry_mode` 白名单加入 `relative_v4`。
- `_get_obs()` 的 geometry 分支加入 `relative_v4`。
- 新增 `_get_move_danger_features()`：输出 `(8,) float32`。
- 新增 `_get_geometry_obs_v4()`：`v2(47) + move_danger(8) = 55`。
- `_get_geometry_obs()` dispatcher 增加 `relative_v4 → _get_geometry_obs_v4()`。
- **不调用** `_get_geometry_obs_v3()`，v4 不再继承 v3 的 current-threat 9 dims。

### CLI 脚本

- `train_ppo.py` / `evaluate_ppo.py` / `analyze_agent.py` 的 `--geometry-mode` choices 均加入 `relative_v4`。
- 创建 Env 时均传递 `geometry_mode=args.geometry_mode`（train 的 `make_env` 与 `eval_env`、evaluate、analyze 都已接线）。
- run name 已包含 reward / action_mode / layout_mode / feature_mode / geometry_mode / seed / run_tag。

---

## 3. Core attack-query 如何重构

重构前：

```text
get_player_attackers()
  -> 只读取 self.state.board / self.state.player_pos
_pawn_attacks_player(row, col)
  -> 只读取 self.state.player_pos
_rook_attacks_player(row, col)
  -> 只读取 self.state.board / self.state.player_pos
```

重构后（全部可传入 hypothetical 参数）：

```python
get_player_attackers(board=None, player_pos=None)
_pawn_attacks_player(row, col, player_pos=None)
_rook_attacks_player(row, col, board=None, player_pos=None)
```

- 默认参数为 `None` 时回退到 `self.state`，保持旧行为。
- Env 不再手写 Pawn / Rook / King 攻击规则，统一调用 `self.game.get_player_attackers(...)`。

---

## 4. relative_v4 的 55 维组成

```text
relative_v2 (47 dims)
├── White King relative geometry: 7 dims
└── 8 directions × ray (present, is_king, is_rook, is_pawn, norm_dist): 40 dims

move danger (8 dims)
└── 每个 MOVE_* 方向一个标量（Direction 顺序）
    0.0 = 该 MOVE 非法，或移动后不处于任何白棋攻击范围
    1.0 = 该 MOVE 合法，且移动后立即处于白棋攻击范围

合计 = 47 + 8 = 55 dims
```

关键实现要点（见 `env._get_move_danger_features`）：

- legality 来自 `self.game.legal_actions()`；非法 MOVE 记 `0.0`（由 action mask 表达 legality，v4 不重复编码）。
- 每个合法 MOVE 构造 **hypothetical board**：
  - 旧位置 `board[old] = EMPTY`
  - 新位置 `board[new] = BLACK_KING`
- 用 `get_player_attackers(board=hypothetical_board, player_pos=(nr, nc))` 判断。
- **同时更新 board 与 player_pos**。若只改 `player_pos` 而保留原 board，旧位置的 Black King 会成为 Rook LOS 的幽灵阻挡物，导致误判为安全（有专门的回归测试）。

---

## 5. 添加了哪些测试

### `tests/test_core.py`

- `test_get_player_attackers_default_matches_explicit_query`
- `test_get_player_attackers_hypothetical_rook`
- `test_get_player_attackers_hypothetical_pawn`
- `test_get_player_attackers_hypothetical_king`

### `tests/test_env.py`

- `test_relative_v4_geometry_shape`（shape 55 / finite / bounds / obs space contains）
- `test_relative_v4_v2_prefix_is_unchanged`（前 47 维 == v2）
- `test_relative_v4_move_danger_shape_and_range`（shape 8，取值只允许 0/1）
- `test_relative_v4_safe_move_has_zero_danger`
- `test_relative_v4_rook_move_is_marked_dangerous`（Rook 立即致死 = 1.0）
- `test_relative_v4_pawn_move_is_marked_dangerous`（Pawn 立即致死 = 1.0）
- `test_relative_v4_illegal_move_has_zero_danger`
- `test_relative_v4_hypothetical_removes_old_position_blocker`（old-position blocker 回归）
- `test_relative_v4_survives_win_terminal_state`（White King 已击杀的 terminal obs 不崩）

---

## 6. pytest 最终结果

```text
pytest -q
67 passed
```

（基线 54 passed；新增 13 个测试，全部通过。未删除任何旧测试。）

---

## 7. smoke train 是否通过

通过。

```text
--reward shaped_v2
--action-mode pruned_shots
--layout-mode random_medium
--feature-mode flat
--geometry-mode relative_v4
--timesteps 5000
--seed 42
--run-tag relv4_smoke
```

- observation space：`geometry = Box(-1.0, 1.0, (55,), float32)`
- 训练完整跑完，无 exception。

---

## 8. smoke load/evaluate 是否通过

通过。

```text
python scripts/evaluate_ppo.py \
    --model models/shaped_v2_pruned_shots_random_medium_flat_relative_v4_seed42_relv4_smoke/best_win_model.zip \
    --reward shaped_v2 \
    --action-mode pruned_shots \
    --layout-mode random_medium \
    --geometry-mode relative_v4 \
    --episodes 20 \
    --seed 10000
```

- 模型正常 load，observation space 无 mismatch。
- 20 episodes 完整运行。

（smoke 仅 5k timesteps，胜率无意义。）

---

## 9. 是否发现并修复额外 bug

- 复查时发现 `core.py` 中 `legal_actions()` 存在一份**逐字重复定义**（旧文件遗留，第二个定义覆盖第一个，行为一致）。本次**未改动**（属于与 v4 无关的历史冗余），仅在此记录，避免后续误以为遗漏。
- `train_ppo.py::make_env` 在 `monitor_path is not None` 时会把 `Monitor` 包装两次（现存行为）。本次未改动，smoke/正式训练均正常。

---

## 10. 尚未解决的问题 / 待办

- 本任务范围内没有阻塞问题。
- 正式 `relative_v4` 300k 训练、1000 episodes independent evaluation、behavior analysis 已完成，结果见第 11 节与 `docs/实验总结.md`。
- 若后续仍要提升，才考虑 `relative_v5 = enemy-response risk`；本任务不实现 v5。

---

## 11. 正式 300k 实验结果（seed 42）

命令：

```text
python scripts/train_ppo.py \
    --reward shaped_v2 \
    --action-mode pruned_shots \
    --layout-mode random_medium \
    --feature-mode flat \
    --geometry-mode relative_v4 \
    --timesteps 300000 \
    --seed 42 \
    --run-tag relv4_300k
```

Callback win rate（`WinRateEvalCallback`，100 eval episodes，eval_seed=20000）：

```text
25k   36.0%   (Death 55.0%, Timeout 9.0%)
50k   49.0%   (Death 47.0%, Timeout 4.0%)
75k   52.0%   (Death 44.0%, Timeout 4.0%)
100k  64.0%   (Death 36.0%, Timeout 0.0%)
150k  73.0%   (Death 27.0%, Timeout 0.0%)
200k  72.0%   (Death 27.0%, Timeout 1.0%)
250k  76.0%   (Death 24.0%, Timeout 0.0%)
300k  76.0%   (Death 24.0%, Timeout 0.0%)
```

Independent evaluation（1000 episodes，eval seed 10000）：

```text
Win:         74.50%
Death:       25.30%
Timeout:      0.20%
Mean return:  12.266
Mean length:  6.60
```

Behavior analysis（1000 episodes，deterministic，seed 10000）：

```text
Outcomes: WIN 745 (74.50%) / DEATH 253 (25.30%) / TIMEOUT 2 (0.20%)

Action distribution:
  MOVE    5002 (75.80%)
  SHOOT   1573 (23.84%)
  RELOAD    24 ( 0.36%)

Shooting:
  hit rate 100.00%  (pruned_shots)
  pawn kills 458 / rook kills 370 / king kills 745

Behavior by outcome:
  [WIN]   n=745  mean_length=7.07  mean_return= 20.489  move=5.14 shoot=1.91
  [DEATH] n=253  mean_length=4.64  mean_return=-11.790  move=4.03 shoot=0.59
  [TIMEOUT] n=2  mean_length=80.00 mean_return= -7.550 move=78.00 shoot=2.00
```

对比 relative_v2（多 seed Win ≈ 58% / Death ≈ 41%）：

```text
relative_v4 seed42：Win 74.5% / Death 25.3%
```

结果落在任务第 20 节给出的「Win ≥ 63~65% / Death ≤ 35~37%」提升区间内，说明
**Immediate movement safety 是当前主要瓶颈之一**，action-conditioned move danger 有效。
下一步若要继续，应研究 enemy reply risk（v5，本任务未实现）。

