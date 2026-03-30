# REVERIE 上的 decay + anti-loop 迁移说明

## 这次做了什么

这次没有重写 `dynamic memory`，因为 `REVERIE` 里原本就已经支持：

- `dynamic_memory_mode = update_only / decay_only / full`

所以这次真正补的是：

- 把 `R2R` 上已经验证过的 `anti-loop` 动作层规则迁移到 `REVERIE`

迁移后的整体思路是：

- `memory` 侧仍然沿用已有的 `dynamic memory`
- `action` 侧新增一个 test-time 的 `anti-loop` 约束

## 改动文件

- [agent_obj.py](/home/japluto/VLN/GridMM_ff/map_nav_src/reverie/agent_obj.py)
- [parser.py](/home/japluto/VLN/GridMM_ff/map_nav_src/reverie/parser.py)
- [run_reverie.sh](/home/japluto/VLN/GridMM_ff/map_nav_src/scripts/run_reverie.sh)

## 具体改了什么

### 1. `parser.py`

给 `REVERIE` 加入了和 `R2R` 对齐的 anti-loop 参数：

- `--anti_loop_enabled`
- `--anti_loop_backtrack_penalty`
- `--anti_loop_revisit_penalty`
- `--anti_loop_revisit_thresh`
- `--anti_loop_min_step`

默认值保持和 `R2R` 一致：

- `anti_loop_enabled = False`
- `anti_loop_backtrack_penalty = 0.08`
- `anti_loop_revisit_penalty = 0.03`
- `anti_loop_revisit_thresh = 2`
- `anti_loop_min_step = 2`

### 2. `agent_obj.py`

在 `REVERIE` 的 `rollout()` 里，增加了与 `R2R` 同风格的 `anti-loop` 逻辑。

插入点仍然是：

- `nav_logits / nav_probs` 已经算出来
- 最终 `argmax` 选动作之前

实现方式：

- 不改原始 `nav_logits`
- 只改一个用于最终动作选择的 `nav_probs` 副本
- 只对 non-stop action 生效
- STOP 不动

### 3. `run_reverie.sh`

补了脚本层开关，风格与 `run_r2r.sh` 保持一致：

- `ANTI_LOOP_MODE=off|on`
- `ANTI_LOOP_EXTRA_ARGS="..."`

## anti-loop 的具体规则

当前迁移的是 `R2R` 上更有效的那条版本：

### 核心不是广义 revisit，而是 immediate backtrack suppression

对每个 non-stop candidate：

1. 先根据当前 graph path，求出从当前 viewpoint 去目标 viewpoint 的执行路径
2. 取执行路径的第一跳 `next_hop`
3. 如果：

- `next_hop == prev_vpid`

则减去：

- `anti_loop_backtrack_penalty`

如果：

- `visit_count[next_hop] >= anti_loop_revisit_thresh`

则再减去：

- `anti_loop_revisit_penalty`

不过根据 `R2R` 的经验，这条线当前真正起作用的主要还是：

- `immediate backtrack penalty`

## 轻量统计

为了后续诊断，这次在 `REVERIE` 里也加了统计输出：

- `anti_loop_trigger_count`
- `backtrack_penalty_count`
- `revisit_penalty_count`
- `anti_loop_action_changed_count`

只要你在 test/eval 时打开 `anti_loop`，默认 GPU 上就会打印这组统计。

## 脚本使用方式

### 默认关闭

```bash
cd /home/japluto/VLN/GridMM_ff/map_nav_src
bash scripts/run_reverie.sh test
```

### 开启 anti-loop

```bash
cd /home/japluto/VLN/GridMM_ff/map_nav_src

DYNAMIC_MEMORY_MODE=decay_only \
ANTI_LOOP_MODE=on \
bash scripts/run_reverie.sh test
```

### 自定义 anti-loop 参数

```bash
cd /home/japluto/VLN/GridMM_ff/map_nav_src

DYNAMIC_MEMORY_MODE=decay_only \
ANTI_LOOP_MODE=on \
ANTI_LOOP_EXTRA_ARGS="--anti_loop_backtrack_penalty 0.22 --anti_loop_revisit_penalty 0.0 --anti_loop_revisit_thresh 2 --anti_loop_min_step 1" \
bash scripts/run_reverie.sh test
```

## 轻量验证结果

我做了一个只跑 `val_train_seen` 的 smoke test，配置是：

- `dynamic_memory_mode = decay_only`
- `anti_loop_enabled = True`
- `anti_loop_backtrack_penalty = 0.22`
- `anti_loop_revisit_penalty = 0.0`
- `anti_loop_revisit_thresh = 2`
- `anti_loop_min_step = 1`

结果：

- `anti_loop_trigger_count = 568`
- `backtrack_penalty_count = 2651`
- `revisit_penalty_count = 0`
- `anti_loop_action_changed_count = 1`

指标：

- `sr = 95.12`
- `oracle_sr = 97.56`
- `spl = 93.58`
- `rgs = 93.50`
- `rgspl = 91.96`

这说明：

- 代码链路是通的
- `anti_loop_stats` 已经能正常输出
- `REVERIE` 的 object grounding 流程没有被这次改动打断

## 当前判断

这次迁移的目标是：

- 先把 `R2R` 上验证过的动作层约束平移到 `REVERIE`
- 保持代码小、逻辑清晰、默认关闭

当前已经达到这个目标。

下一步如果要继续做 `REVERIE` 调参，最自然的顺序是：

1. 先固定 `dynamic_memory_mode = decay_only`
2. 再小范围看 `anti_loop_backtrack_penalty`
3. 主要观察：
   - `lengths`
   - `steps`
   - `spl`
   - `sr`
   - `rgs / rgspl`

一句话总结：

> `REVERIE` 现在已经具备了和 `R2R` 一样的 `decay + anti-loop` 推理改动能力。  
> 当前迁移版保持了“默认关闭、最小插入、只动 test-time 动作选择”的原则。
