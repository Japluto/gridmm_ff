# decay_only 与 anti-loop 阶段性总结

## 当前结论

现阶段在 `R2R` 上，主线配置仍然应该是：

- `dynamic_memory_mode = decay_only`

`anti-loop` 这条线是有价值的，但它目前更像是一个“提升执行效率、缩短路径”的辅助规则，而不是一个能稳定提升最终分数的主方法。

## anti-loop 现在到底在做什么

从当前实验结果看，真正起作用的并不是广义的 `revisit penalty`，而是：

- `immediate backtrack suppression`

也就是：

- 当某个候选动作的执行第一跳 `next_hop` 会立刻回到上一步的 viewpoint 时，对它做惩罚

证据很直接：

- 多轮实验里 `revisit_penalty_count = 0`
- 所以当前有效部分几乎全部来自 `backtrack_penalty`

## 当前找到的较优 anti-loop 配置

在以下 decay 参数固定时：

- `dynamic_memory_decay_lambda = 0.12`
- `dynamic_memory_min_mem_weight = 0.35`

小范围 `val_seen` sweep 后，当前最平衡的 anti-loop 参数是：

- `anti_loop_backtrack_penalty = 0.22`
- `anti_loop_revisit_penalty = 0.0`
- `anti_loop_revisit_thresh = 2`
- `anti_loop_min_step = 1`

选择这组的原因：

- 比 `0.28` 和 `0.34` 更稳
- `SR` 和 `oracle_sr` 更健康
- 同时还能缩短路径、改善推理效率

## 对比：单独 decay_only vs decay_only + anti-loop

### 1. 单独 `decay_only`

当前观察到的结果：

- `val_train_seen`
  - `sr = 84.00`
  - `spl = 79.27`
  - `nDTW = 83.76`
  - `lengths = 12.42`
  - `steps = 6.13`
  - `nav_error = 1.42`
  - `oracle_sr = 90.00`
- `val_seen`
  - `sr = 80.02`
  - `spl = 73.60`
  - `nDTW = 78.97`
  - `lengths = 12.71`
  - `steps = 6.38`
  - `nav_error = 2.30`
  - `oracle_sr = 87.56`
- `val_unseen`
  - `sr = 75.52`
  - `spl = 65.16`
  - `nDTW = 71.01`
  - `lengths = 13.29`
  - `steps = 6.81`
  - `nav_error = 2.73`
  - `oracle_sr = 83.10`

### 2. `decay_only + anti-loop`

代表性的效率型结果：

- `val_train_seen`
  - `sr = 84.67`
  - `spl = 80.22`
  - `nDTW = 84.36`
  - `lengths = 11.71`
  - `steps = 5.78`
  - `nav_error = 1.41`
  - `oracle_sr = 90.00`
- `val_seen`
  - `sr = 79.33`
  - `spl = 73.75`
  - `nDTW = 79.13`
  - `lengths = 11.91`
  - `steps = 5.94`
  - `nav_error = 2.41`
  - `oracle_sr = 86.88`
- `val_unseen`
  - `sr = 74.67`
  - `spl = 65.28`
  - `nDTW = 71.26`
  - `lengths = 12.41`
  - `steps = 6.35`
  - `nav_error = 2.83`
  - `oracle_sr = 82.03`

## 如何理解这组结果

### 为什么 `decay_only` 目前更适合作为主线

如果你的目标是 benchmark 分数，那么当前 `decay_only` 更优：

- `SR` 更高
- `oracle_sr` 更高
- `nav_error` 更低

这说明单纯的 soft forgetting 已经能够有效减弱陈旧 memory 的干扰，而且没有过度约束动作选择。

### 为什么 anti-loop 仍然值得保留

`anti-loop` 并不是没用，它的价值主要体现在：

- 明显缩短轨迹
- 减少步数
- 提升推理速度 / 降低评测时间
- 在 test-time 确实能改动动作选择

但目前它更像是在：

- 用一小部分分数，去换更短的路径和更高的执行效率

所以当前阶段更合理的定位是：

- `decay_only`：主分数配置
- `decay_only + immediate backtrack suppression`：效率优先的备选配置

## 推荐的实际使用方式

### 主配置

如果你最在意最终分数，建议使用：

```bash
cd /home/japluto/VLN/GridMM_ff/map_nav_src

DYNAMIC_MEMORY_MODE=decay_only \
bash scripts/run_r2r.sh test
```

### 备选效率配置

如果你更在意：

- 路径更短
- 评测更快
- 同时分数不要掉太多

可以使用：

```bash
cd /home/japluto/VLN/GridMM_ff/map_nav_src

DYNAMIC_MEMORY_MODE=decay_only \
DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_decay_lambda 0.12 --dynamic_memory_min_mem_weight 0.35" \
ANTI_LOOP_MODE=on \
ANTI_LOOP_EXTRA_ARGS="--anti_loop_backtrack_penalty 0.22 --anti_loop_revisit_penalty 0.0 --anti_loop_revisit_thresh 2 --anti_loop_min_step 1" \
bash scripts/run_r2r.sh test
```

## 最终结论

当前最稳妥的判断是：

- 不要让 `anti-loop` 替代 `decay_only`
- `decay_only` 仍然是当前主结果线
- `anti-loop` 适合保留为一个“缩路径 / 提效率”的辅助配置

一句话总结：

> `decay_only` 更适合追求分数。  
> `decay_only + immediate backtrack suppression` 更适合追求更短路径和更低评测耗时。
