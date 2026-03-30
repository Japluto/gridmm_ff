# 反思报告：dynamic update 与 instruction 优化尝试

这份文档专门复盘两条后来没有成为主线的方向：

1. `dynamic memory` 里的 **dynamic update / full**
2. `instruction` 相关优化尝试

重点不是重复记实验结果，而是回答下面几个问题：

- 当时为什么会想到这条方向
- 具体是怎么实现和尝试的
- 为什么从直觉上看合理，但最后效果不好
- 这些尝试给后续研究留下了什么启发

---

## 1. dynamic update 方向的复盘

### 1.1 最初的出发点

最开始引入 dynamic memory 机制时，核心直觉是：

> Grid Memory 不应该无条件累积历史信息。  
> 旧的、重复的、陈旧的、低价值的信息应该被削弱；  
> 新的、有变化的、有信息量的观测应该更积极地写入 memory。

这个直觉本身并不奇怪，因为离散 VLN 里确实容易遇到两类问题：

- 地图里越来越多的历史 patch 堆积起来，读出时噪声变大
- 某些位置反复被看到，但并没有带来新的有用信息

因此当时设计了两个子方向：

1. `dynamic update`
   - 重点解决“写入”的问题
   - 新观测来了之后，不是固定规则合并，而是按 heuristic gate 决定写多少

2. `soft decay`
   - 重点解决“读出”的问题
   - 旧 memory 不是硬删，而是在读出时降低权重

理论上看，这两者合在一起形成 `full` 是最完整的方案：

- 写入更聪明
- 读出更保守

所以一开始很自然会觉得：

> `full` 应该比单独 `decay_only` 更强。

---

### 1.2 实际是怎么做的

#### 1.2.1 dynamic update 的做法

写入侧使用了一个 heuristic gate：

- `novelty`
- `age`
- `repeat_factor`

组合成一个 update gate，大致思路是：

- 新旧特征差异大，更新更多
- 很久没更新的 memory，更容易被刷新
- 重复观察很多次的 memory，更新应该更谨慎

然后用这个 gate 去做：

- 新旧特征的加权融合
- 同步维护 metadata，如：
  - `last_update_step`
  - `visit_count`

#### 1.2.2 full 的做法

在 `full` 模式下，同时启用：

- `dynamic update`
- `soft decay`

也就是说：

- memory 写入已经不是原始行为
- memory 读出也进一步加入了衰减权重

从系统角度看，`full` 已经不是“一个局部小修”，而是：

> 同时对 memory 的写入分布和读出分布做了双重干预。

---

### 1.3 为什么这条线一开始看起来很合理

这条线的吸引力主要来自三个方面：

#### 1.3.1 很符合直觉

“旧信息别一直堆着，新信息该更快写进去”  
这个判断非常自然，也很容易讲故事。

#### 1.3.2 不需要训练

因为用的是 heuristic gate 和 metadata：

- 不加新模块
- 不需要新 loss
- 不需要 finetune

这很符合当前实验约束。

#### 1.3.3 比 instruction 线更贴近 memory 本体

instruction 线是在动作层加浅层 prior，信息间接；  
dynamic update 线则是直接改 memory 的维护规则，看起来更“对症”。

---

### 1.4 为什么最后结果不如 `decay_only`

这部分是最关键的反思。

#### 1.4.1 `decay_only` 本质上是在“少做错事”

`decay_only` 的本质不是主动教模型做什么，而是：

> 在读出时减少旧 memory、重复 memory、低价值 memory 的影响。

它更像一个“去噪器”或“保守正则化”：

- 不改 memory 内容本身
- 不直接改动作决策
- 只是在原模型已经学会的表示上，稍微压一压坏信息

因此它很容易得到这种结果：

- `lengths / steps` 变好
- `SPL` 变好
- `SR` 不一定大涨，但通常比较稳

#### 1.4.2 dynamic update / full 本质上是在“主动改写 memory”

这一点和 `decay_only` 非常不一样。

`dynamic update` 干的是：

- 改已有 memory slot 的内容
- 改它被后续所有 step 读到的方式

这意味着一旦某一步写错了，错误不是一次性的，而是会累积：

- 当前 step 写偏一点
- 后面每一步都在读这个偏掉的 memory

这种误差会沿着 rollout 一直传下去。

#### 1.4.3 heuristic gate 虽然合理，但不一定和模型训练分布一致

从公式上看：

- `novelty`
- `age`
- `repeat`

这些信号都很合理。

但问题是：

> “从人类直觉上合理”  
> 不等于  
> “和原始训练好的 VLN-BERT / GridMM 表示空间完全兼容”。

模型在训练时见到的 memory 统计分布是原始机制下形成的。  
一旦 test-time 改成新的 update 规则，可能会出现：

- 特征幅值分布变化
- 某些 slot 被更新得太快
- 某些 slot 被合并得太积极
- 原本依赖的历史信息被过早刷新

这些都可能让模型“读到一种它没在训练中学过的地图”。

#### 1.4.4 `full` 把风险叠加了

`full` 的问题比单独 `update_only` 还更明显，因为它同时做了两件事：

1. 写入变了
2. 读出也变了

也就是说：

- 一边在改 memory 内容
- 一边在改 memory 权重

从研究上看它很完整，但从风险上看也更大。

所以最后出现的现象很容易理解：

- `full` 不一定完全崩
- 但经常不如更单纯、更保守的 `decay_only`

---

### 1.5 对这条线的阶段性结论

当前最重要的认识是：

> 在无训练适配条件下，  
> **被动去噪** 往往比 **主动改写 memory** 更稳。

所以这条线留下的最有价值结论不是：

- “dynamic update 完全没价值”

而是：

- 在当前约束下，`decay_only` 更适合作为主线
- `dynamic update / full` 更像研究性探索，不适合作为当前主配置

---

## 2. instruction 优化方向的复盘

### 2.1 最初的出发点

instruction 线最开始的核心想法是：

> 既然 agent 的决策本来就依赖 instruction，  
> 那有没有可能在不训练的前提下，  
> 用 very light-weight 的方式，把 instruction 里最关键的信息再强调一下？

这条线之所以吸引人，是因为它看起来几乎不需要改模型：

- 不需要改 visual branch
- 不需要改 map structure
- 不需要重新训练

只要在 test-time 给 instruction 一点额外的 bias，就可能让模型在模糊决策时更偏向“正确的方向”。

---

### 2.2 实际上做过哪些尝试

#### 2.2.1 instruction augmentation

做法：

- 保留原 instruction
- 追加一个基于关键词抽取的 summary
- 希望起到“强调方向词 / 地标词”的作用

例如：

- 原句不改
- 后面追加 `[SEP] key cues: ...`

#### 2.2.2 global direction rerank

做法：

- 不改 instruction 文本
- 不改 encoder
- 只在最终 action argmax 前，对 non-stop candidate 做一个 very weak rerank
- cue 来自整条 instruction 里的 `left / right / straight`

#### 2.2.3 local first-clause rerank

做法：

- 认为“整句方向词太粗糙，后面的 turn cue 可能不属于当前步”
- 只用 instruction 的第一子句
- 只认显式动作短语，如：
  - `turn left`
  - `turn right`
  - `go straight`

这比 global cue 更保守，希望减少误导。

---

### 2.3 为什么这条线一开始看起来合理

#### 2.3.1 因为指令本来就是任务核心

R2R 本质上就是语言导航。

所以很自然会觉得：

- 方向词
- 局部短语
- 关键词

这些语言线索应该能帮助当前动作选择。

#### 2.3.2 因为它实现成本低

不需要动模型主干，只要：

- 改文本
或者
- 改动作层 rerank

看上去就是一个典型的 test-time baseline。

#### 2.3.3 因为它容易解释

“instruction 说 left，所以候选里更偏 left 一点”  
这种故事非常容易讲，也很容易做 ablation。

---

### 2.4 为什么最后效果不好

#### 2.4.1 instruction augmentation 破坏了输入分布

这是最直接的一条。

模型原本训练时看到的是原始 instruction 分布。  
当 test-time 突然变成：

- 原句
- 再加一个额外 summary

即使人类看起来觉得“只是多强调一点关键词”，  
对于 encoder 来说，这已经是：

> 一个新的输入分布。

它可能导致：

- token pattern 变化
- 句子长度变化
- 注意力分布变化
- 原本对整句结构的理解被打乱

所以这条线出现明显负收益并不意外。

#### 2.4.2 global direction rerank 的信息太粗

整条 instruction 里常常不只一个方向词。

例如：

- 先左转
- 然后右转
- 再直走

如果直接从整条 instruction 里抽 cue，那么当前 step 用到的 prior 很可能不是“当前该做的事”，而是“未来某一步该做的事”。

于是就会出现：

- cue 不是完全没信息
- 但也不够局部
- 最终会改动作，但不一定改对

这就是为什么激进 sanity check 时能看到：

- `rerank_action_changed_count` 上升

但指标并没有跟着变好。

#### 2.4.3 local first-clause rerank 又太保守

后来为了提高 precision，做了两层限制：

1. 只看第一子句
2. 只认显式动作短语

这确实更干净，但副作用是：

- coverage 大幅下降
- 真正能触发 rerank 的样本很少
- 最后几乎退化成 baseline

也就是说，这条线碰到了一个很典型的两难：

- 规则宽一点：能改动作，但改得不准
- 规则严一点：不怎么犯错，但也不怎么生效

#### 2.4.4 浅层语言 prior 的信息量不够

更深层的原因在于：

> 当前 agent 的动作选择并不是只靠一个浅层方向词就能决定的。

真正影响动作的还有：

- 当前局部视觉布局
- candidate 的相对角度
- graph context
- 已访问状态
- instruction 的整体语义

单独把 `left/right/straight` 抽出来，作为一个很浅的 prior，信息量还是太少。

所以最终就会出现：

- 它不是完全无效
- 但也不足以稳定提升

---

### 2.5 对 instruction 线的阶段性结论

当前 instruction 线给出的核心认识是：

> 在不训练、不改 encoder 的条件下，  
> 单靠浅层方向词或关键词，  
> 很难稳定改善当前 agent 的动作选择。

更具体地说：

- `instruction augmentation`
  - 风险太高
  - 会破坏输入分布

- `minimal direction rerank`
  - 能做成 test-time baseline
  - 但信息量不够，难以转成稳定收益

所以这条线目前更像一个：

- 已经验证过边界
- 不值得继续作为主线推进

---

## 3. 这两条线带来的共同启发

虽然 `dynamic update` 和 `instruction 优化` 最终都没有成为主线，但它们带来的启发其实很一致。

### 3.1 在无训练设定下，越“主动替模型做决定”，风险越大

比较下来很明显：

- `decay_only`
  - 属于保守型去噪
  - 更稳

- `dynamic update`
  - 主动改 memory 内容
  - 风险更高

- `dual stop`
  - 主动改 stop 决策
  - 风险更高

- `instruction rerank`
  - 主动改 action 排序
  - 风险更高

这说明在当前条件下，更适合做的是：

> 保守的 filtering / denoising  
> 而不是 aggressive 的 decision rewriting

### 3.2 “看起来合理的启发式”不一定和训练分布兼容

这是整个复盘里非常重要的一点。

无论是：

- 用 heuristic gate 更新 memory
- 还是用方向词给 action rerank

这些规则从人的直觉上都说得通。  
但模型真正学到的是训练时那套 joint distribution。

一旦 test-time 改动太主动，就容易出现：

- 人觉得更聪明
- 模型却觉得更陌生

### 3.3 真正值得保留的是“低侵入、低风险、结构上顺着模型来”的方向

目前看下来，最符合这个原则的是：

- `decay_only`
- 以及作为备选效率版的 `immediate backtrack suppression`

因为它们本质上都在做：

- 降低坏信息影响
- 降低冗余行为影响

而不是直接替模型重写内部决策。

---

## 4. 当前的最终看法

### 关于 dynamic update

结论不是“完全失败”，而是：

- 方向有研究意义
- 但在当前不训练约束下，不适合作为主配置
- 现阶段主线应该回到 `decay_only`

### 关于 instruction 优化

结论更直接：

- 这条最小 instruction 线已经比较清楚地摸到了边界
- 再继续深挖的性价比不高
- 可以阶段性止损

---

## 5. 一句话总结

这两条线留下的最有价值认识是：

> 在当前离散 VLN、无训练、只做 test-time 小改动的前提下，  
> **保守地去掉坏信息**，比 **主动地替模型重写 memory / action / instruction 解释** 更容易稳定带来收益。

这也是为什么到目前为止，真正留下来作为主线的，是：

- `decay_only`

而不是：

- `dynamic update / full`
- `dual stop`
- `instruction augmentation`
- `minimal instruction rerank`
