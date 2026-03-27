## GridMM_ff

这个目录是从原始 [`GridMM`](/home/japluto/VLN/GridMM) 中拆出来的一个**离散环境工作副本**，用途是：

- 后续只在这里改代码
- 只关注离散数据集的训练入口和 eval 入口
- 方便初始化新 Git 仓库并上传 GitHub

当前这份副本**不再以连续环境 `VLN_CE` 为主线**，而是优先服务：

- `R2R`
- `REVERIE`
- `RxR`

## 当前目录结构

保留的主要代码目录：

- [`map_nav_src`](/home/japluto/VLN/GridMM_ff/map_nav_src)
- [`pretrain_src`](/home/japluto/VLN/GridMM_ff/pretrain_src)
- [`preprocess`](/home/japluto/VLN/GridMM_ff/preprocess)

已经迁移到本目录、但被 Git 忽略的大资源：

- [`datasets`](/home/japluto/VLN/GridMM_ff/datasets)
- [`data/pretrained_models`](/home/japluto/VLN/GridMM_ff/data/pretrained_models)

Git 忽略规则见：

- [`.gitignore`](/home/japluto/VLN/GridMM_ff/.gitignore)

## 已做的离散环境改动

### 1. 新建 `GridMM_ff` 工作副本

已经把离散相关代码框架从原仓库复制到了这里，并将离散数据和权重**迁移**到了新目录：

- 原目录中的 `datasets/` 已迁移到 [`GridMM_ff/datasets`](/home/japluto/VLN/GridMM_ff/datasets)
- 原目录中的 `data/pretrained_models/` 已迁移到 [`GridMM_ff/data/pretrained_models`](/home/japluto/VLN/GridMM_ff/data/pretrained_models)

这样后续可以直接在 `GridMM_ff` 下工作，不需要依赖旧目录。

### 2. GitHub 友好的忽略策略

当前 `.gitignore` 采用的是“**保留代码目录，忽略根目录大资源**”的方式：

- 忽略根目录下的 `/datasets/`
- 忽略根目录下的 `/data/`
- 不会误伤 `pretrain_src/data` 这类代码目录

### 3. RxR 单卡脚本适配

已经修改：

- [`map_nav_src/rxr/parser.py`](/home/japluto/VLN/GridMM_ff/map_nav_src/rxr/parser.py)
- [`map_nav_src/scripts/run_rxr.sh`](/home/japluto/VLN/GridMM_ff/map_nav_src/scripts/run_rxr.sh)

主要改动：

- 支持 `--dataset rxr`
- 默认单卡 `NGPUS=1`
- 默认 `BATCH_SIZE=1`
- 默认 `--tokenizer xlm`
- 单卡时直接走 `python3 main_rxr.py`
- 增加路径检查和缺失提示

## 已做的离散环境尝试

### 1. 权重类型检查

新增脚本：

- [`map_nav_src/scripts/check_nav_ckpt.py`](/home/japluto/VLN/GridMM_ff/map_nav_src/scripts/check_nav_ckpt.py)

已经验证：

- [`datasets/trained_models/r2r_best`](/home/japluto/VLN/GridMM_ff/datasets/trained_models/r2r_best)
  - 是离散导航完整 checkpoint
  - 可作为 `resume_file`
- [`datasets/trained_models/reverie_best`](/home/japluto/VLN/GridMM_ff/datasets/trained_models/reverie_best)
  - 可用于 `REVERIE` eval
- [`data/pretrained_models/grid_map.pt`](/home/japluto/VLN/GridMM_ff/data/pretrained_models/grid_map.pt)
  - 不是离散 `resume_file`
  - 更像 backbone 初始化权重
  - 不适合直接拿来跑离散 `RxR eval`

### 2. R2R eval smoke

已经在新目录里成功拉起 `R2R` 的离散 eval 链，使用：

- [`map_nav_src/scripts/run_r2r.sh`](/home/japluto/VLN/GridMM_ff/map_nav_src/scripts/run_r2r.sh)

关键信号：

- 成功加载 checkpoint
- 成功加载 train / val / test split
- 成功生成预测文件
- 成功写出评测日志

输出目录：

- [`datasets/R2R/exprs_map/eval/Grid_Map-dagger-vitbase-single-gpu-seed.0`](/home/japluto/VLN/GridMM_ff/datasets/R2R/exprs_map/eval/Grid_Map-dagger-vitbase-single-gpu-seed.0)

当前 smoke 里已经看到的结果：

- `val_seen: sr 79.92, spl 73.62`
- `val_unseen: sr 75.44, spl 65.06`

对应日志：

- [`valid.txt`](/home/japluto/VLN/GridMM_ff/datasets/R2R/exprs_map/eval/Grid_Map-dagger-vitbase-single-gpu-seed.0/logs/valid.txt)

### 3. REVERIE eval smoke

已经在新目录里成功拉起 `REVERIE` 的离散 eval 链，使用的是：

- [`main_nav_obj.py`](/home/japluto/VLN/GridMM_ff/map_nav_src/main_nav_obj.py)

关键信号：

- 成功加载 train / val split
- 成功加载 `reverie_best`
- 成功写出日志
- 已看到至少一条评测结果输出

当前 smoke 已看到：

- `val_train_seen: sr 95.12, spl 92.93`

对应日志：

- [`valid.txt`](/home/japluto/VLN/GridMM_ff/default/reverie_smoke_eval/logs/valid.txt)

### 4. RxR 当前状态

`RxR` 注释数据和脚本已经对齐，但**目前本地没有可直接用于离散 `RxR eval` 的 finetuned checkpoint**。

当前结论：

- 数据注释在 [`datasets/RXR/annotations`](/home/japluto/VLN/GridMM_ff/datasets/RXR/annotations)
- 代码入口和单卡脚本已经整理好
- 但没有找到合适的 `resume_file`
- 因此还不能直接做 `RxR eval`

## 已知注意事项

### 1. R2R 不建议裸跑 `python main_nav.py`

`R2R` 这套代码默认会初始化分布式环境，直接裸跑容易报：

- `Can't find any rank or local rank`

所以更稳妥的方式是直接走：

- [`run_r2r.sh`](/home/japluto/VLN/GridMM_ff/map_nav_src/scripts/run_r2r.sh)

### 2. Matterport3D 扫描目录

代码里默认会把扫描目录解析成：

- `datasets/Matterport3D/v1_unzip_scans`

当前本目录下没有明确补齐这个目录，但 `R2R` / `REVERIE` 的 smoke 已经能起 eval 链，说明至少在现阶段并没有阻塞基本离散评测。

如果后续某条任务线在 MatterSim 上报 scan data 缺失，再单独补这个目录。

### 3. 后续工作方式

接下来默认在 [`GridMM_ff`](/home/japluto/VLN/GridMM_ff) 内继续开发，优先模式是：

- 修改代码
- 跑离散 eval
- 不再以训练为主要目标

## 常用 eval 命令

已经单独整理在：

- [`map_nav_src/scripts/EVAL_COMMANDS.md`](/home/japluto/VLN/GridMM_ff/map_nav_src/scripts/EVAL_COMMANDS.md)

后续直接按这个文件里的命令跑即可。
