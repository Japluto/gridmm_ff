- [run_r2r.sh](/home/japluto/VLN/GridMM_ff/map_nav_src/scripts/run_r2r.sh)
- [run_reverie.sh](/home/japluto/VLN/GridMM_ff/map_nav_src/scripts/run_reverie.sh)
- [run_rxr.sh](/home/japluto/VLN/GridMM_ff/map_nav_src/scripts/run_rxr.sh)

我加进去的用法是统一的环境变量开关：

- `DYNAMIC_MEMORY_MODE=off`
- `DYNAMIC_MEMORY_MODE=update_only`
- `DYNAMIC_MEMORY_MODE=decay_only`
- `DYNAMIC_MEMORY_MODE=full`

默认不设时就是 `off`。另外我还顺手加了一个可选扩展入口：

- `DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_decay_lambda 0.08 --dynamic_memory_match_radius 0.9"`

**怎么用**

R2R:
```bash
cd /home/japluto/VLN/GridMM_ff/map_nav_src

bash scripts/run_r2r.sh test
DYNAMIC_MEMORY_MODE=update_only bash scripts/run_r2r.sh test
DYNAMIC_MEMORY_MODE=decay_only bash scripts/run_r2r.sh test
DYNAMIC_MEMORY_MODE=full bash scripts/run_r2r.sh test
```

REVERIE:
```bash
cd /home/japluto/VLN/GridMM_ff/map_nav_src

bash scripts/run_reverie.sh test
DYNAMIC_MEMORY_MODE=update_only bash scripts/run_reverie.sh test
DYNAMIC_MEMORY_MODE=decay_only bash scripts/run_reverie.sh test
DYNAMIC_MEMORY_MODE=full bash scripts/run_reverie.sh test
```

RxR:
```bash
cd /home/japluto/VLN/GridMM_ff/map_nav_src

RESUME_FILE=YOUR_RXR_CHECKPOINT bash scripts/run_rxr.sh test
DYNAMIC_MEMORY_MODE=update_only RESUME_FILE=YOUR_RXR_CHECKPOINT bash scripts/run_rxr.sh test
DYNAMIC_MEMORY_MODE=decay_only RESUME_FILE=YOUR_RXR_CHECKPOINT bash scripts/run_rxr.sh test
DYNAMIC_MEMORY_MODE=full RESUME_FILE=YOUR_RXR_CHECKPOINT bash scripts/run_rxr.sh test
```

**如果你要调超参**

例如：
```bash
DYNAMIC_MEMORY_MODE=full \
DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_decay_lambda 0.08 --dynamic_memory_match_radius 0.9" \
bash scripts/run_r2r.sh test
```

我已经做过 `bash -n` 和 `git diff --check`，这三个脚本的语法和格式都通过了。