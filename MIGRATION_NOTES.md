# Migration Notes

## Old File To New Modules

- `train_v4p5_ddp.py`
  - 模型层与主模型拆到 `models/`
  - `LMDB_Dataset` 与 `DistributedBucketSampler` 拆到 `datasets/`
  - DDP、checkpoint、history csv、plot、trainer 拆到 `engine/`
  - loss 公式拆到 `losses/`
- `inverse_design_ga_broadband.py`
  - 模型定义删除，改为直接使用 `models.MetaField`
  - 材料库、结构编码、padding/grid utils、几何合法性拆到 `structure/`
  - GA config / operators / evaluator / runner 拆到 `search/ga/`
- `benchmark_throughput.py`
  - 吞吐 benchmark 主逻辑迁移到 `benchmarks/throughput.py`
  - 轻量入口迁移到 `scripts/bench/benchmark_throughput.py`
  - 不再 import GA 大脚本
- `DL_limited_lamdba.py`
  - 整理为 `material_parameters/drude_lorentz_fit.py`
  - 轻量入口迁移到 `scripts/data/fit_material_drude_lorentz.py`
  - 拟合参数、输入文件、输出路径改为 YAML 管理

## Benchmark Preservation

- 保留了当前 benchmark 中已经存在的关键逻辑:
  - batch doubling
  - free-memory safety guard
  - projected-memory early stop
  - throughput stall early stop
  - OOM / backend limit graceful stop
  - GA paper-number speedup summary
- 调整仅限依赖链清理:
  - 现在直接依赖 `models/MetaField`
  - `get_padded_size` 改为来自 `structure/utils.py`

## Behavior Kept Unchanged

- MetaField 主体网络结构、条件编码、Transformer 深度与原脚本保持一致
- 训练中的 weighted MSE、gradient loss、gradient accumulation 处理保持一致
- GA 的 genome 结构、repair、crossover、mutation、tournament selection、top-5% volumetric fitness 保持一致
- 结构编码到 surrogate 输入张量的核心流程保持一致

## Small Engineering Fixes

- 训练、GA、benchmark 三条主线都改为从 yaml 读取配置
- benchmark 与 GA 彻底解耦
- 材料拟合工具也纳入项目，并统一维护 `materials.json` 与 `material_ri_mapping.json`
- 入口脚本瘦身，只负责 parse config 和组装对象
- 数据预处理脚本移动到 `scripts/data/`
- 文件名中的版本号已去掉，模型名统一为 `MetaField`
