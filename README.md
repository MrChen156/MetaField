# MetaField

**Physics-aware neural surrogates for high-fidelity ultra-fast simulation of nanophotonic sensors**

**中文**
MetaField 是一个面向纳米光子传感器代理建模、逆向设计与计算基准测试的科研代码库。项目围绕统一的全场预测模型展开：训练阶段学习从 3D 几何与多层色散材料到电磁场响应的高保真映射；搜索阶段利用代理模型将遗传算法加速到高通量寻优场景；Benchmark 阶段评估模型在不同硬件下的吞吐量与延迟极限。

**English**
MetaField is a research-oriented codebase for surrogate modeling, inverse design, and benchmarking of nanophotonic sensors. The repository is centered on a shared full-field prediction model: training learns the mapping from 3D geometry and dispersive materials to electromagnetic responses; search uses the surrogate to accelerate high-throughput genetic optimization; benchmarking measures throughput and latency limits across hardware architectures.

---

## Project Scope | 项目范围

- `Training / 训练`: supervised surrogate-model training with LMDB datasets and DDP support.
- `Search / 搜索`: GA-based inverse design using the shared surrogate model and structure encoder.
- `Benchmark / 基准`: throughput-oriented evaluation for deployment analysis and paper-style speedup reporting.
- `Materials / 材料`: Drude-Lorentz fitting utilities for maintaining `materials.json` and `material_ri_mapping.json`.

---

## Installation | 安装

**中文**
推荐使用 conda / mamba 创建独立环境：

```bash
conda env create -f environment.yaml
conda activate metafield
```

`environment.yaml` 默认从 PyPI 安装 `torch`，适合 CPU 与 Apple Silicon/MPS。若使用 NVIDIA CUDA，请根据服务器 CUDA 版本安装对应的 PyTorch wheel 或 conda package，然后再运行训练与 benchmark。

**English**
Use conda / mamba to create an isolated environment:

```bash
conda env create -f environment.yaml
conda activate metafield
```

The default `environment.yaml` installs `torch` from PyPI, which is suitable for CPU and Apple Silicon/MPS setups. For NVIDIA CUDA machines, install the PyTorch wheel or conda package matching the local CUDA runtime before running training or benchmarks.

---

## Dataset Setup | 数据集准备

**中文**
本项目依赖外部仿真数据集。由于原始 H5 与转换后的 LMDB 文件体积较大，数据不随仓库分发，需要单独托管和下载。当前 `unified_v6p3_260209.h5` 版本包含 70,000 条 HDF5 样本，按 padding 后的空间尺寸组织为多个 `size_Z_X` group。

推荐使用方式：

1. 从外部数据发布页面下载数据包。
   链接占位：`[dataset link pending]`
2. 将数据解压到项目之外的独立存储目录，例如 `/path/to/data/unified_v6p3.h5`。
3. 使用仓库内脚本从 H5 生成 split JSON，并转换为 LMDB。
4. 在 `configs/train/metafield_ddp.yaml` 中修改 `data.lmdb_path` 和 `data.split_json`。

**English**
This project depends on an external simulation dataset. Because the original H5 files and the converted LMDB files are too large to ship with the repository, the dataset should be hosted and downloaded separately. The current `unified_v6p3_260209.h5` release contains 70,000 HDF5 samples organized into `size_Z_X` groups according to padded spatial resolution.

Suggested workflow:

1. Download the dataset archive from an external hosting page.
   Placeholder: `[dataset link pending]`
2. Extract it to a storage location outside the repository, for example `/path/to/data/unified_v6p3.h5`.
3. Use the repository scripts to generate the split JSON and convert H5 to LMDB.
4. Update `data.lmdb_path` and `data.split_json` in `configs/train/metafield_ddp.yaml`.

**H5 to training-ready files | H5 到训练文件**

```bash
python3 -m scripts.data.stratified_splitter \
  --h5 /path/to/data/unified_v6p3.h5 \
  --output /path/to/data/datasplit.json \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 2025

python3 -m scripts.data.convert_h5_to_lmdb \
  --h5 /path/to/data/unified_v6p3.h5 \
  --lmdb /path/to/data/unified_v6p3.lmdb \
  --map-size-gb 180 \
  --batch-size 500 \
  --commit-every 2000 \
  --overwrite
```

`stratified_splitter.py` 生成训练/验证/测试划分 JSON；`convert_h5_to_lmdb.py` 将 H5 中的 `x / y / cond / mask` 批量写入 LMDB。两者均支持外部数据路径，因此下载公开 H5 后无需修改源码。

`stratified_splitter.py` generates the train/val/test split JSON. `convert_h5_to_lmdb.py` converts `x / y / cond / mask` records from H5 into LMDB. Both scripts accept external paths, so users do not need to edit source files after downloading the public H5 dataset.

---

## Dataset Record Description | 数据记录说明

**中文**
MetaSPR-SimDB 是一个由 FDTD 全波电磁仿真生成的 HDF5 数据库，用于训练和评估纳米光子 MetaSPR 传感器的神经网络代理模型。每个样本包含：

- `x`: 物理输入张量 `[epsilon_r, epsilon_i, k0x, k0z, SDF]`，其中 `epsilon_r = n^2 - k^2`，`epsilon_i = 2nk`，`k0x/k0z` 由线性扩展后的空间坐标与波矢相乘得到，`SDF` 是基于几何边界的非负归一化距离场。
- `y`: 电磁场输出 `[Re(Ex), Im(Ex), Re(Ez), Im(Ez), Re(Hy) * Z0, Im(Hy) * Z0]`，其中磁场分量已乘以自由空间阻抗 `Z0 = 376.73`。
- `mask`: 原始 FDTD 计算区域的有效区域掩码；padding 区域为 0。
- `cond`: 三维全局条件 `[k0 * 1e-7 - 1, k0 * period / 2, k0 * z_source]`。
- `p`: 结构周期，单位为米。
- `id`: 从源 `.mat` 文件名解析得到的设计编号。

预处理流程包括中心对齐 padding、横向周期 circular padding、纵向 replicate padding、坐标线性外推、复折射率到复介电常数转换、几何边界距离场生成、磁场阻抗缩放、source phase 条件项构造，以及按设计 ID 进行的光谱插值式 train/validation/test 划分。

**English**
MetaSPR-SimDB is an HDF5 full-wave electromagnetic simulation database generated from FDTD simulations of MetaSPR sensor structures. It is intended for training and evaluating neural surrogate models for nanophotonic sensors. Each sample contains:

- `x`: physics-aware input tensor `[epsilon_r, epsilon_i, k0x, k0z, SDF]`, where `epsilon_r = n^2 - k^2`, `epsilon_i = 2nk`, `k0x/k0z` are built from linearly extended spatial coordinates multiplied by the free-space wavenumber, and `SDF` is a non-negative normalized distance-to-geometry-boundary field.
- `y`: electromagnetic field tensor `[Re(Ex), Im(Ex), Re(Ez), Im(Ez), Re(Hy) * Z0, Im(Hy) * Z0]`, with magnetic-field channels scaled by the free-space impedance `Z0 = 376.73`.
- `mask`: valid-region mask for the original FDTD computational window; padded regions are marked as 0.
- `cond`: three global physical condition variables `[k0 * 1e-7 - 1, k0 * period / 2, k0 * z_source]`.
- `p`: structural period in meters.
- `id`: design identifier parsed from the source `.mat` filename.

The preprocessing pipeline performs center-aligned padding, circular padding along the horizontal periodic direction, replicate padding along the vertical direction, coordinate linear extension, complex-index to complex-permittivity conversion, geometry-boundary distance-field generation, magnetic-field impedance scaling, source-phase condition construction, and design-wise spectral-interpolation train/validation/test splitting.

---

## Checkpoints | 权重文件

**中文**
仓库保留空的 `checkpoints/` 目录，但不提交大体积模型权重。训练会默认把权重写入该目录；GA 搜索配置默认读取 `checkpoints/best_model.pth`。运行搜索或需要预训练 surrogate 时，请先自行训练生成权重，或从外部发布页面下载 checkpoint 并放置为：

```text
checkpoints/best_model.pth
```

**English**
The repository keeps an empty `checkpoints/` directory, but large model weights are not tracked by git. Training writes checkpoints there by default, and GA configs expect `checkpoints/best_model.pth`. Before running search or any pretrained-surrogate workflow, either train the model yourself or download the checkpoint from an external release page and place it at:

```text
checkpoints/best_model.pth
```

---

## Repository Layout | 目录结构

```text
MetaField/
├── benchmarks/          # Benchmark implementations
├── checkpoints/         # Local model weights; .pth files are not tracked
├── configs/             # YAML configs for training, search, benchmark, materials
├── datasets/            # LMDB dataset, samplers, dataset utilities
├── engine/              # DDP, checkpoint, logging, trainer
├── losses/              # Field-domain losses and metrics
├── material_parameters/ # Material fitting utilities and fitted material records
├── models/              # Shared MetaField model and layers
├── scripts/             # Lightweight entrypoints and data-prep scripts
├── search/ga/           # Genetic algorithm backend
├── structure/           # Structure encoding, materials, constraints, geometry utils
└── tests/               # Minimal smoke tests
```

---

## Core Data Interface | 核心数据与物理语义

**中文**
训练和推理默认使用四类张量，直接对应底层物理信息：

- `x`: 代理模型输入特征张量，形状通常为 `[B, 5, Z, X]`。包含介电常数实部 `epsilon_r`、虚部 `epsilon_i`、空间波矢项 `k0x`、`k0z`，以及非负几何边界距离场 `SDF`。
- `y`: 目标全场分布张量，形状通常为 `[B, 6, Z, X]`，对应 FDTD 求解得到的复数场分量表示。
- `cond`: 全局物理条件向量，当前为 3 分量条件输入，通过 FiLM 注入网络，用于调制不同结构尺度、入射条件和频率相关信息。
- `mask`: 空间有效区域掩码，用于在 loss 计算与适应度计算时聚焦特定物理区域。

LMDB 中每个样本记录上述字段；split JSON 则定义 train / val / test 中各个 `group` 对应的样本 index。

**English**
Training and inference use four tensor groups that map directly to the underlying physics:

- `x`: surrogate input tensor, typically shaped as `[B, 5, Z, X]`, containing `epsilon_r`, `epsilon_i`, spatial wave-vector terms `k0x` and `k0z`, and the non-negative geometry-boundary distance field `SDF`.
- `y`: target full-field tensor, typically shaped as `[B, 6, Z, X]`, corresponding to the FDTD-computed complex field components.
- `cond`: global physical condition vector. The current implementation uses a 3-component conditioning signal injected through FiLM layers to modulate scale, excitation, and frequency-related effects.
- `mask`: spatial validity mask used to focus losses and fitness evaluation on physically relevant regions.

Each LMDB record stores these fields, while the split JSON defines sample indices for each `group` under train / val / test.

---

## Configuration Scientific Semantics | 配置项科学解释

**中文**
项目入口统一从 YAML 读取配置。不同工作流使用独立配置文件，从而避免训练、搜索、benchmark 与材料拟合参数混杂在单个脚本中。

**English**
All project entrypoints read from YAML configs. Each workflow keeps its own config so that training, search, benchmarking, and material fitting remain decoupled.

### 1. Training Config | 训练配置

文件: `configs/train/metafield_ddp.yaml`

**中文**

- `data.lmdb_path`: LMDB 数据集路径。训练代码从该目录读取已经转换好的 `x / y / cond / mask` 样本。
- `data.split_json`: train / val / test 划分文件路径。该 JSON 记录每个 `size_Z_X` group 中哪些 index 属于不同 split。
- `model.base_channels`: U-Net 主干的基础通道数，控制模型容量与显存消耗。
- `model.heads`: bottleneck Transformer 中的 attention head 数量。
- `model.max_dist`: 相对位置偏置的最大距离截断，用于限制全局 attention 中的相对空间距离表大小。
- `model.cond_embed_dim`: 全局条件向量经过 Fourier feature 与 MLP 后的嵌入维度，FiLM 层使用该嵌入调制卷积特征。
- `train.save_dir`: checkpoint、训练日志、history CSV 和训练曲线输出目录。
- `train.pretrain_path`: warm-start 权重路径；为空时从随机初始化开始训练。
- `train.epochs`, `train.batch_size`, `train.grad_accum_steps`, `train.lr`: 训练轮数、单卡 batch、梯度累积步数和初始学习率。
- `train.field_norm`: 电磁场归一化系数。H5 中的 `y` 没有预先除以该值；训练 loss 中会用它缩放目标场，缓解局域场增强导致的数值尺度问题。
- `train.grad_weight`: 空间梯度 loss 权重，用于鼓励预测场在边界和热点附近保持合理的空间变化。
- `train.cache_clear_interval`: 显存缓存清理周期，主要用于长时间 DDP 训练的工程稳定性。
- `train_loader.*`, `val_loader.*`: DataLoader 并行参数，例如 `num_workers`、`prefetch_factor` 和 `persistent_workers`。
- `optimizer.*`, `scheduler.*`: 优化器和学习率调度参数。

**English**

- `data.lmdb_path`: Path to the converted LMDB dataset. Training reads preprocessed `x / y / cond / mask` samples from this directory.
- `data.split_json`: Path to the train / validation / test split JSON. It records which sample indices in each `size_Z_X` group belong to each split.
- `model.base_channels`: Base channel width of the U-Net backbone; it controls model capacity and memory usage.
- `model.heads`: Number of attention heads in the bottleneck Transformer.
- `model.max_dist`: Maximum relative-distance cutoff for the relative-position bias table used by global attention.
- `model.cond_embed_dim`: Embedding dimension for the global condition vector after Fourier features and MLP projection; FiLM layers use this embedding to modulate convolutional features.
- `train.save_dir`: Directory for checkpoints, logs, history CSV files, and training curves.
- `train.pretrain_path`: Optional warm-start checkpoint path. If empty, training starts from random initialization.
- `train.epochs`, `train.batch_size`, `train.grad_accum_steps`, `train.lr`: Main optimization parameters: epochs, per-device batch size, gradient accumulation, and learning rate.
- `train.field_norm`: Field normalization factor. The H5 `y` tensor is not pre-divided by this value; the training loss applies it to stabilize the scale of electromagnetic fields with localized enhancement.
- `train.grad_weight`: Weight of the spatial-gradient loss, used to regularize field variations near boundaries and hotspots.
- `train.cache_clear_interval`: Interval for clearing device memory cache during long DDP training runs.
- `train_loader.*`, `val_loader.*`: DataLoader parallelism settings such as `num_workers`, `prefetch_factor`, and `persistent_workers`.
- `optimizer.*`, `scheduler.*`: Optimizer and learning-rate schedule parameters.

### 2. Search Config | 搜索配置

文件: `configs/search/ga.yaml`

**中文**

- `surrogate_checkpoint`: GA 调用的 MetaField surrogate 权重路径，默认期望 `checkpoints/best_model.pth`。
- `base_channels`, `heads`, `max_dist`, `cond_embed_dim`, `transformer_depth`: surrogate 模型结构参数，必须与训练 checkpoint 完全匹配。
- `population_size`, `generations`, `elite_count`: GA 种群大小、迭代代数和精英保留数量。
- `tournament_size`: 锦标赛选择规模，影响选择压力。
- `crossover_rate`: 几何和材料 genome 发生杂交的概率。
- `mutation_rate_geo`: 几何与波长基因突变概率。
- `mutation_rate_mat`: 材料堆叠基因突变概率。
- `freq_range`: 搜索波段对应的频率/波矢参数范围。宽带搜索时，激发波长等效参与 genome 优化。
- `r_top_range`, `r_bot_range`, `height_range`, `period_range`: 纳米结构几何变量搜索范围，单位为 nm。
- `min_gap_nm`: 周期内相邻结构之间允许的最小间隙。
- `min_block_cells`: 材料连续层的最小 slot 数，用于避免过薄的非物理材料层。
- `max_material_transitions`: 材料交界面最大变化次数；例如 3 表示最多 4 个连续材料块。
- `allowed_materials`, `adhesion_materials`, `material_stack.*`: 材料编码、底层约束、顶层约束、最大 active slots 和 flow padding 规则。
- `field_norm`: surrogate 输出反归一化系数，应与训练配置保持一致。
- `batch_size`: surrogate 批量评估大小，影响 GA fitness 计算吞吐。
- `devices`: 指定推理设备；为空时自动选择 CUDA、MPS 或 CPU。

**English**

- `surrogate_checkpoint`: Path to the MetaField surrogate weights used by GA, typically `checkpoints/best_model.pth`.
- `base_channels`, `heads`, `max_dist`, `cond_embed_dim`, `transformer_depth`: Surrogate architecture parameters. They must exactly match the training checkpoint.
- `population_size`, `generations`, `elite_count`: Population size, number of GA generations, and number of elite individuals retained each generation.
- `tournament_size`: Tournament selection size; larger values increase selection pressure.
- `crossover_rate`: Probability of crossover for geometry and material genomes.
- `mutation_rate_geo`: Mutation probability for geometry and wavelength genes.
- `mutation_rate_mat`: Mutation probability for material-stack genes.
- `freq_range`: Search range for the excitation frequency / wave-vector parameter. In broadband search, wavelength is optimized as part of the genome.
- `r_top_range`, `r_bot_range`, `height_range`, `period_range`: Search ranges for nanostructure geometry variables in nm.
- `min_gap_nm`: Minimum allowed in-period spacing between neighboring structures.
- `min_block_cells`: Minimum number of contiguous slots per material layer, preventing unrealistically thin layers.
- `max_material_transitions`: Maximum number of material-interface changes. For example, 3 allows at most 4 contiguous material blocks.
- `allowed_materials`, `adhesion_materials`, `material_stack.*`: Material code set, bottom-layer constraints, top-layer constraints, maximum active slots, and flow-padding rules.
- `field_norm`: Output de-normalization factor for surrogate predictions; it should match the training configuration.
- `batch_size`: Batch size for surrogate fitness evaluation during GA.
- `devices`: Explicit inference devices. If empty, CUDA, MPS, or CPU will be selected automatically.

### 3. Benchmark Config | 基准配置

文件: `configs/bench/throughput.yaml`

**中文**

- `input_shape`: benchmark 使用的输入空间尺寸，通常对应 H5/LMDB 中某个 padded group 的 `[Z, X]`。
- `batch_start`, `batch_limit`: batch size 扫描范围；脚本通常按倍增方式寻找吞吐峰值。
- `warmup_iters`: 预热迭代次数，用于排除首次 kernel 编译、缓存初始化等开销。
- `timed_iters`: 正式计时迭代次数。
- `device`: benchmark 设备，例如 `cuda`、`mps` 或 `cpu`。
- `dtype`: 推理精度设置，例如 `float32`、`float16` 或 `bfloat16`，取决于硬件支持。
- `min_safety_gb`: 运行时保留的安全显存/内存余量，避免 batch 扫描触发 OOM。
- `max_stall`: 当吞吐提升进入平台期时提前停止 batch 扫描的阈值。
- `fdtd_seconds`, `ga_population`, `ga_generations`: 用于估算 surrogate 相比传统 FDTD 和大规模 GA 搜索的 paper-style 加速比。

**English**

- `input_shape`: Spatial input size used by benchmark, usually matching the `[Z, X]` shape of a padded H5/LMDB group.
- `batch_start`, `batch_limit`: Batch-size scan range. The benchmark typically doubles the batch size to find peak throughput.
- `warmup_iters`: Warmup iterations used to exclude one-time kernel compilation and cache initialization overhead.
- `timed_iters`: Timed iterations used for throughput measurement.
- `device`: Benchmark device, such as `cuda`, `mps`, or `cpu`.
- `dtype`: Inference precision, such as `float32`, `float16`, or `bfloat16`, depending on hardware support.
- `min_safety_gb`: Reserved memory margin to avoid OOM during batch scanning.
- `max_stall`: Early-stop threshold when throughput improvement reaches a plateau.
- `fdtd_seconds`, `ga_population`, `ga_generations`: Parameters used to estimate paper-style speedup relative to conventional FDTD and large-scale GA search.

### 4. Material Fitting Config | 材料拟合配置

文件: `configs/materials/drude_lorentz_fit.yaml`

**中文**

- `input_file`: 原始光学常数表路径，通常包含波长、`n` 和 `k`。
- `material_name`: 写入材料数据库的材料名称，需与后续材料编码保持一致。
- `input_wavelength_unit`: 输入波长单位，当前支持 `um` 或 `nm`。
- `skiprows`: 读取光学常数表时跳过的表头行数。
- `num_oscillators`: Lorentz 振子数量。更多振子可以提升拟合自由度，但也可能过拟合。
- `k_weight`: 消光系数 `k` 项误差权重，用于平衡 `n` 与 `k` 的拟合优先级。
- `min_wavelength_nm`, `max_wavelength_nm`: 拟合使用的波长范围，应覆盖训练/搜索数据的目标波段。
- `materials_json`: Drude-Lorentz 拟合参数写入位置。
- `material_mapping_json`: 材料名称与整数编码映射文件。
- `default_refractive_index`: 新材料首次注册时写入 mapping 的默认折射率占位值。
- `plot_output`: 若非空，则输出拟合曲线图，用于检查拟合质量。

**English**

- `input_file`: Path to the raw optical-constant table, typically containing wavelength, `n`, and `k`.
- `material_name`: Material name written to the material database. It should stay consistent with material encoding.
- `input_wavelength_unit`: Wavelength unit of the input table; currently `um` and `nm` are supported.
- `skiprows`: Number of header rows skipped when reading the optical-constant table.
- `num_oscillators`: Number of Lorentz oscillators. More oscillators increase fitting flexibility but may overfit.
- `k_weight`: Error weight for the extinction coefficient `k`, balancing the fitting priority between `n` and `k`.
- `min_wavelength_nm`, `max_wavelength_nm`: Wavelength range used for fitting; it should cover the target training/search band.
- `materials_json`: Output path for fitted Drude-Lorentz parameters.
- `material_mapping_json`: Mapping file between material names and integer material codes.
- `default_refractive_index`: Placeholder refractive index written to the mapping file when a new material is first registered.
- `plot_output`: Optional output path for the fitting diagnostic plot.

### Configuration Rule of Thumb | 配置使用建议

**中文**

- 同一份 surrogate 权重对应的 `model.*` 参数，在 training、search 和 benchmark 中应保持一致。
- 搜索配置中的 surrogate 参数若与训练配置不一致，通常会导致权重加载失败或行为错误。
- benchmark 配置应聚焦输入规模、设备条件和测量策略，而不是训练目标或 GA 搜索目标。
- 材料拟合配置应聚焦输入数据格式、拟合范围和输出落盘位置，而不是 surrogate 或 GA 参数。
- 数据路径、checkpoint 路径和输出目录应使用本机绝对路径或清晰的项目相对路径，避免把个人存储路径硬编码进公开配置。

**English**

- The `model.*` parameters associated with the same surrogate checkpoint should remain consistent across training, search, and benchmark configs.
- If surrogate parameters in search configs do not match the training config, checkpoint loading will usually fail or produce invalid behavior.
- Benchmark configs should focus on input size, hardware settings, and measurement protocol rather than training objectives or GA targets.
- Material-fitting configs should focus on input data format, fitting wavelength range, and output paths rather than surrogate or GA parameters.
- Data paths, checkpoint paths, and output directories should use machine-local absolute paths or clear project-relative paths; avoid publishing private storage paths as defaults.

---

## Usage | 运行方式

**Training / 训练**

```bash
torchrun --nproc_per_node=2 -m scripts.train.train --config configs/train/metafield_ddp.yaml
```

**GA Search / GA 搜索**

```bash
python3 -m scripts.search.run_ga --config configs/search/ga.yaml
```

**Throughput Benchmark / 吞吐基准**

```bash
python3 -m scripts.bench.benchmark_throughput --config configs/bench/throughput.yaml
```

**Material Fitting / 材料拟合**

```bash
python3 -m scripts.data.fit_material_drude_lorentz --config configs/materials/drude_lorentz_fit.yaml
```

---

## Data Preparation Scripts | 数据预处理脚本

数据准备相关脚本位于 [scripts/data](/Users/chenmingqian/Code/MetaField/scripts/data)。

- [convert_h5_to_lmdb.py](/Users/chenmingqian/Code/MetaField/scripts/data/convert_h5_to_lmdb.py)
- [data_builder.py](/Users/chenmingqian/Code/MetaField/scripts/data/data_builder.py)
- [patch_phase.py](/Users/chenmingqian/Code/MetaField/scripts/data/patch_phase.py)
- [smart_update_h5.py](/Users/chenmingqian/Code/MetaField/scripts/data/smart_update_h5.py)
- [stratified_splitter.py](/Users/chenmingqian/Code/MetaField/scripts/data/stratified_splitter.py)
- [fit_material_drude_lorentz.py](/Users/chenmingqian/Code/MetaField/scripts/data/fit_material_drude_lorentz.py)

---

## Development Notes | 开发说明

**中文**
如果需要扩展模型、搜索后端或材料体系，建议优先沿用当前模块边界：

- 新模型放入 `models/`
- 新数据读取或 sampler 放入 `datasets/`
- 新结构编码或工艺约束放入 `structure/`
- 新搜索后端放入 `search/`
- 新 benchmark 放入 `benchmarks/`

**English**
When extending the project, keep the current module boundaries:

- new models go to `models/`
- new dataset readers or samplers go to `datasets/`
- new structure encoders or fabrication constraints go to `structure/`
- new search backends go to `search/`
- new benchmarks go to `benchmarks/`

---

## Testing | 测试

```bash
python3 -m pytest tests -q
```

当前 smoke tests 覆盖：

- model forward shape
- LMDB dataset interface
- structure encoding
- material fitting record update


## Release Information | 发布信息

**中文**

- 华中科技大学刘钢教授课题组: http://faculty.hust.edu.cn/liugang5/zh_CN/index.htm
- 本工作通讯作者胡文君副教授: http://faculty.hust.edu.cn/HUWENJUN/zh_CN/index.htm
- 联系方式: hu_wenjun【at】hust【dot】edu【dot】cn

**English**

- Liu Gang Research Group, Huazhong University of Science and Technology: http://faculty.hust.edu.cn/liugang5/zh_CN/index.htm
- Corresponding author: Associate Professor Wenjun Hu: http://faculty.hust.edu.cn/HUWENJUN/zh_CN/index.htm
- Contact: hu_wenjun [at] hust [dot] edu [dot] cn

## Manuscript Draft | 论文草稿

**中文**
当前论文草稿见 [MetaField-Draft](MetaField-Draft6.pdf)。

**English**
The current manuscript draft is available at [MetaField-Draft](MetaField-Draft6.pdf).

## Training Data | 训练数据集

**中文**
训练数据和对应的FAE已经上传至[ScienceDB](https://doi.org/10.57760/sciencedb.37386)。

**English**
Traning database and corresponding FAE are avaliable on [ScienceDB](https://doi.org/10.57760/sciencedb.37386).
