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

- `data.lmdb_path`: LMDB 数据集路径。
- `data.split_json`: train / val split 文件路径。
- `model.base_channels`, `model.heads`, `model.max_dist`, `model.cond_embed_dim`: MetaField 主体结构参数。
- `train.save_dir`: checkpoint、history 和训练曲线输出目录。
- `train.pretrain_path`: warm-start 权重路径；为空则从头训练。
- `train.epochs`, `train.batch_size`, `train.grad_accum_steps`, `train.lr`: 训练主超参数。
- `train.field_norm`: 电磁场归一化系数。由于局域热点区域可能出现极端场增强，该系数用于把目标场缩放到更稳定的优化区间。
- `train.grad_weight`: gradient loss 权重，用于约束场空间梯度的一致性。
- `train.cache_clear_interval`: 显存缓存清理周期。
- `train_loader.*`, `val_loader.*`: DataLoader 并行参数。
- `optimizer.*`, `scheduler.*`: 优化器与学习率调度设置。

### 2. Search Config | 搜索配置

文件: `configs/search/ga.yaml`

- `surrogate_checkpoint`: GA 使用的 surrogate 模型权重。
- `base_channels`, `heads`, `max_dist`, `cond_embed_dim`, `transformer_depth`: surrogate 模型结构参数，必须与训练权重匹配。
- `population_size`, `generations`, `elite_count`: GA 主循环规模。
- `tournament_size`, `crossover_rate`, `mutation_rate_geo`, `mutation_rate_mat`: GA 选择、杂交、突变参数。
- `r_top_range`, `r_bot_range`, `height_range`, `period_range`: 几何变量搜索范围。
- `min_gap_nm`, `min_block_cells`, `max_material_transitions`: 结构合法性与工艺约束。
- `field_norm`: surrogate 输出反归一化系数。
- `batch_size`: surrogate 批量评估大小。
- `devices`: 指定推理设备；为空时自动选择。

### 3. Benchmark Config | 基准配置

文件: `configs/bench/throughput.yaml`

- `input_shape`: benchmark 输入的原始空间尺寸。
- `batch_start`, `batch_limit`: batch doubling 的扫描范围。
- `warmup_iters`, `timed_iters`: 预热轮数与计时轮数。
- `device`, `dtype`: benchmark 使用的设备与数值精度。
- `min_safety_gb`: 运行时保留的最小安全显存/内存阈值。
- `max_stall`: 吞吐提升进入平台期时的提前停止阈值。
- `fdtd_seconds`, `ga_population`, `ga_generations`: 用于生成 paper-style 速度对比摘要。

### 4. Material Fitting Config | 材料拟合配置

文件: `configs/materials/drude_lorentz_fit.yaml`

- `input_file`: 原始光学常数表路径。
- `material_name`: 材料名称。
- `input_wavelength_unit`: 输入波长单位，当前支持 `um` 或 `nm`。
- `skiprows`: 输入表头跳过行数。
- `num_oscillators`: Lorentz 振子数量。
- `k_weight`: 消光系数 `k` 项误差的加权因子。
- `min_wavelength_nm`, `max_wavelength_nm`: 拟合波段。
- `materials_json`: 拟合结果写入位置。
- `material_mapping_json`: 材料名称与编码映射文件。
- `default_refractive_index`: 新材料首次注册时在 mapping 中写入的默认 RI。
- `plot_output`: 若非空，则输出拟合曲线图。

### Configuration Rule of Thumb | 配置使用建议

- 同一份 surrogate 权重对应的 `model.*` 参数，在 training、search 和 benchmark 中应保持一致。
- 搜索配置中的 surrogate 参数若与训练配置不一致，通常会导致权重加载失败或行为错误。
- benchmark 配置应聚焦输入规模、设备条件和测量策略，而不是训练目标或 GA 搜索目标。
- 材料拟合配置应聚焦输入数据格式、拟合范围和输出落盘位置，而不是 surrogate 或 GA 参数。

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
- 预训练权重与数据集下载信息即将发布。

**English**

- Liu Gang Research Group, Huazhong University of Science and Technology: http://faculty.hust.edu.cn/liugang5/zh_CN/index.htm
- Corresponding author: Associate Professor Wenjun Hu: http://faculty.hust.edu.cn/HUWENJUN/zh_CN/index.htm
- Contact: hu_wenjun [at] hust [dot] edu [dot] cn
- Pretrained weights and dataset download information will be released soon.

## Manuscript Draft | 论文草稿

**中文**  
当前论文草稿见 [MetaField-Draft4.pdf](MetaField-Draft4.pdf)。

**English**  
The current manuscript draft is available at [MetaField-Draft4.pdf](MetaField-Draft4.pdf).
