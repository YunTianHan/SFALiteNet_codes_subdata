# SFA-LiteNet

本仓库用于 **SFA-LiteNet 项目开源**，实现岩石裂隙图像的二分类语义分割，包含完整的数据准备、模型、联合损失、训练、验证、测试、检查点和推理流程。

SFA-LiteNet is an open-source implementation of a binary semantic segmentation model for rock fracture images. The project covers data preparation, model training with a combined loss, validation, testing, checkpointing, and inference.

## 数据及使用边界 / Dataset

项目使用 `../dataset` 中已经提供的**真实数据集子集，不代表完整原始数据集**，不生成或混入模拟图像。

The project uses the real rock fracture dataset subset provided in `../dataset`. The release contains only part of the original dataset and uses no simulated images.

```text
github/
├── dataset/                 # 原始数据，只读使用 / Read-only source data
│   ├── images/001.jpg ... 050.jpg
│   └── masks/001.png ... 050.png
└── codes/                   # 本项目 / Project code
```

白色为裂隙，黑色为背景。

White mask pixels indicate fractures, and black pixels indicate background.

### 划分与泄漏控制 / Data splitting and leakage control

先划分原始样本，再仅在训练集合内在线增强；不提前生成增强图像。默认目标比例为 70%/15%/15%，固定 `split_seed: 2026`，五次训练共用一份划分。

The preparation script splits the original samples before augmentation. Augmentation runs on the fly during training. The target split is 70% training, 15% validation, and 15% testing. All five training runs share one split generated with `split_seed: 2026`.

如有采集来源记录，请提供包含以下表头的 CSV 文件。

If provenance records are available, supply a CSV file with the following header.

```csv
item_id,source_id,section_id
```

`item_id` 必须保留 `001` 等前导零；`source_id` 为全局唯一原图编号，`section_id` 为真实采集区段编号。有 CSV 时，同一原图或同一区段的样本都会合并，不会跨训练、验证、测试集合；重复/近重复合并仍然保留。

Preserve leading zeros in `item_id` (for example, `001`). Use a globally unique identifier for each original image in `source_id` and the recorded acquisition section identifier in `section_id`. Samples that share either a source image or an acquisition section are assigned to the same split. The script also groups duplicates and near-duplicates.

准备阶段会核查配对、尺寸、标签、原文件 SHA256、解码像素哈希、来源字段和近重复关系。相同图像若标签冲突会直接报错；合并后不足三个组也会报错，不会强拆来源组。运行训练/评估时会重新核对数据和完整清单；文件、来源或划分配置发生变化时拒绝静默复用。需明确执行 `--overwrite` 才会重新生成清单，已有检查点仍保存原划分快照。近重复比较采用全对比较，适合当前数据规模。

Data preparation checks image–mask pairing, dimensions, labels, SHA-256 hashes of the source files, hashes of decoded pixels, provenance fields, and near-duplicate relationships. It raises an error if identical images have conflicting labels or if grouping leaves fewer than three independent groups. Near-duplicate detection compares all pairs, which is practical for this subset.

Training and evaluation recheck the dataset and the full split manifest. Changes to files, provenance, or split settings require explicit regeneration of the manifest with `--overwrite`. Existing checkpoints retain their original split snapshots.

## 目录 / Project layout

```text
codes/
├── configs/default.yaml     # 模型、路径、增强、损失与训练参数 / Model, paths, augmentation, loss, and training settings
├── sfa_litenet/
│   ├── config.py            # 配置合并、字段检查与路径解析 / Configuration merging, field validation, and path resolution
│   ├── data.py              # 配对读取、来源分组、划分、增强与距离场 / Paired data, provenance groups, splits, augmentation, distance fields
│   ├── models/
│   │   ├── modules.py       # DSC、Haar、ECA、SFF、CA、边界门控、LAD / DSC, Haar, ECA, SFF, CA, boundary gating, and LAD
│   │   └── network.py       # SFA-LiteNet 完整网络 / SFA-LiteNet architecture
│   ├── losses.py            # Focal + Dice + Boundary + soft-clDice / Focal, Dice, boundary, and soft-clDice losses
│   ├── metrics.py           # 逐图区域指标及欧氏距离 BF1 / Per-image region metrics and Euclidean boundary F1
│   ├── engine.py            # 训练、评估、早停、恢复与多种子汇总 / Training, evaluation, early stopping, resumption, and seed summaries
│   └── utils.py             # 随机状态、设备、日志、检查点和文件输出 / Random states, devices, logging, checkpoints, and file output
├── prepare_data.py          # 读取真实数据并生成划分清单 / Validate source data and generate the split manifest
├── train.py                 # 单种子或五种子从零训练 / Train from scratch with one or five seeds
├── evaluate.py              # 验证与测试 / Run validation or testing
├── infer.py                 # 单图/目录推理 / Run inference on an image or a directory
├── benchmark.py             # 真实图像上的参数量、FLOPs 与 FP32 计时 / Count parameters and FLOPs, and time FP32 inference on real images
├── requirements.txt
└── LICENSE                 # 代码 Apache-2.0；不为数据授予许可 / Apache-2.0 license for the code
```

## 模型结构 / Model and loss

默认输入为 `B×3×512×512`。四级编码器输出依次为 `B×32×512×512`、`B×64×256×256`、`B×128×128×128` 和 `B×256×64×64`。首级为 DSC，后三阶段为步幅 2 的残差 DSC。投影至 `B×512×64×64` 后进入 SFF，不额外下采样。三级 LAD 分别融合第三、第二、第一级跳连，输出 128、64、32 通道，最后 1×1 卷积得到 `B×1×512×512`。

With the default input shape of `B×3×512×512`, the four encoder stages produce `B×32×512×512`, `B×64×256×256`, `B×128×128×128`, and `B×256×64×64` feature maps. The first stage uses depthwise separable convolution (DSC). The remaining stages use residual DSC with stride 2.
The encoder output is projected to `B × 512 × 64 × 64` and passed to the SFF module at that resolution. Three LAD decoder stages fuse skip connections from encoder stages 3, 2, and 1 and produce 128, 64, and 32 channels, respectively. A final 1 × 1 convolution produces a single-channel logit map of shape `B × 1 × 512 × 512`.

SFF 空间分支为 DSC+ECA；频率分支使用正交 Haar 滤波器 `[1,1]/sqrt(2)` 和 `[1,-1]/sqrt(2)`，得到 LL/LH/HL/HH。每个子带的平方响应在空间轴求均值，得到 `B×4×C` 能量描述；共享两层 MLP 后沿**四子带维度** Softmax，各通道权重之和为 1。加权子带经逆 Haar 重建，与空间特征拼接、1×1 卷积压缩并加输入残差。不以固定高频权重替代能量门控。

**SFF.** The spatial branch applies DSC followed by efficient channel attention (ECA). The frequency branch uses orthogonal Haar filters `[1, 1] / sqrt(2)` and `[1, -1] / sqrt(2)` to obtain the LL, LH, HL, and HH subbands. Averaging the squared responses over the spatial axes gives an energy tensor of shape `B × 4 × C`. A shared two-layer MLP and a softmax over the four subbands yield energy-dependent weights that sum to one for each channel. An inverse Haar transform reconstructs the weighted features. The module concatenates them with the spatial features, compresses the result with a 1 × 1 convolution, and adds the input residual.

LAD 执行双线性上采样、跳连拼接、DSC 与坐标注意力。边界门从**当前级编码器跳连**重新计算 Haar 高频子带，经 3×3 卷积、BN、SiLU、1×1 卷积得到单通道门，插值至解码尺寸，再计算 `D * sigmoid(G) + D`；不复用瓶颈频率特征。

**LAD.** Each decoder stage applies bilinear upsampling, concatenates the corresponding encoder skip features, and applies DSC and coordinate attention. The boundary gate recomputes high-frequency Haar subbands from that stage’s encoder skip features. A 3 × 3 convolution, batch normalization, SiLU, and a 1 × 1 convolution produce a single-channel gate. After resizing the gate `G` to match the decoder feature map `D`, the stage returns `D * sigmoid(G) + D`.

训练接口返回未激活 logits，以稳定计算损失；评估和推理统一显式 sigmoid，`predict_proba()` 提供直接概率接口。联合损失的默认权重为 Focal/Dice/Boundary/clDice **1/1/0.01/1**，软骨架迭代 10 次。边界损失是预测概率与 GT 有符号距离场的均值，内部负、外部正；损失可为负，这不是计算错误。

**Training objective.** The model returns logits for loss computation. Evaluation and inference apply a sigmoid, and `predict_proba()` returns probabilities directly. The default weights for focal, Dice, boundary, and soft-clDice losses are 1, 1, 0.01, and 1, respectively. Soft skeletonization uses 10 iterations. The boundary loss averages the product of the predicted probability and the signed distance to the ground-truth boundary. Distances are negative inside the fracture and positive outside, so this loss can be negative.

## 环境与依赖安装 / Installation

项目支持 Python 3.10 或 3.11。PyTorch 和 CUDA 请根据操作系统、显卡驱动及硬件选择匹配的安装包；CPU 也可以运行，但训练速度通常较慢。在 `github/codes` 内打开终端并安装项目依赖：

Use Python 3.10 or 3.11. For GPU training, first install a PyTorch build compatible with your operating system, GPU, and driver according to the PyTorch installation instructions. CPU execution is also supported, although training is typically slower. From `github/codes`, install the dependencies within the version ranges specified in `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` 给出项目依赖范围。若需要 CUDA，请先按 PyTorch 官方安装说明选择匹配的发行包；`fvcore` 仅用于 `benchmark.py` 的 FLOPs 统计，主训练不依赖其导入。

`requirements.txt` specifies the supported dependency versions. For CUDA, first select a compatible build using the PyTorch installation instructions. Only the FLOP count in `benchmark.py` requires `fvcore`.

Windows 默认 `num_workers: 0`；可按实际资源调整。默认 FP32，不启用 AMP、预训练或额外学习目标。确定性模式设置随机种子、cuDNN 与确定性算法；若某版本算子不支持确定性，PyTorch 会报错。跨平台或跨版本仍不保证逐位一致。可明确配置 `deterministic: false`，并记录这一变化。

On Windows, the data loader defaults to `num_workers: 0`. Adjust it to suit the available resources.
Training defaults to FP32 with no automatic mixed precision, pretraining, or auxiliary objectives. Deterministic mode sets the random seeds and configures cuDNN and PyTorch to use deterministic algorithms. PyTorch raises an error if the installed version lacks a deterministic implementation for a required operation. Results can still vary across platforms or software versions. If you set `deterministic: false`, record the change with the experiment.

## 使用流程 / Usage

所有配置文件及命令参数中的相对路径均以 **`github/codes` 根目录**解析，避免从不同终端目录启动时路径含义改变。命令本身请在 `github/codes` 运行，或使用脚本绝对路径。

Run these commands from `github/codes`, or invoke the scripts by absolute path. The scripts resolve relative paths in configuration files and command-line arguments against the `github/codes` project root.

### 1. 配置数据集 / Configure the dataset

根据数据实际来源准备 `configs/default.yaml`。有来源记录时提供完整 CSV，并设置 `metadata_csv`；没有来源信息时使用默认的开发划分方式。

Set the data options in `configs/default.yaml`. If provenance records are available, provide a complete CSV and set `data.metadata_csv` to its path. If provenance is unavailable, use the default split for development.

### 2. 生成并核对划分清单 / Prepare and inspect the split

```powershell
python prepare_data.py --config configs/default.yaml
```

检查 `outputs/splits/split.json` 的 `counts`、`provenance.merges`、组和样本归属。改动来源、标签或划分规则后执行 `python prepare_data.py --overwrite`；同一实验中保持清单不变。

Review `counts`, `provenance.merges`, and the group and sample assignments in `outputs/splits/split.json`. After changing provenance records, labels, or split rules, regenerate the manifest with `python prepare_data.py --overwrite`. Keep the manifest fixed throughout an experiment.

### 3. 训练模型 / Train the model

使用种子 42 进行单次训练。

For a single run with seed 42, use

```powershell
python train.py --config configs/default.yaml --seed 42
```

如需从头运行全部五个种子，请使用新的实验目录。

To train all five seeds from scratch, start with a fresh experiment and run

```powershell
python train.py --config configs/default.yaml --all-seeds
```

这两条是两种选择；已经完成单种子后直接 `--all-seeds` 会因已有非空运行目录停止，以防覆盖。需要独立实验时复制配置并更改 `project.experiment`。默认 batch=8，最大 100 轮，AdamW 初始 lr=1e-3、weight decay=1e-4，预热 5 轮，验证 IoU 连续 15 轮未改善早停，保留最佳权重。

Training stops if a target run directory already contains files. If you have already trained one seed and want a separate five-seed experiment, copy the configuration, change `project.experiment`, and pass the new configuration to `--config`.

The defaults are a batch size of 8, a maximum of 100 epochs, and AdamW with an initial learning rate of 0.001, weight decay of 0.0001, and a five-epoch warmup. Training retains the weights with the best validation IoU and stops after 15 consecutive epochs without improvement.

### 4. 恢复中断的训练 / Resume an interrupted run

```powershell
python train.py --config configs/default.yaml --seed 42 --resume outputs/sfa_litenet/seed_42/last.pt
```

不修改优化、模型、损失、增强、预处理和评估配置；恢复包含优化器、轮数、早停状态、最佳模型快照和随机状态。已完成计划或已早停的检查点不会自动续训。`best.pt` 为评估/推理检查点，不支持继续训练；`weights.pt` 为纯最佳 `state_dict`。

Use `last.pt` with the original optimization, model, loss, augmentation, preprocessing, and evaluation settings. It restores the optimizer, epoch, early-stopping state, best-model snapshot, and random states. Resumption applies to interrupted runs. Runs that reached the planned epoch limit or triggered early stopping stay complete. Use `best.pt` for evaluation and inference. The `weights.pt` file contains only the best model’s `state_dict`.

### 5. 在验证集和测试集上评估最佳模型 / Evaluate the best checkpoint on the validation and test sets

```powershell
python evaluate.py --checkpoint outputs/sfa_litenet/seed_42/best.pt --split val
python evaluate.py --checkpoint outputs/sfa_litenet/seed_42/best.pt --split test
python evaluate.py --all-seeds --split test
```

五种子命令要求五个检查点均存在、使用相同真实数据和划分。训练不会自动以测试指标挑选模型。评估采用检查点保存的模型和归一化设置，当前配置提供数据位置、设备及评估选项。

Five-seed evaluation requires all five checkpoints to share the same source data and split. Training selects checkpoints using validation IoU and reserves the test split for independent evaluation. Evaluation reads the model and normalization settings from the checkpoint and the data location, device, and evaluation options from the current configuration.

### 6. 推理与效率测试 / Run inference and benchmark the model

```powershell
python infer.py --checkpoint outputs/sfa_litenet/seed_42/best.pt --input ../dataset/images/001.jpg --output outputs/inference_001
python infer.py --checkpoint outputs/sfa_litenet/seed_42/best.pt --input ../dataset/images --output outputs/inference_subset
python benchmark.py --checkpoint outputs/sfa_litenet/seed_42/best.pt
```

## 配置 / Configuration

配置支持只写需要覆盖的字段，其余继承 `configs/default.yaml`；未知字段直接报错。关键字段包括：

Custom configuration files inherit unspecified values from `configs/default.yaml`. Unknown fields raise an error.

| 配置项 / Setting | 含义与默认值 | Meaning and default |
| --- | --- | --- |
| `data.root` | `../dataset`，只读原始数据 | Read-only source dataset, default `../dataset` |
| `data.split_manifest` | `outputs/splits/split.json`，固定划分 | Fixed split manifest, default `outputs/splits/split.json` |
| `data.image_size` | `[512,512]`，顺序为高、宽 | Height and width, default `[512, 512]` |
| `data.metadata_csv` | 可选真实来源 CSV，默认无 | Optional provenance CSV, unset by default |
| `project.device` | `auto`、`cpu`、`cuda` 或 `cuda:N` | `auto`, `cpu`, `cuda`, or `cuda:N` |
| `project.experiment` | 独立实验名，默认 `sfa_litenet` | Experiment name, default `sfa_litenet` |
| `model.*` | 全部通道数及模块缩减率 | Channel counts and module reduction ratios |
| `loss.*` | 四项权重、Focal 参数、骨架迭代次数 | Four loss weights, focal loss parameters, and skeletonization iterations |
| `training.*` | 随机种子、优化器、学习率计划、早停等 | Seeds, optimizer, learning rate schedule, and early stopping |
| `evaluation.threshold` | 0.5，前景阈值 | Foreground probability threshold, default 0.5 |
| `evaluation.boundary_tolerance` | 2 像素，欧氏距离匹配 | Euclidean boundary matching radius, default 2 pixels |
| `benchmark.*` | 每轮预热 50 次、计时 200 次、重复 5 轮 | Five rounds, each with 50 warmup passes and 200 timed passes |

## 指标与输出 / Metrics and outputs

指标值存为 0–1 浮点数，显示百分数时自行乘以 100。主指标是前景 IoU，不是背景/前景 mIoU。BF1 匹配以欧氏距离变换实现，不用 5×5 方形邻域替代半径 2 的圆形容差。

The primary metric is foreground intersection over union (IoU). Boundary F1 (BF1) uses a Euclidean distance transform to match boundaries within a circular tolerance of two pixels by default. Metrics are stored as floating-point values between 0 and 1. Multiply by 100 to report percentages.

运行后，`outputs/sfa_litenet/seed_42/` 等目录包含：`config.json`、`split.json`、`environment.json`、`run.log`、`history.csv/json`、`last.pt`、`best.pt`、`weights.pt`，以及 `validation/`、`evaluation_val/`、`evaluation_test/` 的 `metrics.json`、`per_image.csv` 和可选预测掩膜。多种子另存 `validation_summary.json`、`test_summary.json`。

Each run directory (for example, `outputs/sfa_litenet/seed_42/`) contains `config.json`, `split.json`, `environment.json`, `run.log`, `history.csv`, `history.json`, `last.pt`, `best.pt`, and `weights.pt`. The `validation/`, `evaluation_val/`, and `evaluation_test/` directories hold `metrics.json`, `per_image.csv`, and optional predicted masks. Multi-seed runs also produce `validation_summary.json` and `test_summary.json`.

参数量按实际模型计算；权重体积采用纯 FP32 `state_dict` 的序列化大小，十进制 MB。FLOPs 使用 fvcore 默认算子口径。效率统计使用真实验证图像，batch=1、FP32、eval、关闭梯度，CUDA 在计时边界同步；输出五轮全部速度与汇总，仅计模型前向。本脚本明确保存全部轮次。

The benchmark counts parameters directly from the model and reports the serialized size of the FP32 `state_dict` in decimal MB. FLOP counts follow the default operator conventions in `fvcore`. Timing covers only the forward pass on real validation images, with batch size 1, FP32, evaluation mode, and gradients disabled. On CUDA, the script synchronizes the device at the timing boundaries. It saves the speed from each of the five rounds and a summary across rounds.

## 许可和来源说明 / License and data attribution

Apache License 2.0，见 [LICENSE](LICENSE)。数据集不随代码许可，其公开分发权利需由数据发布者确认。

The code is distributed under the Apache License 2.0. See [LICENSE](LICENSE). The license covers the software only. Confirm data redistribution rights with the dataset publisher.

数据来源为 Xinrong Liu 提供的公开子集；完整数据集未随本项目发布。

Xinrong Liu provided the public subset. The full original dataset is outside the scope of this release.
