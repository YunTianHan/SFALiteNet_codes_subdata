# SFA-LiteNet

本仓库用于 **SFA-LiteNet 项目开源**，实现岩石裂隙图像的二分类语义分割，包含完整的数据准备、模型、联合损失、训练、验证、测试、检查点和推理流程。

## 数据及使用边界

项目使用 `../dataset` 中已经提供的**真实数据集子集，不代表完整原始数据集**，不生成或混入模拟图像。

```text
github/
├── dataset/                 # 原始数据，只读使用
│   ├── images/001.jpg ... 050.jpg
│   └── masks/001.png ... 050.png
└── codes/                   # 本项目
```

白色为裂隙，黑色为背景。

### 划分与泄漏控制

先划分原始样本，再仅在训练集合内在线增强；不提前生成增强图像。默认目标比例为 70%/15%/15%，固定 `split_seed: 2026`，五次训练共用一份划分。

```csv
item_id,source_id,section_id
```

`item_id` 必须保留 `001` 等前导零；`source_id` 为全局唯一原图编号，`section_id` 为真实采集区段编号。有 CSV 时，同一原图或同一区段的样本都会合并，不会跨训练、验证、测试集合；重复/近重复合并仍然保留。

准备阶段会核查配对、尺寸、标签、原文件 SHA256、解码像素哈希、来源字段和近重复关系。相同图像若标签冲突会直接报错；合并后不足三个组也会报错，不会强拆来源组。运行训练/评估时会重新核对数据和完整清单；文件、来源或划分配置发生变化时拒绝静默复用。需明确执行 `--overwrite` 才会重新生成清单，已有检查点仍保存原划分快照。近重复比较采用全对比较，适合当前数据规模。

## 目录

```text
codes/
├── configs/default.yaml     # 模型、路径、增强、损失与训练参数
├── sfa_litenet/
│   ├── config.py            # 配置合并、字段检查与路径解析
│   ├── data.py              # 配对读取、来源分组、划分、增强与距离场
│   ├── models/
│   │   ├── modules.py       # DSC、Haar、ECA、SFF、CA、边界门控、LAD
│   │   └── network.py       # SFA-LiteNet 完整网络
│   ├── losses.py            # Focal + Dice + Boundary + soft-clDice
│   ├── metrics.py           # 逐图区域指标及欧氏距离 BF1
│   ├── engine.py            # 训练、评估、早停、恢复与多种子汇总
│   └── utils.py             # 随机状态、设备、日志、检查点和文件输出
├── prepare_data.py          # 读取真实数据并生成划分清单
├── train.py                 # 单种子或五种子从零训练
├── evaluate.py              # 验证与测试
├── infer.py                 # 单图/目录推理
├── benchmark.py             # 真实图像上的参数量、FLOPs 与 FP32 计时
├── requirements.txt
└── LICENSE                 # 代码 Apache-2.0；不为数据授予许可
```

## 模型结构

默认输入为 `B×3×512×512`。四级编码器输出依次为 `B×32×512×512`、`B×64×256×256`、`B×128×128×128` 和 `B×256×64×64`。首级为 DSC，后三阶段为步幅 2 的残差 DSC。投影至 `B×512×64×64` 后进入 SFF，不额外下采样。三级 LAD 分别融合第三、第二、第一级跳连，输出 128、64、32 通道，最后 1×1 卷积得到 `B×1×512×512`。

SFF 空间分支为 DSC+ECA；频率分支使用正交 Haar 滤波器 `[1,1]/sqrt(2)` 和 `[1,-1]/sqrt(2)`，得到 LL/LH/HL/HH。每个子带的平方响应在空间轴求均值，得到 `B×4×C` 能量描述；共享两层 MLP 后沿**四子带维度** Softmax，各通道权重之和为 1。加权子带经逆 Haar 重建，与空间特征拼接、1×1 卷积压缩并加输入残差。不以固定高频权重替代能量门控。

LAD 执行双线性上采样、跳连拼接、DSC 与坐标注意力。边界门从**当前级编码器跳连**重新计算 Haar 高频子带，经 3×3 卷积、BN、SiLU、1×1 卷积得到单通道门，插值至解码尺寸，再计算 `D * sigmoid(G) + D`；不复用瓶颈频率特征。

训练接口返回未激活 logits，以稳定计算损失；评估和推理统一显式 sigmoid，`predict_proba()` 提供直接概率接口。联合损失的默认权重为 Focal/Dice/Boundary/clDice **1/1/0.01/1**，软骨架迭代 10 次。边界损失是预测概率与 GT 有符号距离场的均值，内部负、外部正；损失可为负，这不是计算错误。

本项目实现 SFA-LiteNet 及其完整使用流程。

## 环境与依赖安装

项目支持 Python 3.10 或 3.11。PyTorch 和 CUDA 请根据操作系统、显卡驱动及硬件选择匹配的安装包；CPU 也可以运行，但训练速度通常较慢。在 `github/codes` 内打开终端并安装项目依赖：

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` 给出项目依赖范围。若需要 CUDA，请先按 PyTorch 官方安装说明选择匹配的发行包；`fvcore` 仅用于 `benchmark.py` 的 FLOPs 统计，主训练不依赖其导入。

Windows 默认 `num_workers: 0`；可按实际资源调整。默认 FP32，不启用 AMP、预训练或额外学习目标。确定性模式设置随机种子、cuDNN 与确定性算法；若某版本算子不支持确定性，PyTorch 会报错。跨平台或跨版本仍不保证逐位一致。可明确配置 `deterministic: false`，并记录这一变化。

## 使用流程

所有配置文件及命令参数中的相对路径均以 **`github/codes` 根目录**解析，避免从不同终端目录启动时路径含义改变。命令本身请在 `github/codes` 运行，或使用脚本绝对路径。

1. 根据数据实际来源准备 `configs/default.yaml`。有来源记录时提供完整 CSV，并设置 `metadata_csv`；没有来源信息时使用默认的开发划分方式。
2. 核对并准备清单：

   ```powershell
   python prepare_data.py --config configs/default.yaml
   ```

   检查 `outputs/splits/split.json` 的 `counts`、`provenance.merges`、组和样本归属。改动来源、标签或划分规则后执行 `python prepare_data.py --overwrite`；同一实验中保持清单不变。

3. 先训练单个种子，或直接执行五种子完整重复：

   ```powershell
   python train.py --config configs/default.yaml --seed 42
   python train.py --config configs/default.yaml --all-seeds
   ```

   这两条是两种选择；已经完成单种子后直接 `--all-seeds` 会因已有非空运行目录停止，以防覆盖。需要独立实验时复制配置并更改 `project.experiment`。默认 batch=8，最大 100 轮，AdamW 初始 lr=1e-3、weight decay=1e-4，预热 5 轮，验证 IoU 连续 15 轮未改善早停，保留最佳权重。

4. 中断后从 `last.pt` 继续同一运行：

   ```powershell
   python train.py --config configs/default.yaml --seed 42 --resume outputs/sfa_litenet/seed_42/last.pt
   ```

   不修改优化、模型、损失、增强、预处理和评估配置；恢复包含优化器、轮数、早停状态、最佳模型快照和随机状态。已完成计划或已早停的检查点不会自动续训。`best.pt` 为评估/推理检查点，不支持继续训练；`weights.pt` 为纯最佳 `state_dict`。

5. 用最佳模型做验证与独立测试：

   ```powershell
   python evaluate.py --checkpoint outputs/sfa_litenet/seed_42/best.pt --split val
   python evaluate.py --checkpoint outputs/sfa_litenet/seed_42/best.pt --split test
   python evaluate.py --all-seeds --split test
   ```

   五种子命令要求五个检查点均存在、使用相同真实数据和划分。训练不会自动以测试指标挑选模型。评估采用检查点保存的模型和归一化设置，当前配置提供数据位置、设备及评估选项。

6. 推理与可选效率统计：

   ```powershell
   python infer.py --checkpoint outputs/sfa_litenet/seed_42/best.pt --input ../dataset/images/001.jpg --output outputs/inference_001
   python infer.py --checkpoint outputs/sfa_litenet/seed_42/best.pt --input ../dataset/images --output outputs/inference_subset
   python benchmark.py --checkpoint outputs/sfa_litenet/seed_42/best.pt
   ```

## 配置、指标和输出

配置支持只写需要覆盖的字段，其余继承 `configs/default.yaml`；未知字段直接报错。关键字段包括：

| 配置 | 含义与默认值 |
|---|---|
| `data.root` | `../dataset`，只读原始数据 |
| `data.split_manifest` | `outputs/splits/split.json`，固定划分 |
| `data.image_size` | `[512,512]`，顺序为高、宽 |
| `data.metadata_csv` | 可选真实来源 CSV，默认无 |
| `project.device` | `auto`、`cpu`、`cuda` 或 `cuda:N` |
| `project.experiment` | 独立实验名，默认 `sfa_litenet` |
| `model.*` | 全部通道数及模块缩减率 |
| `loss.*` | 四项权重、Focal 参数、骨架迭代次数 |
| `training.*` | 随机种子、优化器、学习率计划、早停等 |
| `evaluation.threshold` | 0.5，前景阈值 |
| `evaluation.boundary_tolerance` | 2 像素，欧氏距离匹配 |
| `benchmark.*` | 每轮预热 50 次、计时 200 次、重复 5 轮 |

指标值存为 0–1 浮点数，显示百分数时自行乘以 100。主指标是前景 IoU，不是背景/前景 mIoU。BF1 匹配以欧氏距离变换实现，不用 5×5 方形邻域替代半径 2 的圆形容差。

运行后，`outputs/sfa_litenet/seed_42/` 等目录包含：`config.json`、`split.json`、`environment.json`、`run.log`、`history.csv/json`、`last.pt`、`best.pt`、`weights.pt`，以及 `validation/`、`evaluation_val/`、`evaluation_test/` 的 `metrics.json`、`per_image.csv` 和可选预测掩膜。多种子另存 `validation_summary.json`、`test_summary.json`。

参数量按实际模型计算；权重体积采用纯 FP32 `state_dict` 的序列化大小，十进制 MB。FLOPs 使用 fvcore 默认算子口径。效率统计使用真实验证图像，batch=1、FP32、eval、关闭梯度，CUDA 在计时边界同步；输出五轮全部速度与汇总，仅计模型前向。本脚本明确保存全部轮次。

## 许可和来源说明

Apache License 2.0，见 [LICENSE](LICENSE)。数据集不随代码许可，其公开分发权利需由数据发布者确认。

数据来源为 Xinrong Liu 提供的公开子集；完整数据集未随本项目发布。
