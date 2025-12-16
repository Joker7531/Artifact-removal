# cGAN Baseline for EEG Artifact Removal

基于 STFT 域的条件生成对抗网络（cGAN）用于 EEG 信号噪声抑制的基线验证系统。

**主要特性**:
- 0.5-40Hz 带通滤波，关注生理相关频段
- 实部+虚部双通道输入，保留完整复数信息
- 复数距离损失 + 幅度损失的多目标优化
- 训练预热阶段，先优化生成器再对抗训练
- 频域维度自适应处理（补零或裁剪）

## 项目结构

```
cGAN_baseline/
├── data/                    # 数据处理模块
│   ├── __init__.py
│   └── dataset.py          # NPZ 加载、section 划分、STFT 变换、样本构建
├── models/                 # 模型定义
│   ├── __init__.py
│   └── cgan.py            # U-Net Generator + PatchGAN Discriminator
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── trainer.py         # 训练器、损失函数、优化器
│   └── visualization.py   # 评估指标、可视化工具
├── checkpoints/           # 模型保存目录
├── logs/                  # 训练日志
├── results/               # 结果保存目录
├── train.py              # 主训练脚本
├── test.py               # 测试脚本
└── requirements.txt      # 依赖包
```

## 快速开始

### 1. 安装依赖

```bash
cd cGAN_baseline
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train.py --num_epochs 100 --batch_size 32
```

### 3. 测试模型

```bash
python test.py --checkpoint ./checkpoints/checkpoint_epoch_100_best.pth
```

## 数据格式

使用 `dataset_cz_v2.npz` 文件，包含以下字段：

- `raw`: 原始 EEG 信号 (63846588,)
- `clean`: 干净 EEG 信号 (63846588,)
- `artifact`: 伪迹信号 (本阶段不使用)
- `subject_id`: 被试 ID (63846588,)
- `task_id`: 任务 ID (63846588,)
- `section_id`: Session ID (63846588,)

## 模型架构

### Generator: U-Net

- 编码器-解码器结构，带跳跃连接
- 输入: raw STFT (2, F, T) - 通道0: 实部, 通道1: 虚部
- 输出: clean STFT (2, F, T)
- 4 层下采样 + 4 层上采样
- Tanh激活（实部虚部可负）

### Discriminator: PatchGAN

- 条件判别器
- 输入: raw(2通道) + clean/fake(2通道) concatenated (4, F, T)
- 输出: Patch-level 判别结果

## 训练配置

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `n_fft` | 1024 | STFT 窗口大小 |
| `hop_length` | 250 | STFT 跳跃步长 |
| `n_frames` | 64 | 每个样本的时间帧数 |
| `stride` | 8 | 滑动窗口步长 |
| `lowcut` | 0.5 | 带通滤波下限（Hz） |
| `highcut` | 40.0 | 带通滤波上限（Hz） |
| `freq_dim_mode` | 'pad' | 频域处理：'pad'(补零到528) 或 'crop'(切到512) |
| `batch_size` | 32 | 批大小 |
| `num_epochs` | 100 | 训练轮数 |
| `warmup_epochs` | 5 | 预热阶段epoch数（冻结判别器） |
| `lr_g` | 0.0002 | Generator 学习率 |
| `lr_d` | 0.0002 | Discriminator 学习率 |
| `lambda_complex` | 100.0 | 复数L1损失权重（实部+虚部） |
| `lambda_mag` | 50.0 | 幅度L1损失权重 |
| `lambda_gan` | 1.0 | GAN 损失权重（辅助） |

## 损失函数

### Generator Loss

```
Loss_G = λ_complex * Loss_Complex + λ_mag * Loss_Mag + λ_GAN * Loss_GAN
```

- `Loss_Complex`: 复数L1损失（实部+虚部的L1距离）
- `Loss_Mag`: 幅度L1损失（sqrt(real²+imag²)的L1距离）
- `Loss_GAN`: 对抗损失（欺骗 Discriminator）

**训练策略**:
1. 预热阶段（前5个epoch）：仅优化Loss_Complex和Loss_Mag，冻结判别器
2. 对抗阶段（之后）：全部损失联合优化，判别器参与训练

### Discriminator Loss

```
Loss_D = 0.5 * (Loss_D_real + Loss_D_fake)
```

- `Loss_D_real`: 判别真实样本 (raw, clean_gt)
- `Loss_D_fake`: 判别假样本 (raw, clean_pred)

## 评估指标

- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- SNR (Signal-to-Noise Ratio)

## 可视化输出

1. 训练历史曲线
   - Generator Loss
   - Discriminator Loss
   - Loss Components

2. STFT 谱图对比
   - Raw STFT
   - Clean STFT (Ground Truth)
   - Reconstructed STFT

3. 时域波形对比
4. 功率谱密度（PSD）对比

## 推理与可视化

### 快速演示

```bash
# 快速训练并推理（3 epoch 演示）
python demo_inference.py
```

输出：
- `demo_results/demo_inference_result.png`: 4面板对比图（Raw/Clean/Pred频谱 + 时域波形）
- `demo_results/demo_slices.png`: 频率和时间切片对比

### 完整推理流程

```bash
# 1. 训练模型
python train.py --num_epochs 100 --batch_size 32

# 2. 推理并可视化
python inference.py --checkpoint ./checkpoints/checkpoint_epoch_100_best.pth --num_samples 5
```

推理输出：
- 详细对比图（9个子图）：STFT频谱、误差图、频率/时间切片、指标分布
- 统计信息JSON文件：MSE, MAE, SNR等
- 重建波形NPZ文件

### 可视化类型

1. **STFT频谱对比**：Raw/Clean/Predicted
2. **误差热力图**：绝对误差分布
3. **频率切片**：不同频率的时域曲线
4. **时间切片**：不同时刻的频谱
5. **指标分布**：MSE/MAE直方图

## 注意事项

1. 本项目为基线验证阶段，目标是证明 STFT 域 raw→clean 的条件生成是可学习的
2. 不使用 artifact 数据
3. 不引入复杂模型（Diffusion/Transformer/VAE）
4. 重点关注训练稳定性和重建质量

## 作者

EEG Artifact Removal Research Team

## 更新日志

### 2025-12-16 v2

- 添加 0.5-40Hz 带通滤波
- 修改为实部+虚部双通道输入（替代单通道幅度谱）
- 更新STFT参数：n_fft=1024, hop_length=250, n_frames=64
- 添加频域维度处理（513 -> 528补零 或 512裁剪）
- 新增复数L1损失和幅度L1损失
- 实现训练预热阶段（前5个epoch冻结判别器）
- 更新模型：输入(2,F,T), 输出(2,F,T), Tanh激活

### 2025-12-16 v1

- 初始版本发布
- 实现数据处理模块（section 划分、STFT 变换）
- 实现 cGAN 模型（U-Net Generator + PatchGAN Discriminator）
- 实现训练模块（损失函数、优化器、训练循环）
- 实现评估与可视化模块
- 创建主训练和测试脚本
