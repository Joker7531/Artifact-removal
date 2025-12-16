# cGAN Baseline for EEG Artifact Removal

基于 STFT 域的条件生成对抗网络（cGAN）用于 EEG 信号噪声抑制的基线验证系统。

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
- 输入: raw STFT magnitude (1, F, T)
- 输出: clean STFT magnitude (1, F, T)
- 4 层下采样 + 4 层上采样

### Discriminator: PatchGAN

- 条件判别器
- 输入: raw + clean/fake concatenated (2, F, T)
- 输出: Patch-level 判别结果

## 训练配置

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `n_fft` | 256 | STFT 窗口大小 |
| `hop_length` | 128 | STFT 跳跃步长 |
| `n_frames` | 32 | 每个样本的时间帧数 |
| `stride` | 8 | 滑动窗口步长 |
| `batch_size` | 32 | 批大小 |
| `num_epochs` | 100 | 训练轮数 |
| `lr_g` | 0.0002 | Generator 学习率 |
| `lr_d` | 0.0002 | Discriminator 学习率 |
| `lambda_l1` | 100.0 | L1 损失权重（主要） |
| `lambda_gan` | 1.0 | GAN 损失权重（辅助） |

## 损失函数

### Generator Loss

```
Loss_G = λ_GAN * Loss_GAN + λ_L1 * Loss_L1
```

- `Loss_GAN`: 对抗损失（欺骗 Discriminator）
- `Loss_L1`: L1 重建损失（保持频谱能量一致性）

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

## 注意事项

1. 本项目为基线验证阶段，目标是证明 STFT 域 raw→clean 的条件生成是可学习的
2. 不使用 artifact 数据
3. 不引入复杂模型（Diffusion/Transformer/VAE）
4. 重点关注训练稳定性和重建质量

## 作者

EEG Artifact Removal Research Team

## 更新日志

### 2025-12-16

- 初始版本发布
- 实现数据处理模块（section 划分、STFT 变换）
- 实现 cGAN 模型（U-Net Generator + PatchGAN Discriminator）
- 实现训练模块（损失函数、优化器、训练循环）
- 实现评估与可视化模块
- 创建主训练和测试脚本
