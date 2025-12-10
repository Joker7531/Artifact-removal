# U-Net Denoising AutoEncoder for EEG Signal Processing

基于 **U-Net** 的 STFT 域降噪自编码器，适用于 EEG / RF / Audio 等一维信号降噪任务。

## 📂 项目结构

```
Unet_denosing/
├── data_preparation.py      # 数据加载、切分、混合与 STFT 预处理
├── unet_model.py             # U-Net 模型定义与损失函数
├── train_unet.py             # 训练脚本（支持断点续训、早停）
├── inference_unet.py         # 推理与可视化脚本
├── data_augmentation.py      # 数据增强策略
├── README.md                 # 本文件
├── checkpoints_unet/         # 模型检查点（训练后生成）
├── logs_unet/                # TensorBoard 日志（训练后生成）
├── dataset_mixed/            # 混合后的数据集（运行后生成）
└── results_unet/             # 推理结果（推理后生成）
```

---

## 🚀 快速开始

### 1️⃣ 环境安装

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scipy scikit-learn tqdm tensorboard
```

### 2️⃣ 数据准备与训练

```bash
# 方式 1: 使用默认参数训练（推荐新手）
python train_unet.py

# 方式 2: 自定义参数
python train_unet.py \
    --data_dir 2_Data_processed \
    --num_files 10 \
    --segment_length 2048 \
    --overlap_ratio 0.5 \
    --n_fft 256 \
    --hop_length 64 \
    --mode magnitude \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-3 \
    --base_channels 64 \
    --depth 4
```

**训练参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `2_Data_processed` | 数据目录 |
| `--num_files` | `10` | 数据文件数量（10 组） |
| `--segment_length` | `2048` | 信号片段长度 |
| `--overlap_ratio` | `0.5` | 片段重叠比例 |
| `--test_ratio` | `0.2` | 测试集比例 |
| `--n_fft` | `256` | FFT 点数 |
| `--hop_length` | `64` | 帧移 |
| `--mode` | `magnitude` | STFT 模式（`magnitude` 或 `complex`） |
| `--batch_size` | `16` | Batch size |
| `--epochs` | `100` | 训练 epoch 数 |
| `--lr` | `1e-3` | 学习率 |
| `--base_channels` | `64` | U-Net 基础通道数 |
| `--depth` | `4` | U-Net 深度 |

### 3️⃣ 推理与可视化

```bash
# 使用最佳模型进行推理
python inference_unet.py \
    --model_path checkpoints_unet/best.pth \
    --visualize \
    --save_results \
    --num_visualize 10
```

**推理参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `checkpoints_unet/best.pth` | 模型权重路径 |
| `--visualize` | `False` | 是否可视化 |
| `--save_results` | `False` | 是否保存降噪结果 |
| `--num_samples` | `0` | 推理样本数（0 表示全部） |
| `--num_visualize` | `5` | 可视化样本数 |
| `--sample_rate` | `250` | 采样率（用于可视化） |

---

## 📊 模型架构

### U-Net 结构

```
输入: (batch, channels, freq, time)
  └─ channels=1 (magnitude 模式) 或 2 (complex 模式)
  └─ freq = n_fft // 2 + 1 (例如 256//2+1=129)
  └─ time = (segment_length - n_fft) // hop_length + 1

编码器 (Encoder):
  DoubleConv(1/2 → 64)
  Down(64 → 128)
  Down(128 → 256)
  Down(256 → 512)
  Down(512 → 1024)

解码器 (Decoder) + 跳连:
  Up(1024 → 512) ← skip from Down(256→512)
  Up(512 → 256)  ← skip from Down(128→256)
  Up(256 → 128)  ← skip from Down(64→128)
  Up(128 → 64)   ← skip from DoubleConv

输出: Conv(64 → 1/2)
```

### 损失函数

- **L1 Loss**: 幅度谱差异
- **L2 Loss (MSE)**: 能量误差
- **多尺度 STFT Loss**（可选）: 多分辨率频谱损失

组合损失：`Total Loss = λ1 * L1 + λ2 * L2`

---

## 🔧 核心功能

### 1. 数据处理流程

```python
from data_preparation import EEGDataPreparation, create_dataloaders

# 初始化
prep = EEGDataPreparation(
    data_dir="2_Data_processed",
    num_files=10,
    test_ratio=0.2,
    random_seed=42
)

# 加载全部数据
clean_signals, noisy_signals = prep.load_all_data()

# 切分与混合
clean_segments, noisy_segments = prep.segment_signals(
    clean_signals, noisy_signals,
    segment_length=2048,
    overlap_ratio=0.5
)

# 划分训练/测试集
clean_train, clean_test, noisy_train, noisy_test = prep.split_train_test(
    clean_segments, noisy_segments
)

# 保存数据集
prep.save_dataset(clean_train, clean_test, noisy_train, noisy_test)
```

### 2. STFT Dataset

```python
from data_preparation import STFTDataset

# 创建 STFT 数据集
dataset = STFTDataset(
    clean_data=clean_train,
    noisy_data=noisy_train,
    n_fft=256,
    hop_length=64,
    mode='magnitude',  # 或 'complex'
    normalize=True
)
```

### 3. 模型构建

```python
from unet_model import build_unet

# 构建模型
model = build_unet(
    mode='magnitude',
    base_channels=64,
    depth=4,
    device='cuda'
)

# 查看模型信息
model_info = model.get_model_size()
print(f"参数量: {model_info['total_params']:,}")
print(f"模型大小: {model_info['size_mb']:.2f} MB")
```

### 4. 推理

```python
from inference_unet import UNetInference

# 创建推理器
inferencer = UNetInference(
    model_path='checkpoints_unet/best.pth',
    device='cuda'
)

# 单个信号降噪
denoised_signal = inferencer.denoise_signal(noisy_signal)

# 批量降噪
denoised_signals = inferencer.denoise_batch(noisy_signals)
```

---

## 📈 训练建议

### 超参数推荐

| 数据量 | Batch Size | Learning Rate | Epochs | 早停耐心 |
|--------|-----------|---------------|--------|---------|
| < 1000 样本 | 8-16 | 5e-4 | 50-100 | 10 |
| 1000-5000 | 16-32 | 1e-3 | 100-200 | 15 |
| > 5000 | 32-64 | 1e-3 | 200-300 | 20 |

### STFT 参数推荐

| 采样率 | n_fft | hop_length | 时间分辨率 | 频率分辨率 |
|--------|-------|-----------|-----------|-----------|
| 250 Hz | 128 | 32 | 128 ms | ~2 Hz |
| 250 Hz | 256 | 64 | 256 ms | ~1 Hz |
| 250 Hz | 512 | 128 | 512 ms | ~0.5 Hz |

**选择建议：**
- EEG 信号（低频为主）：`n_fft=256, hop_length=64`
- Audio 信号（宽频）：`n_fft=512, hop_length=128`
- 快速变化信号：减小 `hop_length` 提高时间分辨率

### 学习率调度

```bash
# ReduceLROnPlateau（推荐）
python train_unet.py --scheduler plateau

# StepLR
python train_unet.py --scheduler step

# CosineAnnealing
python train_unet.py --scheduler cosine
```

---

## 🎯 数据增强策略

当数据量不足时，使用 `data_augmentation.py` 中的增强方法：

```python
from data_augmentation import DataAugmentation

aug = DataAugmentation()

# 1. 添加噪声
noisy_aug = aug.add_gaussian_noise(signal, noise_level=0.1)

# 2. 时间平移
shifted = aug.time_shift(signal, max_shift=100)

# 3. 幅度缩放
scaled = aug.amplitude_scale(signal, scale_range=(0.8, 1.2))

# 4. 时间遮罩
masked = aug.time_mask(signal, num_masks=2, mask_len=50)

# 5. SpecAugment（在 STFT 上）
aug_stft = aug.spec_augment(stft, freq_mask_param=10, time_mask_param=10)

# 6. Mixup
mixed = aug.mixup(signal1, signal2, alpha=0.5)
```

**增强策略组合建议：**
- 训练初期：仅使用轻微噪声 + 幅度缩放
- 训练中期：添加时间平移 + 时间遮罩
- 数据严重不足：使用 Mixup + SpecAugment

---

## 📊 评估指标

推理脚本自动计算以下指标：

1. **MSE (Mean Squared Error)**: 均方误差
2. **MAE (Mean Absolute Error)**: 平均绝对误差
3. **SNR (Signal-to-Noise Ratio)**: 信噪比改善
   - SNR Before: 原始带噪信号的 SNR
   - SNR After: 降噪后信号的 SNR
   - SNR Improvement: 改善量（dB）

---

## 🔍 TensorBoard 监控

训练过程中实时监控：

```bash
tensorboard --logdir logs_unet
```

可视化内容：
- 训练/验证损失曲线
- L1/L2 损失分量
- 学习率变化
- 每个 epoch 的指标

---

## 📌 常见问题

### Q1: 内存不足怎么办？

**解决方案：**
1. 减小 `batch_size`（例如从 16 降到 8）
2. 减小 `n_fft`（例如从 512 降到 256）
3. 减小 `segment_length`（例如从 2048 降到 1024）
4. 减小 U-Net 深度或通道数

```bash
python train_unet.py --batch_size 8 --base_channels 32 --depth 3
```

### Q2: 训练速度慢？

**优化方法：**
1. 使用 GPU：确保 `torch.cuda.is_available()` 返回 `True`
2. 增加 `num_workers`（但 Windows 建议设为 0）
3. 减小数据集（使用 `--use_prepared_data` 跳过数据准备）
4. 使用混合精度训练（需手动添加 AMP 支持）

### Q3: 降噪效果不好？

**调优建议：**
1. **检查 STFT 参数**：确保 `n_fft` 和 `hop_length` 适合你的信号特性
2. **增加模型容量**：`--base_channels 128 --depth 5`
3. **调整损失权重**：`--l1_weight 1.0 --l2_weight 1.0`
4. **使用 complex 模式**：`--mode complex`（保留相位信息）
5. **增加训练数据**：使用数据增强或收集更多数据

### Q4: 如何在新数据上测试？

```python
from inference_unet import UNetInference
import numpy as np

# 加载模型
inferencer = UNetInference('checkpoints_unet/best.pth')

# 加载你的信号（一维 numpy 数组）
your_signal = np.load('your_signal.npy')

# 降噪
denoised = inferencer.denoise_signal(your_signal)

# 保存
np.save('denoised_output.npy', denoised)
```

---

## 📜 许可证

MIT License

---

## 🙏 致谢

- U-Net 架构: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- SpecAugment: [Park et al., 2019](https://arxiv.org/abs/1904.08779)

---

**作者**: GitHub Copilot  
**日期**: 2025-12-10  
**版本**: 1.0.0
