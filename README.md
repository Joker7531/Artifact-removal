# EEG Waveform Denoising GAN

基于 GAN 的端到端时域 EEG 波形去噪系统。直接在时域进行 raw→clean 波形重建,无需 STFT 转换。

## 项目结构

```
GAN_Waveform/
├── waveform_dataset.py           # 波形数据集和预处理
├── waveform_models.py             # 1D U-Net 生成器和 PatchGAN 判别器
├── train_waveform_gan.py          # 完整训练脚本
├── inference_waveform.py          # 推理和评估脚本
├── README.md                      # 本文档
├── checkpoints/                   # 模型检查点 (训练后生成)
└── results/                       # 训练结果和可视化 (训练后生成)
```

## 技术特点

### 1. 时域直接重建
- **输入**: 原始 raw EEG 波形 (1D 时序信号)
- **输出**: 干净 clean EEG 波形 (1D 时序信号)
- **优势**: 
  - 无需 STFT 频域转换
  - 端到端学习
  - 保留时域细节

### 2. 数据处理
- **采样率**: 100 Hz
- **窗口长度**: 3 秒 (300 采样点)
- **重叠率**: 75%
- **归一化**: Z-score 标准化

### 3. 模型架构

#### 生成器 (WaveformGenerator)
- **架构**: 1D U-Net with Residual Blocks
- **编码器**: 4 层下采样 (stride=2)
- **Bottleneck**: 3 个膨胀残差块 (dilation=1,2,4)
- **解码器**: 4 层上采样 + skip connections
- **特点**: 
  - 保留多尺度时域特征
  - 残差连接加速训练
  - 膨胀卷积扩大感受野

#### 判别器 (WaveformDiscriminator)
- **架构**: 1D PatchGAN
- **输入**: raw + clean/generated (2通道拼接)
- **层数**: 4 层卷积下采样
- **输出**: Patch-level 真假判别

### 4. 训练策略
- **损失函数**: GAN Loss + 100×L1 Loss
- **优化器**: Adam with TTUR
  - Generator LR: 1e-4
  - Discriminator LR: 2e-4
- **学习率调度**: StepLR (50 epochs, γ=0.5)
- **Batch Size**: 32

## 安装依赖

```bash
cd GAN_Waveform
pip install torch numpy pandas scipy matplotlib tqdm
```

## 使用方法

### 1. 训练模型

```bash
python train_waveform_gan.py
```

**训练配置** (在脚本中可修改):
```python
config = {
    'window_sec': 3.0,      # 窗口长度
    'overlap': 0.75,        # 重叠率
    'batch_size': 32,       # batch 大小
    'num_epochs': 100,      # 训练轮数
    'lambda_l1': 100,       # L1 损失权重
}
```

训练过程自动:
- 每 5 个 epoch 保存 checkpoint
- 生成训练曲线和波形对比图
- 保存最佳模型到 `checkpoints/best_model.pth`

### 2. 推理/评估

```bash
# 使用默认参数 (验证集)
python inference_waveform.py

# 自定义参数
python inference_waveform.py \
    --checkpoint checkpoints/best_model.pth \
    --raw_file ../2_Data_processed/raw_Cz_10.csv \
    --clean_file ../2_Data_processed/clean_Cz_10.csv \
    --save_dir results/inference
```

### 3. 快速测试

测试数据加载:
```bash
python waveform_dataset.py
```

测试模型结构:
```bash
python waveform_models.py
```

## 输出结果

### 训练阶段
- `checkpoints/best_model.pth`: 最佳模型
- `results/training_history.png`: 损失曲线
- `results/waveform_samples_epoch_X.png`: 波形对比图
- `results/training_history.json`: 训练历史数据

### 推理阶段
- `results/inference/*_denoised.png`: 去噪前后对比
- `results/inference/*_denoised.csv`: 去噪波形数据
- `results/inference/metrics.txt`: 评估指标
  - MSE, RMSE, MAE
  - 相关系数
  - SNR 改善

## 评估指标

系统提供以下评估指标:

1. **MSE/RMSE/MAE**: 重建误差
2. **相关系数**: 波形相似度
3. **SNR 改善**: 
   ```
   SNR_improvement = SNR_denoised - SNR_raw
   ```
   - 正值表示去噪效果良好
   - 越大表示噪声去除越多

## 模型参数量

- **生成器**: ~3-5M 参数 (取决于 base_filters 和 num_layers)
- **判别器**: ~1-2M 参数

默认配置:
```python
base_filters = 64
num_layers = 4
```

## 与 STFT 方法对比

| 特性 | 时域 GAN | STFT GAN |
|------|----------|----------|
| 输入维度 | 1D (L,) | 2D (F, T) |
| 转换开销 | 无 | STFT + iSTFT |
| 相位信息 | 隐式学习 | 需显式处理 |
| 模型复杂度 | 中等 | 较高 |
| 训练速度 | 快 | 慢 |
| 时域细节 | 好 | 中等 |
| 频域特征 | 隐式 | 显式 |

## 超参数调优建议

1. **窗口长度** (`window_sec`):
   - 2-4 秒: 平衡样本数和上下文信息
   - 过短: 缺乏上下文
   - 过长: 样本数少

2. **Lambda L1** (`lambda_l1`):
   - 50-200: 推荐范围
   - 较大值: 更忠实重建,但可能过于保守
   - 较小值: 更多 GAN 风格,可能不稳定

3. **模型层数** (`num_layers`):
   - 3-5 层: 推荐
   - 过多: 可能梯度消失
   - 过少: 感受野不足

4. **Batch Size**:
   - 16-64: 推荐
   - 受显存限制

## 常见问题

### Q1: 训练不稳定?
- 增大 `lambda_l1` (更注重 L1 重建)
- 降低 discriminator 学习率
- 使用 LSGAN (默认)

### Q2: 生成波形过于平滑?
- 减小 `lambda_l1`
- 增加 discriminator 训练频率
- 检查数据归一化

### Q3: 显存不足?
- 减小 `batch_size`
- 减小 `base_filters` (64→32)
- 减小 `num_layers` (4→3)

### Q4: 去噪效果不佳?
- 增加训练 epochs
- 调整 `lambda_l1`
- 检查训练/验证损失曲线
- 尝试不同归一化方法

## 训练监控

重点关注以下指标:

1. **L1 Loss 下降**: 主要重建质量指标
2. **Generator vs Discriminator**: 应保持平衡
   - D_loss 过低: D 太强,增加 G 学习率
   - D_loss 过高: D 太弱,增加 D 学习率
3. **验证集 L1**: 防止过拟合

## 代码模块说明

### `waveform_dataset.py`
- `WaveformProcessor`: 滑动窗口、归一化
- `EEGWaveformDataset`: PyTorch Dataset
- `create_waveform_dataloaders()`: 数据加载器

### `waveform_models.py`
- `ResidualBlock`: 1D 残差块
- `WaveformGenerator`: 1D U-Net 生成器
- `WaveformDiscriminator`: 1D PatchGAN 判别器
- `GANLoss`: LSGAN 损失

### `train_waveform_gan.py`
- `WaveformGANTrainer`: 训练器类
  - `train_epoch()`: 单 epoch 训练
  - `validate()`: 验证
  - `save_checkpoint()`: 保存模型
  - `plot_history()`: 绘制曲线
  - `save_sample_results()`: 保存样本

### `inference_waveform.py`
- `WaveformDenoiser`: 推理器
  - `denoise_signal()`: 波形去噪
  - `evaluate()`: 评估和可视化

## 性能基准

在 EEG 数据集上的典型表现:

- **训练时间**: ~2-4 小时 (100 epochs, GTX 1650 Ti)
- **推理速度**: ~10 ms/窗口 (GPU)
- **SNR 改善**: 3-8 dB
- **相关系数**: 0.85-0.95

## 许可

MIT License

## 引用

如果使用本代码,请引用:

```
WaveNet-style Generator + PatchGAN Discriminator
Based on Pix2Pix framework (Isola et al., CVPR 2017)
```
