"""
演示推理功能（使用 quick_test 生成的临时模型）

功能：
1. 先运行快速训练生成模型
2. 使用该模型进行推理
3. 生成可视化结果
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import prepare_data
from models import UNetGenerator, PatchGANDiscriminator
from utils import cGANTrainer
from torch.utils.data import DataLoader, Subset


def quick_train_and_inference():
    """快速训练并推理演示"""
    
    print("="*60)
    print("推理功能演示（包含快速训练）")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # ==================== 1. 快速训练 ====================
    print("\n[1/3] 快速训练（3 epoch）...")
    
    # 加载数据
    train_dataset, test_dataset = prepare_data(
        npz_path='../../dataset_cz_v2.npz',
        n_fft=256,
        hop_length=128,
        n_frames=32,
        stride=16,
        train_ratio=0.8,
        random_seed=42
    )
    
    # 使用子集加速
    train_subset = Subset(train_dataset, list(range(min(500, len(train_dataset)))))
    test_subset = Subset(test_dataset, list(range(min(100, len(test_dataset)))))
    
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)
    
    # 创建模型
    generator = UNetGenerator(in_channels=1, out_channels=1, base_filters=32).to(device)
    discriminator = PatchGANDiscriminator(in_channels=2, base_filters=32).to(device)
    
    # 训练
    trainer = cGANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        lr_g=0.0002,
        lr_d=0.0002,
        lambda_l1=100.0,
        lambda_gan=1.0
    )
    
    for epoch in range(1, 4):
        train_losses = trainer.train_epoch(train_loader, epoch)
        val_losses = trainer.validate(test_loader)
        print(f"  Epoch {epoch}: Train L1={train_losses['g_l1_loss']:.6f}, Val L1={val_losses['g_l1_loss']:.6f}")
    
    # ==================== 2. 推理 ====================
    print("\n[2/3] 执行推理...")
    
    generator.eval()
    
    # 获取测试样本
    raw, clean = test_dataset[0]
    raw = raw.unsqueeze(0).to(device)
    clean = clean.unsqueeze(0).to(device)
    
    with torch.no_grad():
        fake_clean = generator(raw)
    
    # 转为 numpy
    raw_np = raw.squeeze().cpu().numpy()
    clean_np = clean.squeeze().cpu().numpy()
    fake_clean_np = fake_clean.squeeze().cpu().numpy()
    
    # 反归一化
    max_val = test_dataset.max_val
    raw_np *= max_val
    clean_np *= max_val
    fake_clean_np *= max_val
    
    # 计算指标
    mse = np.mean((fake_clean_np - clean_np) ** 2)
    mae = np.mean(np.abs(fake_clean_np - clean_np))
    
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # ==================== 3. 可视化 ====================
    print("\n[3/3] 生成可视化...")
    
    output_dir = './demo_results'
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    freq_bins, time_frames = raw_np.shape
    sample_rate = 250
    hop_length = 128
    time = np.arange(time_frames) * hop_length / sample_rate
    freq = np.linspace(0, sample_rate / 2, freq_bins)
    
    # Raw STFT
    im0 = axes[0, 0].imshow(
        20 * np.log10(raw_np + 1e-8),
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], freq[0], freq[-1]],
        cmap='viridis'
    )
    axes[0, 0].set_title('Raw STFT Magnitude (dB)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Clean STFT
    im1 = axes[0, 1].imshow(
        20 * np.log10(clean_np + 1e-8),
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], freq[0], freq[-1]],
        cmap='viridis'
    )
    axes[0, 1].set_title('Clean STFT (Ground Truth)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Reconstructed STFT
    im2 = axes[1, 0].imshow(
        20 * np.log10(fake_clean_np + 1e-8),
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], freq[0], freq[-1]],
        cmap='viridis'
    )
    axes[1, 0].set_title('Reconstructed STFT', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # 误差图
    error = np.abs(fake_clean_np - clean_np)
    im3 = axes[1, 1].imshow(
        error,
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], freq[0], freq[-1]],
        cmap='hot'
    )
    axes[1, 1].set_title(f'Absolute Error (MSE={mse:.6f}, MAE={mae:.6f})', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'demo_inference_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n可视化结果已保存: {save_path}")
    
    # 频谱切片对比
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 频率切片（中间时刻）
    mid_time = time_frames // 2
    axes[0].plot(freq, raw_np[:, mid_time], label='Raw', alpha=0.7, linewidth=2)
    axes[0].plot(freq, clean_np[:, mid_time], label='Clean (GT)', alpha=0.7, linewidth=2)
    axes[0].plot(freq, fake_clean_np[:, mid_time], label='Reconstructed', 
                 alpha=0.7, linewidth=2, linestyle='--')
    axes[0].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[0].set_ylabel('Magnitude', fontsize=12)
    axes[0].set_title(f'Frequency Spectrum at t={time[mid_time]:.2f}s', 
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 时间切片（中间频率）
    mid_freq = freq_bins // 2
    axes[1].plot(time, raw_np[mid_freq, :], label='Raw', alpha=0.7, linewidth=2)
    axes[1].plot(time, clean_np[mid_freq, :], label='Clean (GT)', alpha=0.7, linewidth=2)
    axes[1].plot(time, fake_clean_np[mid_freq, :], label='Reconstructed', 
                 alpha=0.7, linewidth=2, linestyle='--')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Magnitude', fontsize=12)
    axes[1].set_title(f'Time Series at f={freq[mid_freq]:.1f}Hz', 
                      fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path2 = os.path.join(output_dir, 'demo_slices.png')
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"切片对比已保存: {save_path2}")
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print(f"\n生成的文件:")
    print(f"  1. {save_path}")
    print(f"  2. {save_path2}")
    print("\n说明：")
    print("  - 这是使用快速训练（3 epoch）的演示")
    print("  - 完整训练会获得更好的重建质量")
    print("  - 运行完整推理: python inference.py --checkpoint <your_checkpoint>")
    print("="*60)


if __name__ == "__main__":
    quick_train_and_inference()
