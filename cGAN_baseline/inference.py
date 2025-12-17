"""
推理脚本：使用训练好的模型进行预测并可视化

使用方法:
    python inference.py --checkpoint ./checkpoints/checkpoint_epoch_100_best.pth --num_samples 10
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')

from data import prepare_data
from models import UNetGenerator
from utils import SignalReconstructor, EvaluationMetrics, Visualizer


def inference_and_visualize(args):
    """推理和可视化主流程"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ==================== 加载数据 ====================
    print("\n" + "="*60)
    print("数据加载")
    print("="*60)
    
    _, test_dataset = prepare_data(
        npz_path=args.data_path,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_frames=args.n_frames,
        stride=args.stride,
        train_ratio=0.8,
        random_seed=42
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # ==================== 加载模型 ====================
    print("\n" + "="*60)
    print("模型加载")
    print("="*60)
    
    generator = UNetGenerator(
        in_channels=2,  # 实部+虚部双通道
        out_channels=2,
        base_filters=args.base_filters
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    print(f"模型已加载: {args.checkpoint}")
    print(f"训练轮数: {checkpoint['epoch']}")
    
    # ==================== 推理 ====================
    print("\n" + "="*60)
    print("推理中...")
    print("="*60)
    
    visualizer = Visualizer(save_dir=args.output_dir)
    
    all_metrics = {
        'mse': [],
        'mae': [],
        'mse_db': []
    }
    
    sample_count = 0
    
    with torch.no_grad():
        for idx, (raw, clean) in enumerate(test_loader):
            if sample_count >= args.num_samples:
                break
            
            raw = raw.to(device)
            clean = clean.to(device)
            
            # 生成预测
            fake_clean = generator(raw)
            
            # 转为 numpy (B, 2, F, T)
            raw_np = raw.squeeze(0).cpu().numpy()  # (2, F, T)
            clean_np = clean.squeeze(0).cpu().numpy()
            fake_clean_np = fake_clean.squeeze(0).cpu().numpy()
            
            # 反标准化（Z-score）
            real_mean = test_dataset.real_mean
            real_std = test_dataset.real_std
            imag_mean = test_dataset.imag_mean
            imag_std = test_dataset.imag_std
            
            raw_np[0] = raw_np[0] * real_std + real_mean
            raw_np[1] = raw_np[1] * imag_std + imag_mean
            clean_np[0] = clean_np[0] * real_std + real_mean
            clean_np[1] = clean_np[1] * imag_std + imag_mean
            fake_clean_np[0] = fake_clean_np[0] * real_std + real_mean
            fake_clean_np[1] = fake_clean_np[1] * imag_std + imag_mean
            
            # 计算幅度谱（从实部+虚部）
            raw_mag = np.sqrt(raw_np[0]**2 + raw_np[1]**2)  # (F, T)
            clean_mag = np.sqrt(clean_np[0]**2 + clean_np[1]**2)
            fake_clean_mag = np.sqrt(fake_clean_np[0]**2 + fake_clean_np[1]**2)
            
            # 计算指标
            mse = EvaluationMetrics.mse(fake_clean_mag, clean_mag)
            mae = EvaluationMetrics.mae(fake_clean_mag, clean_mag)
            mse_db = 10 * np.log10(mse + 1e-10)
            
            all_metrics['mse'].append(mse)
            all_metrics['mae'].append(mae)
            all_metrics['mse_db'].append(mse_db)
            
            # 可视化 STFT 谱图
            visualizer.plot_spectrograms(
                raw_mag=raw_mag,
                clean_mag=clean_mag,
                reconstructed_mag=fake_clean_mag,
                sample_rate=args.sample_rate,
                hop_length=args.hop_length,
                save_name=f'inference_spectrogram_{sample_count:03d}.png'
            )
            
            # 创建详细对比图
            plot_detailed_comparison(
                raw_mag, clean_mag, fake_clean_mag,
                idx=sample_count,
                output_dir=args.output_dir,
                sample_rate=args.sample_rate,
                hop_length=args.hop_length,
                mse=mse,
                mae=mae
            )
            
            sample_count += 1
            
            if sample_count % 10 == 0:
                print(f"已处理 {sample_count} 个样本...")
    
    # ==================== 统计结果 ====================
    print("\n" + "="*60)
    print("推理结果统计")
    print("="*60)
    
    print(f"\nSTFT 域指标 (基于 {sample_count} 个样本):")
    print(f"  MSE:    {np.mean(all_metrics['mse']):.6f} ± {np.std(all_metrics['mse']):.6f}")
    print(f"  MAE:    {np.mean(all_metrics['mae']):.6f} ± {np.std(all_metrics['mae']):.6f}")
    print(f"  MSE(dB): {np.mean(all_metrics['mse_db']):.2f} ± {np.std(all_metrics['mse_db']):.2f}")
    
    # 绘制指标分布图
    plot_metrics_distribution(all_metrics, args.output_dir)
    
    # 保存统计结果
    save_statistics(all_metrics, args.output_dir, sample_count)
    
    print(f"\n结果已保存到: {args.output_dir}")
    print("="*60)


def plot_detailed_comparison(raw_mag, clean_mag, recon_mag, idx, output_dir, 
                             sample_rate=250, hop_length=128, mse=0, mae=0):
    """
    创建详细的对比图（包含谱图、误差图、频谱切片）
    
    参数:
        raw_mag: Raw STFT 幅度谱 (freq_bins, time_frames)
        clean_mag: Clean STFT 幅度谱
        recon_mag: 重建 STFT 幅度谱
        idx: 样本编号
        output_dir: 输出目录
        sample_rate: 采样率
        hop_length: STFT 跳跃步长
        mse, mae: 评估指标
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    freq_bins, time_frames = raw_mag.shape
    time = np.arange(time_frames) * hop_length / sample_rate
    freq = np.linspace(0.5, 40, freq_bins)  # 0.5-40Hz范围
    
    # 转换为dB
    raw_db = 20 * np.log10(raw_mag + 1e-8)
    clean_db = 20 * np.log10(clean_mag + 1e-8)
    recon_db = 20 * np.log10(recon_mag + 1e-8)
    
    # 计算统一的颜色映射范围
    vmin = min(raw_db.min(), clean_db.min(), recon_db.min())
    vmax = max(raw_db.max(), clean_db.max(), recon_db.max())
    
    # 1. Raw STFT
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        raw_db,
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], freq[0], freq[-1]],
        cmap='viridis',
        vmin=vmin,
        vmax=vmax
    )
    ax1.set_title('Raw STFT Magnitude (dB)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Clean STFT (Ground Truth)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        clean_db,
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], freq[0], freq[-1]],
        cmap='viridis',
        vmin=vmin,
        vmax=vmax
    )
    ax2.set_title('Clean STFT (Ground Truth)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Reconstructed STFT
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(
        recon_db,
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], freq[0], freq[-1]],
        cmap='viridis',
        vmin=vmin,
        vmax=vmax
    )
    ax3.set_title('Reconstructed STFT', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    plt.colorbar(im3, ax=ax3)
    
    # 4. 误差图（绝对误差）
    ax4 = fig.add_subplot(gs[1, 0])
    error = np.abs(recon_mag - clean_mag)
    im4 = ax4.imshow(
        error,
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], freq[0], freq[-1]],
        cmap='hot'
    )
    ax4.set_title('Absolute Error', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (Hz)')
    plt.colorbar(im4, ax=ax4)
    
    # 5. 相对误差图
    ax5 = fig.add_subplot(gs[1, 1])
    relative_error = error / (clean_mag + 1e-8)
    im5 = ax5.imshow(
        np.clip(relative_error, 0, 2),  # 限制显示范围
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], freq[0], freq[-1]],
        cmap='hot'
    )
    ax5.set_title('Relative Error (clipped to [0, 2])', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Frequency (Hz)')
    plt.colorbar(im5, ax=ax5)
    
    # 6. 频谱切片对比（中间时刻）
    ax6 = fig.add_subplot(gs[1, 2])
    mid_time = time_frames // 2
    ax6.plot(freq, raw_mag[:, mid_time], label='Raw', alpha=0.7, linewidth=1.5)
    ax6.plot(freq, clean_mag[:, mid_time], label='Clean (GT)', alpha=0.7, linewidth=1.5)
    ax6.plot(freq, recon_mag[:, mid_time], label='Reconstructed', alpha=0.7, linewidth=1.5, linestyle='--')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Magnitude')
    ax6.set_title(f'Frequency Spectrum at t={time[mid_time]:.2f}s', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 时间序列对比（特定频率）
    ax7 = fig.add_subplot(gs[2, 0])
    mid_freq = freq_bins // 2
    ax7.plot(time, raw_mag[mid_freq, :], label='Raw', alpha=0.7, linewidth=1.5)
    ax7.plot(time, clean_mag[mid_freq, :], label='Clean (GT)', alpha=0.7, linewidth=1.5)
    ax7.plot(time, recon_mag[mid_freq, :], label='Reconstructed', alpha=0.7, linewidth=1.5, linestyle='--')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Magnitude')
    ax7.set_title(f'Time Series at f={freq[mid_freq]:.1f}Hz', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 误差时间序列
    ax8 = fig.add_subplot(gs[2, 1])
    error_time = np.mean(error, axis=0)  # 沿频率轴平均
    ax8.plot(time, error_time, color='red', linewidth=1.5)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Mean Absolute Error')
    ax8.set_title('Error Over Time (Freq-Averaged)', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # 9. 指标文本信息
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    metrics_text = f"""
    样本编号: {idx}
    
    STFT 形状: {raw_mag.shape}
    时间长度: {time[-1]:.2f}s
    频率范围: 0-{freq[-1]:.1f}Hz
    
    评估指标:
    ─────────────────
    MSE:  {mse:.6f}
    MAE:  {mae:.6f}
    
    误差统计:
    ─────────────────
    最大误差: {np.max(error):.4f}
    平均误差: {np.mean(error):.4f}
    误差标准差: {np.std(error):.4f}
    
    相对误差:
    ─────────────────
    平均: {np.mean(relative_error):.4f}
    中位数: {np.median(relative_error):.4f}
    """
    
    ax9.text(0.1, 0.9, metrics_text, transform=ax9.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Detailed Analysis - Sample {idx}', fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(os.path.join(output_dir, f'detailed_comparison_{idx:03d}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_distribution(metrics, output_dir):
    """绘制评估指标的分布图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # MSE 分布
    axes[0].hist(metrics['mse'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(metrics['mse']), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(metrics["mse"]):.6f}')
    axes[0].set_xlabel('MSE')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('MSE Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE 分布
    axes[1].hist(metrics['mae'], bins=30, color='seagreen', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(metrics['mae']), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(metrics["mae"]):.6f}')
    axes[1].set_xlabel('MAE')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('MAE Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # MSE (dB) 分布
    axes[2].hist(metrics['mse_db'], bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[2].axvline(np.mean(metrics['mse_db']), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(metrics["mse_db"]):.2f} dB')
    axes[2].set_xlabel('MSE (dB)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('MSE (dB) Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def save_statistics(metrics, output_dir, num_samples):
    """保存统计结果到文本文件"""
    stats_file = os.path.join(output_dir, 'inference_statistics.txt')
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("推理结果统计\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"样本数量: {num_samples}\n\n")
        
        f.write("STFT 域指标:\n")
        f.write("-"*60 + "\n")
        f.write(f"MSE:\n")
        f.write(f"  均值:   {np.mean(metrics['mse']):.8f}\n")
        f.write(f"  标准差: {np.std(metrics['mse']):.8f}\n")
        f.write(f"  最小值: {np.min(metrics['mse']):.8f}\n")
        f.write(f"  最大值: {np.max(metrics['mse']):.8f}\n")
        f.write(f"  中位数: {np.median(metrics['mse']):.8f}\n\n")
        
        f.write(f"MAE:\n")
        f.write(f"  均值:   {np.mean(metrics['mae']):.8f}\n")
        f.write(f"  标准差: {np.std(metrics['mae']):.8f}\n")
        f.write(f"  最小值: {np.min(metrics['mae']):.8f}\n")
        f.write(f"  最大值: {np.max(metrics['mae']):.8f}\n")
        f.write(f"  中位数: {np.median(metrics['mae']):.8f}\n\n")
        
        f.write(f"MSE (dB):\n")
        f.write(f"  均值:   {np.mean(metrics['mse_db']):.2f}\n")
        f.write(f"  标准差: {np.std(metrics['mse_db']):.2f}\n")
        f.write(f"  最小值: {np.min(metrics['mse_db']):.2f}\n")
        f.write(f"  最大值: {np.max(metrics['mse_db']):.2f}\n")
        f.write(f"  中位数: {np.median(metrics['mse_db']):.2f}\n\n")
        
        f.write("="*60 + "\n")
    
    print(f"\n统计结果已保存到: {stats_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cGAN Inference and Visualization')
    
    # 数据参数
    parser.add_argument('--data_path', type=str,
                        default='/root/autodl-tmp/dataset_cz_v2.npz',
                        help='NPZ 数据文件路径')
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=250)
    parser.add_argument('--n_frames', type=int, default=64)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--sample_rate', type=int, default=500,
                        help='EEG 采样率')
    
    # 模型参数
    parser.add_argument('--base_filters', type=int, default=64)
    parser.add_argument('--checkpoint', type=str,
                        default='cGAN_baseline/checkpoints/best_model.pth',
                        help='模型检查点路径')
    
    # 推理参数
    parser.add_argument('--num_samples', type=int, default=20,
                        help='推理样本数量')
    parser.add_argument('--output_dir', type=str,
                        default='cGAN_baseline/inference_results',
                        help='结果保存目录')
    
    args = parser.parse_args()
    
    inference_and_visualize(args)
