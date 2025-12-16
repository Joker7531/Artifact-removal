"""
测试脚本：评估训练好的 cGAN 模型

使用方法:
    python test.py --checkpoint ./checkpoints/checkpoint_epoch_100_best.pth
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from data import prepare_data
from models import UNetGenerator
from utils import SignalReconstructor, EvaluationMetrics, Visualizer


def test(args):
    """测试流程"""
    
    # ==================== 设置设备 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
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
        in_channels=1,
        out_channels=1,
        base_filters=args.base_filters
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    print(f"模型已加载: {args.checkpoint}")
    print(f"训练轮数: {checkpoint['epoch']}")
    
    # ==================== 评估 ====================
    print("\n" + "="*60)
    print("开始评估")
    print("="*60)
    
    reconstructor = SignalReconstructor(
        n_fft=args.n_fft,
        hop_length=args.hop_length
    )
    visualizer = Visualizer(save_dir=args.results_dir)
    
    all_mse = []
    all_mae = []
    all_snr = []
    
    with torch.no_grad():
        for idx, (raw, clean) in enumerate(test_loader):
            if idx >= args.num_samples:
                break
            
            raw = raw.to(device)
            clean = clean.to(device)
            
            # 生成
            fake_clean = generator(raw)
            
            # 转为 numpy
            raw_np = raw.squeeze().cpu().numpy()  # (F, T)
            clean_np = clean.squeeze().cpu().numpy()
            fake_clean_np = fake_clean.squeeze().cpu().numpy()
            
            # 反归一化
            max_val = test_dataset.max_val
            raw_np *= max_val
            clean_np *= max_val
            fake_clean_np *= max_val
            
            # 计算指标（STFT 域）
            mse = EvaluationMetrics.mse(fake_clean_np, clean_np)
            mae = EvaluationMetrics.mae(fake_clean_np, clean_np)
            
            all_mse.append(mse)
            all_mae.append(mae)
            
            # 可视化前几个样本
            if idx < args.num_visualize:
                # STFT 谱图对比
                visualizer.plot_spectrograms(
                    raw_mag=raw_np,
                    clean_mag=clean_np,
                    reconstructed_mag=fake_clean_np,
                    sample_rate=250,
                    hop_length=args.hop_length,
                    save_name=f'spectrogram_sample_{idx}.png'
                )
                
                print(f"\nSample {idx}:")
                print(f"  MSE (STFT): {mse:.6f}")
                print(f"  MAE (STFT): {mae:.6f}")
    
    # ==================== 统计结果 ====================
    print("\n" + "="*60)
    print("评估结果汇总")
    print("="*60)
    
    print(f"\nSTFT 域指标 (平均):")
    print(f"  MSE: {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
    print(f"  MAE: {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
    
    print(f"\n结果已保存到: {args.results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cGAN Baseline Testing')
    
    # 数据参数
    parser.add_argument('--data_path', type=str,
                        default='../../dataset_cz_v2.npz',
                        help='NPZ 数据文件路径')
    parser.add_argument('--n_fft', type=int, default=256)
    parser.add_argument('--hop_length', type=int, default=128)
    parser.add_argument('--n_frames', type=int, default=32)
    parser.add_argument('--stride', type=int, default=8)
    
    # 模型参数
    parser.add_argument('--base_filters', type=int, default=64)
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/checkpoint_epoch_100_best.pth',
                        help='模型检查点路径')
    
    # 测试参数
    parser.add_argument('--num_samples', type=int, default=100,
                        help='评估样本数量')
    parser.add_argument('--num_visualize', type=int, default=5,
                        help='可视化样本数量')
    parser.add_argument('--results_dir', type=str,
                        default='./results',
                        help='结果保存目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    test(args)
