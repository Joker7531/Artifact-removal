"""
主训练脚本

使用方法:
    python train.py --config config.yaml
    
或使用默认配置:
    python train.py
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

from data import prepare_data
from models import UNetGenerator, PatchGANDiscriminator
from utils import train_cgan, Visualizer


def main(args):
    """主训练流程"""
    
    # ==================== 设置设备 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ==================== 数据准备 ====================
    print("\n" + "="*60)
    print("数据准备")
    print("="*60)
    
    train_dataset, test_dataset = prepare_data(
        npz_path=args.data_path,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_frames=args.n_frames,
        stride=args.stride,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
        freq_dim_mode=args.freq_dim_mode,
        lowcut=args.lowcut,
        highcut=args.highcut
    )
    
    print(f"\n训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 查看样本形状
    sample_raw, sample_clean = train_dataset[0]
    print(f"\n样本形状:")
    print(f"  Raw: {sample_raw.shape}")
    print(f"  Clean: {sample_clean.shape}")
    
    # ==================== 模型创建 ====================
    print("\n" + "="*60)
    print("模型创建")
    print("="*60)
    
    generator = UNetGenerator(
        in_channels=2,  # 实部+虚部
        out_channels=2,
        base_filters=args.base_filters
    )
    
    discriminator = PatchGANDiscriminator(
        in_channels=4,  # raw(2) + clean/fake(2)
        base_filters=args.base_filters
    )
    
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    
    print(f"\nGenerator 参数量: {g_params:,}")
    print(f"Discriminator 参数量: {d_params:,}")
    print(f"总参数量: {g_params + d_params:,}")
    
    # ==================== 开始训练 ====================
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    print(f"\n训练配置:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate (G): {args.lr_g}")
    print(f"  Learning Rate (D): {args.lr_d}")
    print(f"  Lambda Complex: {args.lambda_complex}")
    print(f"  Lambda Magnitude: {args.lambda_mag}")
    print(f"  Lambda GAN: {args.lambda_gan}")
    print(f"  Warmup Epochs: {args.warmup_epochs}")
    
    trainer = train_cgan(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        generator=generator,
        discriminator=discriminator,
        device=device,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_complex=args.lambda_complex,
        lambda_mag=args.lambda_mag,
        lambda_gan=args.lambda_gan,
        warmup_epochs=args.warmup_epochs,
        instance_noise_std=args.instance_noise_std,
        save_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        visualize_dir=args.results_dir
    )
    
    print("\n训练完成!")
    print(f"最佳模型已保存到: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    print(f"可视化结果保存在: {args.results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cGAN Baseline Training')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, 
                        default='/root/autodl-tmp/dataset_cz_v2.npz',
                        help='NPZ 数据文件路径')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='STFT 窗口大小')
    parser.add_argument('--hop_length', type=int, default=250,
                        help='STFT 跳跃步长')
    parser.add_argument('--n_frames', type=int, default=64,
                        help='每个样本的时间帧数')
    parser.add_argument('--stride', type=int, default=4,
                        help='滑动窗口步长')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--freq_dim_mode', type=str, default='pad',
                        choices=['pad', 'crop'],
                        help='频域维度处理方式: pad(补零到528) 或 crop(切到512)')
    parser.add_argument('--lowcut', type=float, default=0.5,
                        help='带通滤波下限频率(Hz)')
    parser.add_argument('--highcut', type=float, default=40.0,
                        help='带通滤波上限频率(Hz)')
    
    # 模型参数
    parser.add_argument('--base_filters', type=int, default=64,
                        help='基础卷积核数量')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小')
    parser.add_argument('--lr_g', type=float, default=0.0002,
                        help='Generator 学习率')
    parser.add_argument('--lr_d', type=float, default=0.00005,
                        help='Discriminator 学习率（降低以平衡训练）')
    parser.add_argument('--lambda_complex', type=float, default=100.0,
                        help='复数L1损失权重(实部+虚部)')
    parser.add_argument('--lambda_mag', type=float, default=50.0,
                        help='幅度L1损失权重')
    parser.add_argument('--lambda_gan', type=float, default=1.0,
                        help='GAN损失权重')
    parser.add_argument('--warmup_epochs', type=int, default=8,
                        help='预热阶段epoch数(冻结判别器)')
    parser.add_argument('--instance_noise_std', type=float, default=0.1,
                        help='判别器输入噪声标准差（防止D过强）')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='cGAN_baseline/checkpoints',
                        help='模型保存目录')
    parser.add_argument('--results_dir', type=str,
                        default='cGAN_baseline/results',
                        help='结果保存目录')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志输出间隔')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 运行训练
    main(args)
