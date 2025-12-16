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
        random_seed=args.seed
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
        in_channels=1,
        out_channels=1,
        base_filters=args.base_filters
    )
    
    discriminator = PatchGANDiscriminator(
        in_channels=2,
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
    print(f"  Lambda L1: {args.lambda_l1}")
    print(f"  Lambda GAN: {args.lambda_gan}")
    
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
        lambda_l1=args.lambda_l1,
        lambda_gan=args.lambda_gan,
        save_dir=args.checkpoint_dir,
        log_interval=args.log_interval
    )
    
    # ==================== 可视化训练历史 ====================
    print("\n" + "="*60)
    print("保存训练历史")
    print("="*60)
    
    visualizer = Visualizer(save_dir=args.results_dir)
    visualizer.plot_training_history(
        history=trainer.history,
        save_name='training_history.png'
    )
    
    print(f"\n训练历史已保存到: {os.path.join(args.results_dir, 'training_history.png')}")
    print("\n训练完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cGAN Baseline Training')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, 
                        default='../dataset_cz_v2.npz',
                        help='NPZ 数据文件路径')
    parser.add_argument('--n_fft', type=int, default=256,
                        help='STFT 窗口大小')
    parser.add_argument('--hop_length', type=int, default=128,
                        help='STFT 跳跃步长')
    parser.add_argument('--n_frames', type=int, default=32,
                        help='每个样本的时间帧数')
    parser.add_argument('--stride', type=int, default=8,
                        help='滑动窗口步长')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    
    # 模型参数
    parser.add_argument('--base_filters', type=int, default=64,
                        help='基础卷积核数量')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--lr_g', type=float, default=0.0002,
                        help='Generator 学习率')
    parser.add_argument('--lr_d', type=float, default=0.0002,
                        help='Discriminator 学习率')
    parser.add_argument('--lambda_l1', type=float, default=100.0,
                        help='L1 损失权重')
    parser.add_argument('--lambda_gan', type=float, default=1.0,
                        help='GAN 损失权重')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--results_dir', type=str,
                        default='./results',
                        help='结果保存目录')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志输出间隔')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 运行训练
    main(args)
