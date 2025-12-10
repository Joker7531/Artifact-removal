"""
U-Net Denoising AutoEncoder 训练脚本
支持断点续训、学习率调度、早停等功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from data_preparation import EEGDataPreparation, create_dataloaders
from unet_model import build_unet, CombinedLoss


class Trainer:
    """
    U-Net 训练器
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints_unet',
        log_dir: str = 'logs_unet'
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # 创建目录
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """
        训练一个 epoch
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_l2 = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (noisy, clean) in enumerate(pbar):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            pred = self.model(noisy)
            
            # 计算损失
            loss, loss_dict = self.criterion(pred, clean)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            epoch_loss += loss_dict['total']
            epoch_l1 += loss_dict['l1']
            epoch_l2 += loss_dict['l2']
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'l1': f"{loss_dict['l1']:.4f}",
                'l2': f"{loss_dict['l2']:.4f}"
            })
            
            # TensorBoard 记录
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss_dict['total'], global_step)
            self.writer.add_scalar('Train/L1', loss_dict['l1'], global_step)
            self.writer.add_scalar('Train/L2', loss_dict['l2'], global_step)
        
        # 平均损失
        avg_loss = epoch_loss / len(self.train_loader)
        avg_l1 = epoch_l1 / len(self.train_loader)
        avg_l2 = epoch_l2 / len(self.train_loader)
        
        return avg_loss, avg_l1, avg_l2
    
    @torch.no_grad()
    def validate(self):
        """
        验证
        """
        self.model.eval()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_l2 = 0.0
        
        for noisy, clean in tqdm(self.test_loader, desc="Validation"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # 前向传播
            pred = self.model(noisy)
            
            # 计算损失
            loss, loss_dict = self.criterion(pred, clean)
            
            epoch_loss += loss_dict['total']
            epoch_l1 += loss_dict['l1']
            epoch_l2 += loss_dict['l2']
        
        avg_loss = epoch_loss / len(self.test_loader)
        avg_l1 = epoch_l1 / len(self.test_loader)
        avg_l2 = epoch_l2 / len(self.test_loader)
        
        return avg_loss, avg_l1, avg_l2
    
    def save_checkpoint(self, is_best: bool = False):
        """
        保存检查点
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新模型
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 最佳模型已保存: {best_path}")
        
        # 定期保存 epoch 模型
        if self.current_epoch % 10 == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{self.current_epoch}.pth'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✅ 检查点已加载: epoch {self.current_epoch}, best_loss {self.best_loss:.4f}")
    
    def train(self, num_epochs: int, early_stop_patience: int = 10):
        """
        完整训练流程
        """
        print(f"\n🚀 开始训练 U-Net Denoising AutoEncoder")
        print(f"   总 Epoch: {num_epochs}")
        print(f"   早停耐心: {early_stop_patience}")
        print(f"   设备: {self.device}")
        print(f"   学习率: {self.optimizer.param_groups[0]['lr']}")
        
        patience_counter = 0
        
        for epoch in range(self.current_epoch + 1, num_epochs + 1):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_l1, train_l2 = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_l1, val_l2 = self.validate()
            self.val_losses.append(val_loss)
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # TensorBoard 记录
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印信息
            print(f"\n📊 Epoch {epoch}/{num_epochs}")
            print(f"   Train Loss: {train_loss:.4f} (L1: {train_l1:.4f}, L2: {train_l2:.4f})")
            print(f"   Val Loss:   {val_loss:.4f} (L1: {val_l1:.4f}, L2: {val_l2:.4f})")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                patience_counter = 0
                print(f"   ⭐ 新最佳模型! Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
            
            self.save_checkpoint(is_best)
            
            # 早停
            if patience_counter >= early_stop_patience:
                print(f"\n⏹️  早停触发! {early_stop_patience} 个 epoch 无改善")
                break
        
        self.writer.close()
        print(f"\n✅ 训练完成! 最佳 Val Loss: {self.best_loss:.4f}")


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}")
    
    # ====== 1. 数据准备 ======
    print("\n" + "="*50)
    print("1️⃣  数据准备")
    print("="*50)
    
    if args.use_prepared_data and os.path.exists('dataset_mixed/train/clean/data.npy'):
        # 加载已准备好的数据
        print("📂 加载已准备好的数据...")
        clean_train = np.load('dataset_mixed/train/clean/data.npy')
        clean_test = np.load('dataset_mixed/test/clean/data.npy')
        noisy_train = np.load('dataset_mixed/train/noisy/data.npy')
        noisy_test = np.load('dataset_mixed/test/noisy/data.npy')
    else:
        # 重新准备数据
        prep = EEGDataPreparation(
            data_dir=args.data_dir,
            num_files=args.num_files,
            test_ratio=args.test_ratio,
            random_seed=args.seed
        )
        
        clean_signals, noisy_signals = prep.load_all_data()
        clean_segments, noisy_segments = prep.segment_signals(
            clean_signals, noisy_signals,
            segment_length=args.segment_length,
            overlap_ratio=args.overlap_ratio
        )
        clean_train, clean_test, noisy_train, noisy_test = prep.split_train_test(
            clean_segments, noisy_segments
        )
        prep.save_dataset(clean_train, clean_test, noisy_train, noisy_test)
    
    # 创建 DataLoader
    train_loader, test_loader = create_dataloaders(
        clean_train, clean_test, noisy_train, noisy_test,
        batch_size=args.batch_size,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        mode=args.mode,
        num_workers=args.num_workers
    )
    
    # ====== 2. 构建模型 ======
    print("\n" + "="*50)
    print("2️⃣  构建模型")
    print("="*50)
    
    model = build_unet(
        mode=args.mode,
        base_channels=args.base_channels,
        depth=args.depth,
        device=device
    )
    
    # ====== 3. 损失函数与优化器 ======
    print("\n" + "="*50)
    print("3️⃣  损失函数与优化器")
    print("="*50)
    
    criterion = CombinedLoss(
        l1_weight=args.l1_weight,
        l2_weight=args.l2_weight,
        use_multiscale=args.use_multiscale
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    
    print(f"   损失函数: L1({args.l1_weight}) + L2({args.l2_weight})")
    print(f"   优化器: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"   调度器: {args.scheduler}")
    
    # ====== 4. 训练器 ======
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # 加载检查点（如果有）
    if args.resume:
        checkpoint_path = Path(args.checkpoint_dir) / 'latest.pth'
        if checkpoint_path.exists():
            trainer.load_checkpoint(str(checkpoint_path))
        else:
            print(f"⚠️  未找到检查点: {checkpoint_path}")
    
    # ====== 5. 训练 ======
    print("\n" + "="*50)
    print("4️⃣  开始训练")
    print("="*50)
    
    trainer.train(num_epochs=args.epochs, early_stop_patience=args.patience)
    
    # 保存训练配置
    config = vars(args)
    config['best_loss'] = trainer.best_loss
    config['trained_epochs'] = trainer.current_epoch
    
    config_path = Path(args.checkpoint_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n💾 训练配置已保存: {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='U-Net Denoising AutoEncoder Training')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='2_Data_processed', help='数据目录')
    parser.add_argument('--num_files', type=int, default=10, help='数据文件数量')
    parser.add_argument('--segment_length', type=int, default=2048, help='信号片段长度')
    parser.add_argument('--overlap_ratio', type=float, default=0.5, help='片段重叠比例')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--use_prepared_data', action='store_true', help='使用已准备好的数据')
    
    # STFT 参数
    parser.add_argument('--n_fft', type=int, default=256, help='FFT 点数')
    parser.add_argument('--hop_length', type=int, default=64, help='帧移')
    parser.add_argument('--mode', type=str, default='magnitude', choices=['magnitude', 'complex'], 
                        help='STFT 模式')
    
    # 模型参数
    parser.add_argument('--base_channels', type=int, default=64, help='基础通道数')
    parser.add_argument('--depth', type=int, default=4, help='网络深度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='训练 epoch 数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--scheduler', type=str, default='plateau', 
                        choices=['step', 'plateau', 'cosine', 'none'], help='学习率调度器')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心')
    
    # 损失函数参数
    parser.add_argument('--l1_weight', type=float, default=1.0, help='L1 损失权重')
    parser.add_argument('--l2_weight', type=float, default=0.5, help='L2 损失权重')
    parser.add_argument('--use_multiscale', action='store_true', help='使用多尺度损失')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_unet', help='检查点目录')
    parser.add_argument('--log_dir', type=str, default='logs_unet', help='日志目录')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    main(args)
