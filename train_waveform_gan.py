"""
Training Script for Waveform GAN
完整的时域波形重建训练循环
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from waveform_models import WaveformGenerator, WaveformDiscriminator, GANLoss, weights_init
from waveform_dataset import create_waveform_dataloaders


class WaveformGANTrainer:
    """Waveform GAN 训练器"""
    
    def __init__(self,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 config: dict):
        """
        Args:
            generator: 生成器模型
            discriminator: 判别器模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 计算设备
            config: 训练配置字典
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 优化器 (TTUR + L2 正则化)
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config['g_lr'],
            betas=(config['beta1'], 0.999),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config['d_lr'],
            betas=(config['beta1'], 0.999),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler_g = optim.lr_scheduler.StepLR(
            self.optimizer_g,
            step_size=config['lr_decay_epochs'],
            gamma=config['lr_decay_gamma']
        )
        self.scheduler_d = optim.lr_scheduler.StepLR(
            self.optimizer_d,
            step_size=config['lr_decay_epochs'],
            gamma=config['lr_decay_gamma']
        )
        
        # 损失函数
        self.criterion_gan = GANLoss(gan_mode=config['gan_mode']).to(device)
        self.criterion_l1 = nn.L1Loss()
        self.lambda_l1 = config['lambda_l1']
        
        # 早停机制
        self.patience = config.get('patience', 15)
        self.min_delta = config.get('min_delta', 1e-4)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop = False
        
        # 训练历史
        self.history = {
            'train_g_loss': [],
            'train_d_loss': [],
            'train_g_gan_loss': [],
            'train_g_l1_loss': [],
            'val_g_loss': [],
            'val_l1_loss': [],
            'epoch_times': []
        }
        
        # 创建保存目录
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path(config['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_g_gan_loss = 0
        epoch_g_l1_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            raw_wave = batch['raw'].to(self.device)      # (B, 1, L)
            clean_wave = batch['clean'].to(self.device)  # (B, 1, L)
            
            # ============================================
            # 1. 训练判别器 D
            # ============================================
            self.optimizer_d.zero_grad()
            
            # 生成假波形
            fake_wave = self.generator(raw_wave)
            
            # 真实样本的损失
            pred_real = self.discriminator(raw_wave, clean_wave)
            loss_d_real = self.criterion_gan(pred_real, True)
            
            # 生成样本的损失
            pred_fake = self.discriminator(raw_wave, fake_wave.detach())
            loss_d_fake = self.criterion_gan(pred_fake, False)
            
            # 总判别器损失
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            self.optimizer_d.step()
            
            # ============================================
            # 2. 训练生成器 G
            # ============================================
            self.optimizer_g.zero_grad()
            
            # 生成器希望判别器认为生成的是真的
            pred_fake = self.discriminator(raw_wave, fake_wave)
            loss_g_gan = self.criterion_gan(pred_fake, True)
            
            # L1 损失 (波形级重建)
            loss_g_l1 = self.criterion_l1(fake_wave, clean_wave)
            
            # 总生成器损失
            loss_g = loss_g_gan + self.lambda_l1 * loss_g_l1
            loss_g.backward()
            self.optimizer_g.step()
            
            # 累积损失
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            epoch_g_gan_loss += loss_g_gan.item()
            epoch_g_l1_loss += loss_g_l1.item()
            
            # 更新进度条
            pbar.set_postfix({
                'G': f'{loss_g.item():.4f}',
                'D': f'{loss_d.item():.4f}',
                'L1': f'{loss_g_l1.item():.4f}'
            })
        
        # 计算平均损失
        num_batches = len(self.train_loader)
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_gan_loss = epoch_g_gan_loss / num_batches
        avg_g_l1_loss = epoch_g_l1_loss / num_batches
        
        return avg_g_loss, avg_d_loss, avg_g_gan_loss, avg_g_l1_loss
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.generator.eval()
        
        val_g_loss = 0
        val_l1_loss = 0
        
        for batch in self.val_loader:
            raw_wave = batch['raw'].to(self.device)
            clean_wave = batch['clean'].to(self.device)
            
            # 生成
            fake_wave = self.generator(raw_wave)
            
            # 计算损失
            pred_fake = self.discriminator(raw_wave, fake_wave)
            loss_g_gan = self.criterion_gan(pred_fake, True)
            loss_l1 = self.criterion_l1(fake_wave, clean_wave)
            loss_g = loss_g_gan + self.lambda_l1 * loss_l1
            
            val_g_loss += loss_g.item()
            val_l1_loss += loss_l1.item()
        
        # 平均损失
        num_batches = len(self.val_loader)
        avg_val_g_loss = val_g_loss / num_batches
        avg_val_l1_loss = val_l1_loss / num_batches
        
        return avg_val_g_loss, avg_val_l1_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存模型 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        # 保存最新的 checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def plot_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Generator Loss
        axes[0, 0].plot(self.history['train_g_loss'], label='Train G Loss', alpha=0.8)
        axes[0, 0].plot(self.history['val_g_loss'], label='Val G Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Generator Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Discriminator Loss
        axes[0, 1].plot(self.history['train_d_loss'], label='Train D Loss', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Generator GAN Loss
        axes[1, 0].plot(self.history['train_g_gan_loss'], label='GAN Loss', alpha=0.8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Generator GAN Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # L1 Loss
        axes[1, 1].plot(self.history['train_g_l1_loss'], label='Train L1', alpha=0.8)
        axes[1, 1].plot(self.history['val_l1_loss'], label='Val L1', alpha=0.8)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('L1 Reconstruction Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.results_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training history plot saved to {plot_path}")
    
    def save_sample_results(self, epoch):
        """保存样本生成结果 - 三条曲线在同一窗口对比"""
        self.generator.eval()
        
        with torch.no_grad():
            # 从验证集取一个 batch
            batch = next(iter(self.val_loader))
            raw_wave = batch['raw'].to(self.device)
            clean_wave = batch['clean'].to(self.device)
            
            # 生成
            fake_wave = self.generator(raw_wave)
            
            # 转到 CPU
            raw_wave = raw_wave.cpu().numpy()
            clean_wave = clean_wave.cpu().numpy()
            fake_wave = fake_wave.cpu().numpy()
            
            # 取前 4 个样本可视化
            num_samples = min(4, raw_wave.shape[0])
            
            # 改为单列布局，每个样本一个子图
            fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
            if num_samples == 1:
                axes = [axes]
            
            fs = self.config['fs']
            
            for i in range(num_samples):
                time_axis = np.arange(raw_wave.shape[2]) / fs
                
                # 在同一个窗口绘制三条曲线
                axes[i].plot(time_axis, raw_wave[i, 0], linewidth=0.8, alpha=0.7, 
                            label='Raw (Noisy)', color='blue')
                axes[i].plot(time_axis, clean_wave[i, 0], linewidth=0.8, alpha=0.8, 
                            label='Ground Truth (Clean)', color='green')
                axes[i].plot(time_axis, fake_wave[i, 0], linewidth=0.8, alpha=0.8, 
                            label='Generated (Denoised)', color='red', linestyle='--')
                
                axes[i].set_title(f'Sample {i+1}: Waveform Comparison', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('Amplitude (μV)', fontsize=10)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend(loc='upper right', fontsize=9)
                
                if i == num_samples - 1:
                    axes[i].set_xlabel('Time (s)', fontsize=10)
            
            plt.tight_layout()
            sample_path = self.results_dir / f'waveform_samples_epoch_{epoch+1}.png'
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Sample results saved to {sample_path}")
    
    def train(self, start_epoch=0):
        """完整训练流程"""
        print("\n" + "="*80)
        print("Starting Waveform GAN Training".center(80))
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Epochs: {self.config['num_epochs']}")
        print(f"G learning rate: {self.config['g_lr']}")
        print(f"D learning rate: {self.config['d_lr']}")
        print(f"Lambda L1: {self.lambda_l1}")
        print("="*80 + "\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            epoch_start_time = time.time()
            
            # 训练
            train_g_loss, train_d_loss, train_g_gan_loss, train_g_l1_loss = self.train_epoch(epoch)
            
            # 验证
            val_g_loss, val_l1_loss = self.validate()
            
            # 学习率调度
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            epoch_time = time.time() - epoch_start_time
            
            # 记录历史
            self.history['train_g_loss'].append(train_g_loss)
            self.history['train_d_loss'].append(train_d_loss)
            self.history['train_g_gan_loss'].append(train_g_gan_loss)
            self.history['train_g_l1_loss'].append(train_g_l1_loss)
            self.history['val_g_loss'].append(val_g_loss)
            self.history['val_l1_loss'].append(val_l1_loss)
            self.history['epoch_times'].append(epoch_time)
            
            # 打印结果
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} - Time: {epoch_time:.2f}s")
            print(f"  Train - G Loss: {train_g_loss:.4f}, D Loss: {train_d_loss:.4f}, "
                  f"GAN: {train_g_gan_loss:.4f}, L1: {train_g_l1_loss:.4f}")
            print(f"  Val   - G Loss: {val_g_loss:.4f}, L1: {val_l1_loss:.4f}")
            
            # 早停检查
            if val_l1_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_l1_loss
                self.patience_counter = 0
                is_best = True
                print(f"  ✓ New best model! (Val L1: {self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                is_best = False
                print(f"  No improvement. Patience: {self.patience_counter}/{self.patience}")
            
            # 检查是否需要早停
            if self.patience_counter >= self.patience:
                print(f"\n早停触发! 验证损失在 {self.patience} 个 epoch 内未改善。")
                self.early_stop = True
            
            # 定期保存
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch, is_best=is_best)
                self.plot_history()
                self.save_sample_results(epoch)
            
            # 早停退出
            if self.early_stop:
                self.save_checkpoint(epoch, is_best=False)
                self.plot_history()
                self.save_sample_results(epoch)
                break
        
        # 训练结束
        print("\n" + "="*80)
        if self.early_stop:
            print("Training Stopped Early!".center(80))
        else:
            print("Training Completed!".center(80))
        print("="*80)
        print(f"Best validation L1 loss: {self.best_val_loss:.4f}")
        print(f"Total training time: {sum(self.history['epoch_times'])/3600:.2f} hours")
        
        # 保存最终结果
        if not self.early_stop:
            self.save_checkpoint(self.config['num_epochs']-1, is_best=False)
            self.plot_history()
            self.save_sample_results(self.config['num_epochs']-1)
        
        # 保存训练配置和历史
        history_path = self.results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")


def main():
    """主训练函数"""
    
    # ==================== 配置参数 ====================
    config = {
        # 数据参数
        'data_dir': '../2_Data_processed',
        'train_indices': list(range(1, 9)),  # 01-08
        'val_indices': [9, 10],              # 09-10
        
        # 波形参数
        'fs': 100,
        'window_sec': 3.0,
        'overlap': 0.75,
        
        # 模型参数（防过拟合）
        'base_filters': 64,
        'num_layers': 4,
        'dropout_rate': 0.3,  # Dropout 比率
        
        # 训练参数
        'batch_size': 32,
        'num_epochs': 100,
        'g_lr': 1e-4,
        'd_lr': 2e-4,
        'beta1': 0.5,
        'lambda_l1': 100,
        'gan_mode': 'lsgan',
        'weight_decay': 1e-5,  # L2 正则化（权重衰减）
        
        # 早停参数
        'patience': 15,      # 早停耐心值
        'min_delta': 1e-4,   # 最小改善阈值
        
        # 学习率衰减
        'lr_decay_epochs': 50,
        'lr_decay_gamma': 0.5,
        
        # 数据增强
        'use_augmentation': True,
        
        # 保存参数
        'save_freq': 5,
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results',
        
        # 其他
        'num_workers': 0,  # Windows 上建议设为 0
        'seed': 42
    }
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # ==================== 数据加载 ====================
    print("\nLoading waveform data...")
    train_loader, val_loader = create_waveform_dataloaders(
        data_dir=config['data_dir'],
        train_indices=config['train_indices'],
        val_indices=config['val_indices'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_augmentation=config['use_augmentation'],
        fs=config['fs'],
        window_sec=config['window_sec'],
        overlap=config['overlap']
    )
    
    # 获取数据维度
    sample_batch = next(iter(train_loader))
    print(f"Waveform shape: {sample_batch['raw'].shape}")
    
    # ==================== 创建模型 ====================
    print("\nCreating models with regularization...")
    generator = WaveformGenerator(
        in_channels=1, 
        out_channels=1,
        base_filters=config['base_filters'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate']
    )
    discriminator = WaveformDiscriminator(
        in_channels=2,
        base_filters=config['base_filters'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate']
    )
    
    # 初始化权重
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # ==================== 训练 ====================
    trainer = WaveformGANTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
