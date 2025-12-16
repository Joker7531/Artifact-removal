"""
训练模块：cGAN 训练循环、损失函数、优化器配置

核心功能：
1. Generator 损失：复数L1损失 + 幅度L1损失 + GAN 对抗损失
2. Discriminator 损失：真假判别损失
3. 训练循环：交替训练 D 和 G，支持预热阶段
4. 模型保存与恢复
5. 训练日志记录
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple


class GANLoss(nn.Module):
    """GAN 损失函数（支持 LSGAN 和标准 GAN）"""
    
    def __init__(self, gan_mode='lsgan'):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented')
    
    def forward(self, prediction, target_is_real):
        """
        参数:
            prediction: Discriminator 输出
            target_is_real: 是否为真实样本
        """
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        
        loss = self.loss(prediction, target)
        return loss


class cGANTrainer:
    """
    cGAN 训练器（支持预热阶段和复数损失）
    """
    
    def __init__(
        self,
        generator,
        discriminator,
        device,
        lr_g=0.0002,
        lr_d=0.0002,
        beta1=0.5,
        beta2=0.999,
        lambda_complex=100.0,  # 复数 L1 损失权重（主要）
        lambda_mag=50.0,       # 幅度 L1 损失权重
        lambda_gan=1.0,        # GAN 损失权重（辅助）
        gan_mode='lsgan',
        warmup_epochs=5        # 预热阶段epoch数（冻结判别器）
    ):
        """
        参数:
            generator: Generator 模型
            discriminator: Discriminator 模型
            device: 训练设备
            lr_g: Generator 学习率
            lr_d: Discriminator 学习率
            beta1, beta2: Adam 优化器参数
            lambda_complex: 复数 L1 损失权重（实部+虚部）
            lambda_mag: 幅度 L1 损失权重
            lambda_gan: GAN 对抗损失权重
            gan_mode: GAN 损失类型
            warmup_epochs: 预热阶段epoch数
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        self.lambda_complex = lambda_complex
        self.lambda_mag = lambda_mag
        self.lambda_gan = lambda_gan
        self.warmup_epochs = warmup_epochs
        
        # 损失函数
        self.criterion_gan = GANLoss(gan_mode).to(device)
        self.criterion_l1 = nn.L1Loss()
        
        # 优化器
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=lr_g,
            betas=(beta1, beta2)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, beta2)
        )
        
        # 训练历史
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'g_complex_loss': [],
            'g_mag_loss': [],
            'g_gan_loss': [],
            'd_real_loss': [],
            'd_fake_loss': []
        }
    
    def compute_magnitude(self, complex_stft):
        """
        计算复数STFT的幅度
        
        参数:
            complex_stft: (B, 2, F, T) - 通道0: 实部, 通道1: 虚部
            
        返回:
            magnitude: (B, 1, F, T)
        """
        real = complex_stft[:, 0:1, :, :]  # (B, 1, F, T)
        imag = complex_stft[:, 1:2, :, :]
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        return magnitude
    
    def train_step(self, raw, clean, epoch) -> Dict[str, float]:
        """
        单步训练
        
        参数:
            raw: (B, 2, F, T) - 实部+虚部
            clean: (B, 2, F, T)
            epoch: 当前epoch（用于判断是否在预热阶段）
            
        返回:
            losses: 各项损失值的字典
        """
        raw = raw.to(self.device)
        clean = clean.to(self.device)
        
        # ==================== 训练 Discriminator ====================
        # 预热阶段跳过判别器训练
        if epoch > self.warmup_epochs:
            self.optimizer_d.zero_grad()
            
            # 生成假样本
            fake_clean = self.generator(raw)
            
            # 判别真实样本
            pred_real = self.discriminator(raw, clean)
            loss_d_real = self.criterion_gan(pred_real, True)
            
            # 判别假样本
            pred_fake = self.discriminator(raw, fake_clean.detach())
            loss_d_fake = self.criterion_gan(pred_fake, False)
            
            # 总 D 损失
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            self.optimizer_d.step()
        else:
            # 预热阶段：仅计算但不更新判别器
            with torch.no_grad():
                fake_clean = self.generator(raw)
                pred_real = self.discriminator(raw, clean)
                pred_fake = self.discriminator(raw, fake_clean)
                loss_d_real = self.criterion_gan(pred_real, True)
                loss_d_fake = self.criterion_gan(pred_fake, False)
                loss_d = (loss_d_real + loss_d_fake) * 0.5
        
        # ==================== 训练 Generator ====================
        self.optimizer_g.zero_grad()
        
        fake_clean = self.generator(raw)
        
        # 1. 复数 L1 损失（实部+虚部）
        loss_g_complex = self.criterion_l1(fake_clean, clean)
        
        # 2. 幅度 L1 损失
        fake_mag = self.compute_magnitude(fake_clean)
        clean_mag = self.compute_magnitude(clean)
        loss_g_mag = self.criterion_l1(fake_mag, clean_mag)
        
        # 3. GAN 对抗损失（预热阶段lambda_gan=0）
        if epoch > self.warmup_epochs:
            pred_fake = self.discriminator(raw, fake_clean)
            loss_g_gan = self.criterion_gan(pred_fake, True)
            gan_weight = self.lambda_gan
        else:
            loss_g_gan = torch.tensor(0.0).to(self.device)
            gan_weight = 0.0
        
        # 总 G 损失
        loss_g = (self.lambda_complex * loss_g_complex + 
                  self.lambda_mag * loss_g_mag + 
                  gan_weight * loss_g_gan)
        loss_g.backward()
        self.optimizer_g.step()
        
        # 记录损失
        losses = {
            'g_loss': loss_g.item(),
            'd_loss': loss_d.item(),
            'g_complex_loss': loss_g_complex.item(),
            'g_mag_loss': loss_g_mag.item(),
            'g_gan_loss': loss_g_gan.item() if isinstance(loss_g_gan, torch.Tensor) else 0.0,
            'd_real_loss': loss_d_real.item(),
            'd_fake_loss': loss_d_fake.item()
        }
        
        return losses
    
    def train_epoch(self, dataloader, epoch) -> Dict[str, float]:
        """
        训练一个 epoch
        
        参数:
            dataloader: 数据加载器
            epoch: 当前 epoch 编号
            
        返回:
            avg_losses: 平均损失
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {key: [] for key in ['g_loss', 'd_loss', 'g_complex_loss', 'g_mag_loss', 'g_gan_loss', 'd_real_loss', 'd_fake_loss']}
        
        # 判断是否在预热阶段
        stage = "Warmup" if epoch <= self.warmup_epochs else "Adversarial"
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [{stage}]')
        
        for raw, clean in pbar:
            losses = self.train_step(raw, clean, epoch)
            
            for key, value in losses.items():
                epoch_losses[key].append(value)
            
            # 更新进度条
            pbar.set_postfix({
                'G': f"{losses['g_loss']:.4f}",
                'D': f"{losses['d_loss']:.4f}",
                'Cplx': f"{losses['g_complex_loss']:.4f}",
                'Mag': f"{losses['g_mag_loss']:.4f}"
            })
        
        # 计算平均损失
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        # 记录历史
        for key, value in avg_losses.items():
            self.history[key].append(value)
        
        return avg_losses
    
    def validate(self, dataloader) -> Dict[str, float]:
        """
        验证模式评估
        
        参数:
            dataloader: 验证数据加载器
            
        返回:
            avg_losses: 平均损失
        """
        self.generator.eval()
        self.discriminator.eval()
        
        val_losses = {key: [] for key in ['g_loss', 'd_loss', 'g_complex_loss', 'g_mag_loss']}
        
        with torch.no_grad():
            for raw, clean in dataloader:
                raw = raw.to(self.device)
                clean = clean.to(self.device)
                
                # Generator 前向传播
                fake_clean = self.generator(raw)
                
                # 计算损失
                # 复数 L1 损失
                loss_g_complex = self.criterion_l1(fake_clean, clean)
                
                # 幅度 L1 损失
                fake_mag = self.compute_magnitude(fake_clean)
                clean_mag = self.compute_magnitude(clean)
                loss_g_mag = self.criterion_l1(fake_mag, clean_mag)
                
                # GAN 损失
                pred_fake = self.discriminator(raw, fake_clean)
                loss_g_gan = self.criterion_gan(pred_fake, True)
                
                loss_g = (self.lambda_complex * loss_g_complex + 
                          self.lambda_mag * loss_g_mag + 
                          self.lambda_gan * loss_g_gan)
                
                pred_real = self.discriminator(raw, clean)
                pred_fake = self.discriminator(raw, fake_clean)
                loss_d = (self.criterion_gan(pred_real, True) + 
                         self.criterion_gan(pred_fake, False)) * 0.5
                
                val_losses['g_loss'].append(loss_g.item())
                val_losses['d_loss'].append(loss_d.item())
                val_losses['g_complex_loss'].append(loss_g_complex.item())
                val_losses['g_mag_loss'].append(loss_g_mag.item())
        
        avg_losses = {key: np.mean(values) for key, values in val_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, save_path, epoch, is_best=False):
        """
        保存模型检查点
        
        参数:
            save_path: 保存路径
            epoch: 当前 epoch
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'history': self.history
        }
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = save_path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载模型检查点
        
        参数:
            checkpoint_path: 检查点路径
            
        返回:
            epoch: 恢复的 epoch
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.history = checkpoint['history']
        
        return checkpoint['epoch']


def train_cgan(
    train_dataset,
    test_dataset,
    generator,
    discriminator,
    device,
    num_epochs=100,
    batch_size=32,
    lr_g=0.0002,
    lr_d=0.0002,
    lambda_complex=100.0,
    lambda_mag=50.0,
    lambda_gan=1.0,
    warmup_epochs=5,
    save_dir='./checkpoints',
    log_interval=10
):
    """
    完整的 cGAN 训练流程
    
    参数:
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        generator: Generator 模型
        discriminator: Discriminator 模型
        device: 训练设备
        num_epochs: 训练轮数
        batch_size: 批大小
        lr_g, lr_d: 学习率
        lambda_complex: 复数L1损失权重
        lambda_mag: 幅度L1损失权重
        lambda_gan: GAN损失权重
        warmup_epochs: 预热阶段epoch数
        save_dir: 模型保存目录
        log_interval: 日志输出间隔
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 创建训练器
    trainer = cGANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        lr_g=lr_g,
        lr_d=lr_d,
        lambda_complex=lambda_complex,
        lambda_mag=lambda_mag,
        lambda_gan=lambda_gan,
        warmup_epochs=warmup_epochs
    )
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_losses = trainer.train_epoch(train_loader, epoch)
        
        # 验证
        val_losses = trainer.validate(test_loader)
        
        # 日志输出
        if epoch % log_interval == 0:
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train - G: {train_losses['g_loss']:.4f}, D: {train_losses['d_loss']:.4f}, Complex: {train_losses['g_complex_loss']:.4f}, Mag: {train_losses['g_mag_loss']:.4f}")
            print(f"  Val   - G: {val_losses['g_loss']:.4f}, D: {val_losses['d_loss']:.4f}, Complex: {val_losses['g_complex_loss']:.4f}, Mag: {val_losses['g_mag_loss']:.4f}")
        
        # 保存检查点
        is_best = val_losses['g_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['g_loss']
        
        if epoch % 10 == 0 or is_best:
            save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            trainer.save_checkpoint(save_path, epoch, is_best=is_best)
    
    print("\n训练完成!")
    return trainer


if __name__ == "__main__":
    print("训练模块测试")
    print("请使用 train.py 启动完整训练")
