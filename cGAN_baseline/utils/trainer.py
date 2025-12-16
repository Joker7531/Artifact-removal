"""
训练模块：cGAN 训练循环、损失函数、优化器配置

核心功能：
1. Generator 损失：L1 重建损失 + GAN 对抗损失
2. Discriminator 损失：真假判别损失
3. 训练循环：交替训练 D 和 G
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
    cGAN 训练器
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
        lambda_l1=100.0,  # L1 损失权重（主要）
        lambda_gan=1.0,   # GAN 损失权重（辅助）
        gan_mode='lsgan'
    ):
        """
        参数:
            generator: Generator 模型
            discriminator: Discriminator 模型
            device: 训练设备
            lr_g: Generator 学习率
            lr_d: Discriminator 学习率
            beta1, beta2: Adam 优化器参数
            lambda_l1: L1 重建损失权重
            lambda_gan: GAN 对抗损失权重
            gan_mode: GAN 损失类型
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        self.lambda_l1 = lambda_l1
        self.lambda_gan = lambda_gan
        
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
            'g_l1_loss': [],
            'g_gan_loss': [],
            'd_real_loss': [],
            'd_fake_loss': []
        }
    
    def train_step(self, raw, clean) -> Dict[str, float]:
        """
        单步训练
        
        参数:
            raw: (B, 1, F, T)
            clean: (B, 1, F, T)
            
        返回:
            losses: 各项损失值的字典
        """
        raw = raw.to(self.device)
        clean = clean.to(self.device)
        
        # ==================== 训练 Discriminator ====================
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
        
        # ==================== 训练 Generator ====================
        self.optimizer_g.zero_grad()
        
        # GAN 损失：欺骗 Discriminator
        fake_clean = self.generator(raw)
        pred_fake = self.discriminator(raw, fake_clean)
        loss_g_gan = self.criterion_gan(pred_fake, True)
        
        # L1 重建损失
        loss_g_l1 = self.criterion_l1(fake_clean, clean)
        
        # 总 G 损失
        loss_g = self.lambda_gan * loss_g_gan + self.lambda_l1 * loss_g_l1
        loss_g.backward()
        self.optimizer_g.step()
        
        # 记录损失
        losses = {
            'g_loss': loss_g.item(),
            'd_loss': loss_d.item(),
            'g_l1_loss': loss_g_l1.item(),
            'g_gan_loss': loss_g_gan.item(),
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
        
        epoch_losses = {key: [] for key in ['g_loss', 'd_loss', 'g_l1_loss', 'g_gan_loss', 'd_real_loss', 'd_fake_loss']}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for raw, clean in pbar:
            losses = self.train_step(raw, clean)
            
            for key, value in losses.items():
                epoch_losses[key].append(value)
            
            # 更新进度条
            pbar.set_postfix({
                'G': f"{losses['g_loss']:.4f}",
                'D': f"{losses['d_loss']:.4f}",
                'L1': f"{losses['g_l1_loss']:.4f}"
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
        
        val_losses = {key: [] for key in ['g_loss', 'd_loss', 'g_l1_loss']}
        
        with torch.no_grad():
            for raw, clean in dataloader:
                raw = raw.to(self.device)
                clean = clean.to(self.device)
                
                # Generator 前向传播
                fake_clean = self.generator(raw)
                
                # 计算损失
                pred_fake = self.discriminator(raw, fake_clean)
                loss_g_gan = self.criterion_gan(pred_fake, True)
                loss_g_l1 = self.criterion_l1(fake_clean, clean)
                loss_g = self.lambda_gan * loss_g_gan + self.lambda_l1 * loss_g_l1
                
                pred_real = self.discriminator(raw, clean)
                pred_fake = self.discriminator(raw, fake_clean)
                loss_d = (self.criterion_gan(pred_real, True) + 
                         self.criterion_gan(pred_fake, False)) * 0.5
                
                val_losses['g_loss'].append(loss_g.item())
                val_losses['d_loss'].append(loss_d.item())
                val_losses['g_l1_loss'].append(loss_g_l1.item())
        
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
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
    lambda_l1=100.0,
    lambda_gan=1.0,
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
        lambda_l1, lambda_gan: 损失权重
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
        lambda_l1=lambda_l1,
        lambda_gan=lambda_gan
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
            print(f"  Train - G: {train_losses['g_loss']:.4f}, D: {train_losses['d_loss']:.4f}, L1: {train_losses['g_l1_loss']:.4f}")
            print(f"  Val   - G: {val_losses['g_loss']:.4f}, D: {val_losses['d_loss']:.4f}, L1: {val_losses['g_l1_loss']:.4f}")
        
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
