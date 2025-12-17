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


class WGANLoss(nn.Module):
    """WGAN 损失函数（Wasserstein距离）"""
    
    def __init__(self):
        super(WGANLoss, self).__init__()
    
    def forward(self, prediction, target_is_real):
        """
        WGAN损失：
        - 真实样本：最大化D(real)，即最小化-D(real)
        - 假样本：最小化D(fake)，即最大化-D(fake)
        
        参数:
            prediction: Discriminator 输出（未激活的分数）
            target_is_real: 是否为真实样本
        """
        if target_is_real:
            # 对于真实样本，判别器希望输出尽可能大（正值）
            # Loss = -mean(D(real))
            loss = -prediction.mean()
        else:
            # 对于假样本，判别器希望输出尽可能小（负值）
            # Loss = mean(D(fake))
            loss = prediction.mean()
        
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
        lr_g=0.0001,           # WGAN-GP推荐较小的学习率
        lr_d=0.0001,           # D和G使用相同学习率
        beta1=0.0,             # WGAN-GP推荐beta1=0
        beta2=0.9,             # WGAN-GP推荐beta2=0.9
        lambda_complex=100.0,  # 复数 L1 损失权重（主要）
        lambda_mag=50.0,       # 幅度 L1 损失权重
        lambda_gan=1.0,        # GAN 损失权重（辅助）
        lambda_gp=10.0,        # 梯度惩罚权重
        warmup_epochs=5        # 预热阶段epoch数（冻结判别器）
    ):
        """
        参数:
            generator: Generator 模型
            discriminator: Discriminator 模型
            device: 训练设备
            lr_g: Generator 学习率
            lr_d: Discriminator 学习率
            beta1, beta2: Adam 优化器参数（WGAN-GP: beta1=0, beta2=0.9）
            lambda_complex: 复数 L1 损失权重（实部+虚部）
            lambda_mag: 幅度 L1 损失权重
            lambda_gan: GAN 对抗损失权重
            lambda_gp: 梯度惩罚权重
            warmup_epochs: 预热阶段epoch数
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        self.lambda_complex = lambda_complex
        self.lambda_mag = lambda_mag
        self.lambda_gan = lambda_gan
        self.lambda_gp = lambda_gp
        self.warmup_epochs = warmup_epochs
        
        # 损失函数（WGAN）
        self.criterion_gan = WGANLoss()
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
        
        # 保存初始学习率用于衰减计算
        self.initial_lr_g = lr_g
        self.initial_lr_d = lr_d
        
        # 训练历史
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'g_complex_loss': [],
            'g_mag_loss': [],
            'g_gan_loss': [],
            'd_real_loss': [],
            'd_fake_loss': [],
            'gradient_penalty': [],  # 梯度惩罚
            'lr_g': [],  # 记录G的学习率
            'lr_d': []   # 记录D的学习率
        }
    
    def compute_gradient_penalty(self, real_data, fake_data, raw):
        """
        计算WGAN-GP的梯度惩罚项
        
        参数:
            real_data: 真实clean数据 (B, 2, F, T)
            fake_data: 生成的clean数据 (B, 2, F, T)
            raw: 条件输入 (B, 2, F, T)
            
        返回:
            gradient_penalty: 梯度惩罚值
        """
        batch_size = real_data.size(0)
        
        # 随机插值系数
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        alpha = alpha.expand_as(real_data)
        
        # 插值数据
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        # 判别器输出
        disc_interpolates = self.discriminator(raw, interpolates)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 梯度的L2范数
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # 梯度惩罚：(||gradient|| - 1)^2
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def update_learning_rate(self, epoch, num_epochs, decay_start_epoch=100):
        """
        更新学习率（线性衰减策略）
        
        参数:
            epoch: 当前epoch
            num_epochs: 总epoch数
            decay_start_epoch: 开始衰减的epoch（前decay_start_epoch个epoch保持不变）
        """
        if epoch <= decay_start_epoch:
            # 前decay_start_epoch个epoch保持初始学习率
            lr_g = self.initial_lr_g
            lr_d = self.initial_lr_d
        else:
            # 后续epoch线性衰减到0
            decay_epochs = num_epochs - decay_start_epoch
            current_decay = epoch - decay_start_epoch
            decay_factor = 1.0 - (current_decay / decay_epochs)
            lr_g = self.initial_lr_g * decay_factor
            lr_d = self.initial_lr_d * decay_factor
        
        # 更新优化器学习率
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] = lr_g
        for param_group in self.optimizer_d.param_groups:
            param_group['lr'] = lr_d
        
        return lr_g, lr_d
    
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
        单步训练（WGAN-GP版本）
        
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
            
            # 梯度惩罚
            gradient_penalty = self.compute_gradient_penalty(clean, fake_clean.detach(), raw)
            
            # 总 D 损失 = -E[D(real)] + E[D(fake)] + λ_gp * GP
            loss_d = loss_d_real + loss_d_fake + self.lambda_gp * gradient_penalty
            loss_d.backward()
            self.optimizer_d.step()
            
            gp_value = gradient_penalty.item()
        else:
            # 预热阶段：仅计算但不更新判别器
            with torch.no_grad():
                fake_clean = self.generator(raw)
                pred_real = self.discriminator(raw, clean)
                pred_fake = self.discriminator(raw, fake_clean)
                loss_d_real = self.criterion_gan(pred_real, True)
                loss_d_fake = self.criterion_gan(pred_fake, False)
                gradient_penalty = torch.tensor(0.0).to(self.device)
                loss_d = loss_d_real + loss_d_fake
                gp_value = 0.0
        
        # ==================== 训练 Generator ====================
        self.optimizer_g.zero_grad()
        
        fake_clean = self.generator(raw)
        
        # 1. 复数 L1 损失（实部+虚部）
        loss_g_complex = self.criterion_l1(fake_clean, clean)
        
        # 2. 幅度 L1 损失
        fake_mag = self.compute_magnitude(fake_clean)
        clean_mag = self.compute_magnitude(clean)
        loss_g_mag = self.criterion_l1(fake_mag, clean_mag)
        
        # 3. WGAN 对抗损失（预热阶段lambda_gan=0）
        if epoch > self.warmup_epochs:
            pred_fake = self.discriminator(raw, fake_clean)
            # WGAN: 生成器希望判别器输出尽可能大（正值）
            # Loss_G = -mean(D(fake))
            loss_g_gan = -pred_fake.mean()
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
            'd_fake_loss': loss_d_fake.item(),
            'gradient_penalty': gp_value
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
        
        epoch_losses = {key: [] for key in ['g_loss', 'd_loss', 'g_complex_loss', 'g_mag_loss', 'g_gan_loss', 'd_real_loss', 'd_fake_loss', 'gradient_penalty']}
        
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
                'GP': f"{losses['gradient_penalty']:.4f}"
            })
        
        # 计算平均损失
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        # 记录历史
        for key, value in avg_losses.items():
            self.history[key].append(value)
        
        # 记录当前学习率
        current_lr_g = self.optimizer_g.param_groups[0]['lr']
        current_lr_d = self.optimizer_d.param_groups[0]['lr']
        self.history['lr_g'].append(current_lr_g)
        self.history['lr_d'].append(current_lr_d)
        
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
    lr_g=0.0001,  # WGAN-GP使用较小学习率
    lr_d=0.0001,
    lambda_complex=100.0,
    lambda_mag=50.0,
    lambda_gan=1.0,
    lambda_gp=10.0,  # 梯度惩罚权重
    warmup_epochs=5,
    lr_decay_start=100,  # 学习率开始衰减的epoch
    save_dir='./checkpoints',
    log_interval=10,
    visualize_dir='./results'
):
    """
    完整的 cGAN 训练流程（WGAN-GP版本）
    
    参数:
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        generator: Generator 模型
        discriminator: Discriminator 模型
        device: 训练设备
        num_epochs: 训练轮数
        batch_size: 批大小
        lr_g: Generator学习率
        lr_d: Discriminator学习率
        lambda_complex: 复数L1损失权重
        lambda_mag: 幅度L1损失权重
        lambda_gan: GAN损失权重
        lambda_gp: 梯度惩罚权重
        warmup_epochs: 预热阶段epoch数
        lr_decay_start: 学习率开始衰减的epoch（前lr_decay_start个epoch保持不变）
        save_dir: 模型保存目录
        log_interval: 日志输出间隔
        visualize_dir: 可视化结果保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(visualize_dir, exist_ok=True)
    
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
        lambda_gp=lambda_gp,
        warmup_epochs=warmup_epochs
    )
    
    # 导入可视化工具
    from .visualization import Visualizer
    visualizer = Visualizer(save_dir=visualize_dir)
    
    # 训练循环
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, num_epochs + 1):
        # 更新学习率（线性衰减策略）
        current_lr_g, current_lr_d = trainer.update_learning_rate(epoch, num_epochs, lr_decay_start)
        
        # 训练
        train_losses = trainer.train_epoch(train_loader, epoch)
        
        # 验证
        val_losses = trainer.validate(test_loader)
        
        # 日志输出
        if epoch % log_interval == 0:
            print(f"\nEpoch {epoch}/{num_epochs} (LR_G: {current_lr_g:.6f}, LR_D: {current_lr_d:.6f})")
            print(f"  Train - G: {train_losses['g_loss']:.4f}, D: {train_losses['d_loss']:.4f}, Complex: {train_losses['g_complex_loss']:.4f}, Mag: {train_losses['g_mag_loss']:.4f}")
            print(f"  Val   - G: {val_losses['g_loss']:.4f}, D: {val_losses['d_loss']:.4f}, Complex: {val_losses['g_complex_loss']:.4f}, Mag: {val_losses['g_mag_loss']:.4f}")
        
        # 只保存最佳模型
        is_best = val_losses['g_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['g_loss']
            best_epoch = epoch
            
            # 保存最佳模型（只保留一个最优模型）
            save_path = os.path.join(save_dir, 'best_model.pth')
            trainer.save_checkpoint(save_path, epoch, is_best=True)
            print(f"\n  ✓ 新的最佳模型已保存 (Val Loss: {best_val_loss:.4f})")
            
            # 进行3个样本的推理可视化
            print(f"  生成推理可视化样本...")
            visualize_inference_samples(
                trainer.generator,
                test_dataset,
                device,
                visualizer,
                epoch,
                num_samples=3
            )
            print(f"  ✓ 可视化完成，保存至 {visualize_dir}")
        
        # 每5个epoch更新训练历史绘图
        if epoch % 5 == 0:
            print(f"\n  更新训练历史曲线...")
            visualizer.plot_training_history(
                history=trainer.history,
                save_name='training_history_latest.png'
            )
    
    print(f"\n训练完成!")
    print(f"最佳模型: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
    return trainer


def visualize_inference_samples(
    generator,
    test_dataset,
    device,
    visualizer,
    epoch,
    num_samples=3
):
    """
    对测试集中的样本进行推理并可视化
    
    参数:
        generator: 生成器模型
        test_dataset: 测试数据集
        device: 设备
        visualizer: 可视化器
        epoch: 当前epoch
        num_samples: 可视化样本数量
    """
    generator.eval()
    
    with torch.no_grad():
        for idx in range(min(num_samples, len(test_dataset))):
            raw, clean = test_dataset[idx]
            
            # 增加batch维度
            raw_batch = raw.unsqueeze(0).to(device)
            
            # 推理
            fake_clean = generator(raw_batch)
            
            # 转为numpy（去除batch维度）
            raw_np = raw.cpu().numpy()  # (2, F, T)
            clean_np = clean.cpu().numpy()
            fake_np = fake_clean.squeeze(0).cpu().numpy()
            
            # 提取实部和虚部
            raw_real = raw_np[0]  # (F, T)
            raw_imag = raw_np[1]
            clean_real = clean_np[0]
            clean_imag = clean_np[1]
            fake_real = fake_np[0]
            fake_imag = fake_np[1]
            
            # 计算幅度谱
            raw_mag = np.sqrt(raw_real**2 + raw_imag**2)
            clean_mag = np.sqrt(clean_real**2 + clean_imag**2)
            fake_mag = np.sqrt(fake_real**2 + fake_imag**2)
            
            # 可视化STFT谱图（使用幅度）
            visualizer.plot_spectrograms(
                raw_mag=raw_mag,
                clean_mag=clean_mag,
                reconstructed_mag=fake_mag,
                sample_rate=500,  # EEG采样率
                hop_length=250,   # 与训练时一致
                save_name=f'inference_epoch_{epoch}_sample_{idx}.png'
            )
    
    generator.train()


if __name__ == "__main__":
    print("训练模块测试")
    print("请使用 train.py 启动完整训练")
