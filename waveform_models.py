"""
1D Waveform GAN Models
使用 1D 卷积处理时序 EEG 信号
包含 WaveNet-style 生成器和 1D PatchGAN 判别器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """1D 残差块"""
    
    def __init__(self, channels, dilation=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, 
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, 
                               padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out = out + residual
        out = F.relu(out)
        
        return out


class WaveformGenerator(nn.Module):
    """
    简化的 1D U-Net Generator for Waveform Denoising
    输入: (B, 1, L) - raw waveform
    输出: (B, 1, L) - clean waveform
    带 Dropout 防止过拟合
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_filters=64, num_layers=4, dropout_rate=0.3):
        """
        Args:
            in_channels: 输入通道数 (1 for single channel EEG)
            out_channels: 输出通道数 (1 for denoised signal)
            base_filters: 基础滤波器数量
            num_layers: 编码器/解码器层数
            dropout_rate: Dropout 比率
        """
        super(WaveformGenerator, self).__init__()
        
        # 编码器 (添加 Dropout)
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, 15, stride=2, padding=7),
            nn.BatchNorm1d(base_filters),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_filters, base_filters*2, 15, stride=2, padding=7),
            nn.BatchNorm1d(base_filters*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(base_filters*2, base_filters*4, 15, stride=2, padding=7),
            nn.BatchNorm1d(base_filters*4),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(base_filters*4, base_filters*8, 15, stride=2, padding=7),
            nn.BatchNorm1d(base_filters*8),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_filters*8, dilation=1),
            ResidualBlock(base_filters*8, dilation=2),
            ResidualBlock(base_filters*8, dilation=4)
        )
        
        # 解码器 (添加 Dropout)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(base_filters*16, base_filters*4, 15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(base_filters*4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(base_filters*8, base_filters*2, 15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(base_filters*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(base_filters*4, base_filters, 15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(base_filters*2, base_filters, 15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 最终输出
        self.final = nn.Sequential(
            nn.Conv1d(base_filters, out_channels, 7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)      # (B, 64, L/2)
        e2 = self.enc2(e1)     # (B, 128, L/4)
        e3 = self.enc3(e2)     # (B, 256, L/8)
        e4 = self.enc4(e3)     # (B, 512, L/16)
        
        # Bottleneck
        b = self.bottleneck(e4)  # (B, 512, L/16)
        
        # 解码 (with skip connections and size matching)
        d4 = self.dec4(torch.cat([b, e4], dim=1))     # (B, 256, L/8)
        if d4.shape[2] != e3.shape[2]:
            d4 = F.interpolate(d4, size=e3.shape[2], mode='linear', align_corners=True)
        
        d3 = self.dec3(torch.cat([d4, e3], dim=1))    # (B, 128, L/4)
        if d3.shape[2] != e2.shape[2]:
            d3 = F.interpolate(d3, size=e2.shape[2], mode='linear', align_corners=True)
        
        d2 = self.dec2(torch.cat([d3, e2], dim=1))    # (B, 64, L/2)
        if d2.shape[2] != e1.shape[2]:
            d2 = F.interpolate(d2, size=e1.shape[2], mode='linear', align_corners=True)
        
        d1 = self.dec1(torch.cat([d2, e1], dim=1))    # (B, 64, L)
        if d1.shape[2] != x.shape[2]:
            d1 = F.interpolate(d1, size=x.shape[2], mode='linear', align_corners=True)
        
        # 输出
        out = self.final(d1)   # (B, 1, L)
        
        return out


class WaveformDiscriminator(nn.Module):
    """
    1D PatchGAN Discriminator for Waveform
    判别 raw 和 clean/generated 波形对的真假
    带 Dropout 防止过拟合
    """
    
    def __init__(self, in_channels=2, base_filters=64, num_layers=4, dropout_rate=0.3):
        """
        Args:
            in_channels: 输入通道数 (2 = raw + target/generated)
            base_filters: 基础滤波器数量
            num_layers: 判别器层数
            dropout_rate: Dropout 比率
        """
        super(WaveformDiscriminator, self).__init__()
        
        layers = []
        current_filters = base_filters
        
        # 第一层 (无 BatchNorm, 添加 Dropout)
        layers.append(nn.Sequential(
            nn.Conv1d(in_channels, current_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate)
        ))
        
        # 中间层 (添加 Dropout)
        for i in range(num_layers - 1):
            prev_filters = current_filters
            current_filters = min(current_filters * 2, 512)
            
            layers.append(nn.Sequential(
                nn.Conv1d(prev_filters, current_filters, kernel_size=4, 
                         stride=2, padding=1, bias=False),
                nn.BatchNorm1d(current_filters),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate)
            ))
        
        # 最后一层 (添加 Dropout)
        layers.append(nn.Sequential(
            nn.Conv1d(current_filters, current_filters, kernel_size=4, 
                     stride=1, padding=1, bias=False),
            nn.BatchNorm1d(current_filters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate)
        ))
        
        # 输出层
        layers.append(nn.Conv1d(current_filters, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, raw, target):
        """
        Args:
            raw: 原始波形 (B, 1, L)
            target: 目标/生成波形 (B, 1, L)
            
        Returns:
            判别结果 (B, 1, L')
        """
        # 拼接输入
        combined = torch.cat([raw, target], dim=1)  # (B, 2, L)
        return self.model(combined)


class GANLoss(nn.Module):
    """
    GAN Loss (支持 LSGAN 和 Vanilla GAN)
    """
    
    def __init__(self, gan_mode='lsgan'):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')
    
    def forward(self, prediction, target_is_real):
        """
        Args:
            prediction: 判别器输出
            target_is_real: True 表示真实样本,False 表示生成样本
            
        Returns:
            loss 值
        """
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        
        return self.loss(prediction, target)


def weights_init(m):
    """
    初始化网络权重
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    generator = WaveformGenerator(in_channels=1, out_channels=1, 
                                  base_filters=64, num_layers=4).to(device)
    discriminator = WaveformDiscriminator(in_channels=2, base_filters=64, 
                                         num_layers=4).to(device)
    
    # 初始化权重
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # 测试输入 (3 秒 @ 100Hz = 300 samples)
    batch_size = 4
    seq_length = 300
    
    raw_waveform = torch.randn(batch_size, 1, seq_length).to(device)
    clean_waveform = torch.randn(batch_size, 1, seq_length).to(device)
    
    # 生成器前向传播
    print("\n=== Generator Test ===")
    generated = generator(raw_waveform)
    print(f"Input shape: {raw_waveform.shape}")
    print(f"Generated shape: {generated.shape}")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # 判别器前向传播
    print("\n=== Discriminator Test ===")
    real_output = discriminator(raw_waveform, clean_waveform)
    fake_output = discriminator(raw_waveform, generated.detach())
    print(f"Real output shape: {real_output.shape}")
    print(f"Fake output shape: {fake_output.shape}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # 测试损失
    print("\n=== Loss Test ===")
    criterion_gan = GANLoss(gan_mode='lsgan').to(device)
    criterion_l1 = nn.L1Loss()
    
    loss_g_gan = criterion_gan(fake_output, True)
    loss_g_l1 = criterion_l1(generated, clean_waveform)
    loss_g = loss_g_gan + 100 * loss_g_l1
    
    loss_d_real = criterion_gan(real_output, True)
    loss_d_fake = criterion_gan(fake_output, False)
    loss_d = (loss_d_real + loss_d_fake) * 0.5
    
    print(f"Generator Loss: {loss_g.item():.4f} (GAN: {loss_g_gan.item():.4f}, L1: {loss_g_l1.item():.4f})")
    print(f"Discriminator Loss: {loss_d.item():.4f}")
    
    print("\n✓ All models work correctly!")
