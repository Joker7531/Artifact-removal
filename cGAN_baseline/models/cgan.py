"""
cGAN 模型定义：U-Net Generator + PatchGAN Discriminator

Generator: 
    - 基于 U-Net 架构（Encoder-Decoder with Skip Connections）
    - 输入: raw STFT (2, F, T) - 通道0: 实部, 通道1: 虚部
    - 输出: clean STFT (2, F, T)

Discriminator:
    - PatchGAN 架构
    - 输入: raw + clean/fake concatenated (4, F, T)
    - 输出: Patch-level 判别结果
"""

import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    """
    U-Net Generator for complex STFT mapping (实部+虚部双通道)
    
    输入形状: (batch, 2, freq_bins, time_frames) - 通道0: 实部, 通道1: 虚部
    输出形状: (batch, 2, freq_bins, time_frames)
    """
    
    def __init__(self, in_channels=2, out_channels=2, base_filters=64):
        super(UNetGenerator, self).__init__()
        
        # Encoder (下采样路径)
        self.enc1 = self._conv_block(in_channels, base_filters, normalize=False)
        self.down1 = self._downsample_block(base_filters, base_filters * 2)
        
        self.enc2 = self._conv_block(base_filters * 2, base_filters * 2)
        self.down2 = self._downsample_block(base_filters * 2, base_filters * 4)
        
        self.enc3 = self._conv_block(base_filters * 4, base_filters * 4)
        self.down3 = self._downsample_block(base_filters * 4, base_filters * 8)
        
        self.enc4 = self._conv_block(base_filters * 8, base_filters * 8)
        self.down4 = self._downsample_block(base_filters * 8, base_filters * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 8, base_filters * 8)
        
        # Decoder (上采样路径 + Skip Connections)
        self.dec4 = self._upconv_block(base_filters * 8, base_filters * 8)
        self.dec3 = self._upconv_block(base_filters * 16, base_filters * 4)  # 16 = 8 + 8 (skip)
        self.dec2 = self._upconv_block(base_filters * 8, base_filters * 2)   # 8 = 4 + 4
        self.dec1 = self._upconv_block(base_filters * 4, base_filters)       # 4 = 2 + 2
        
        # 输出层（线性输出，不使用Tanh）
        self.final = nn.Conv2d(base_filters * 2, out_channels, kernel_size=3, padding=1)
        
    def _conv_block(self, in_ch, out_ch, normalize=True, use_spectral_norm=True):
        """卷积块：Conv -> BN -> LeakyReLU（带spectral_norm）"""
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        
        layers = [conv]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _downsample_block(self, in_ch, out_ch, use_spectral_norm=True):
        """下采样块：使用步长卷积代替池化"""
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        return nn.Sequential(
            conv,
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _upconv_block(self, in_ch, out_ch, use_spectral_norm=True):
        """上采样块：Resize-Convolution代替转置卷积"""
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # Resize
            conv,  # Convolution
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)               # (B, 64, F, T)
        d1 = self.down1(e1)              # (B, 128, F/2, T/2)
        
        e2 = self.enc2(d1)               # (B, 128, F/2, T/2)
        d2 = self.down2(e2)              # (B, 256, F/4, T/4)
        
        e3 = self.enc3(d2)               # (B, 256, F/4, T/4)
        d3 = self.down3(e3)              # (B, 512, F/8, T/8)
        
        e4 = self.enc4(d3)               # (B, 512, F/8, T/8)
        d4 = self.down4(e4)              # (B, 512, F/16, T/16)
        
        # Bottleneck
        b = self.bottleneck(d4)          # (B, 512, F/16, T/16)
        
        # Decoder with skip connections
        up4 = self.dec4(b)               # (B, 512, F/8, T/8)
        up4 = torch.cat([up4, e4], dim=1)  # (B, 1024, F/8, T/8)
        
        up3 = self.dec3(up4)             # (B, 256, F/4, T/4)
        up3 = torch.cat([up3, e3], dim=1)  # (B, 512, F/4, T/4)
        
        up2 = self.dec2(up3)             # (B, 128, F/2, T/2)
        up2 = torch.cat([up2, e2], dim=1)  # (B, 256, F/2, T/2)
        
        up1 = self.dec1(up2)             # (B, 64, F, T)
        up1 = torch.cat([up1, e1], dim=1)  # (B, 128, F, T)
        
        # Output (线性输出)
        out = self.final(up1)            # (B, 2, F, T)
        
        return out


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for conditional GAN (双通道输入)
    
    输入形状: (batch, 4, freq_bins, time_frames)  # 4 = raw(2) + clean/fake(2)
    输出形状: (batch, 1, H', W')  # Patch-level 判别
    """
    
    def __init__(self, in_channels=4, base_filters=64):
        super(PatchGANDiscriminator, self).__init__()
        
        # 使用卷积层逐步降低空间分辨率（带spectral_norm）
        self.model = nn.Sequential(
            # 第一层：不使用 BatchNorm
            nn.utils.spectral_norm(nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第二层
            nn.utils.spectral_norm(nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第三层
            nn.utils.spectral_norm(nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第四层
            nn.utils.spectral_norm(nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=4, stride=1, padding=1)),
            nn.BatchNorm2d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出层：输出 Patch 判别结果
            nn.utils.spectral_norm(nn.Conv2d(base_filters * 8, 1, kernel_size=4, stride=1, padding=1))
        )
        
    def forward(self, raw, clean_or_fake):
        """
        参数:
            raw: (B, 2, F, T) - 条件输入（raw STFT 实部+虚部）
            clean_or_fake: (B, 2, F, T) - 真实或生成的 clean STFT
            
        返回:
            out: (B, 1, H', W') - Patch-level 判别结果
        """
        # 沿 channel 维度拼接
        x = torch.cat([raw, clean_or_fake], dim=1)  # (B, 4, F, T)
        out = self.model(x)
        return out


def test_models():
    """测试模型的输入输出形状"""
    batch_size = 4
    freq_bins = 96  # 0.5-40Hz约81个bin，pad到96
    time_frames = 64
    
    # 创建模型
    generator = UNetGenerator(in_channels=2, out_channels=2, base_filters=64)
    discriminator = PatchGANDiscriminator(in_channels=4, base_filters=64)
    
    # 测试输入 (双通道: 实部+虚部)
    raw = torch.randn(batch_size, 2, freq_bins, time_frames)
    clean = torch.randn(batch_size, 2, freq_bins, time_frames)
    
    print("=" * 60)
    print("模型测试 (双通道: 实部+虚部, Spectral Norm, Resize-Conv)")
    print("=" * 60)
    
    # Generator 测试
    print(f"\nGenerator:")
    print(f"  输入形状: {raw.shape}")
    fake_clean = generator(raw)
    print(f"  输出形状: {fake_clean.shape}")
    print(f"  输出范围: [{fake_clean.min():.4f}, {fake_clean.max():.4f}] (线性输出)")
    
    # Discriminator 测试
    print(f"\nDiscriminator:")
    print(f"  输入形状: raw={raw.shape}, clean={clean.shape}")
    real_out = discriminator(raw, clean)
    fake_out = discriminator(raw, fake_clean.detach())
    print(f"  输出形状 (real): {real_out.shape}")
    print(f"  输出形状 (fake): {fake_out.shape}")
    
    # 参数量统计
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\n参数量:")
    print(f"  Generator: {g_params:,}")
    print(f"  Discriminator: {d_params:,}")
    print("=" * 60)


if __name__ == "__main__":
    test_models()
