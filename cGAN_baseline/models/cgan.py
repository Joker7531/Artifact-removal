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
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.enc4 = self._conv_block(base_filters * 4, base_filters * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 8, base_filters * 8)
        
        # Decoder (上采样路径 + Skip Connections)
        self.dec4 = self._upconv_block(base_filters * 8, base_filters * 8)
        self.dec3 = self._upconv_block(base_filters * 16, base_filters * 4)  # 16 = 8 + 8 (skip)
        self.dec2 = self._upconv_block(base_filters * 8, base_filters * 2)   # 8 = 4 + 4
        self.dec1 = self._upconv_block(base_filters * 4, base_filters)       # 4 = 2 + 2
        
        # 输出层
        self.final = nn.Sequential(
            nn.Conv2d(base_filters * 2, out_channels, kernel_size=3, padding=1),  # 2 = 1 + 1
            nn.Tanh()  # 实部和虚部可以是负值
        )
        
        self.pool = nn.MaxPool2d(2)
        
    def _conv_block(self, in_ch, out_ch, normalize=True):
        """卷积块：Conv -> BN -> LeakyReLU"""
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _upconv_block(self, in_ch, out_ch):
        """上采样块：ConvTranspose -> BN -> ReLU"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)           # (B, 64, F, T)
        e2 = self.enc2(self.pool(e1))   # (B, 128, F/2, T/2)
        e3 = self.enc3(self.pool(e2))   # (B, 256, F/4, T/4)
        e4 = self.enc4(self.pool(e3))   # (B, 512, F/8, T/8)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 512, F/16, T/16)
        
        # Decoder with skip connections (使用插值处理尺寸不匹配)
        d4 = self.dec4(b)                   # (B, 512, F/8, T/8)
        # 调整 d4 尺寸以匹配 e4
        if d4.shape[2:] != e4.shape[2:]:
            d4 = nn.functional.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)     # (B, 1024, F/8, T/8)
        
        d3 = self.dec3(d4)                  # (B, 256, F/4, T/4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = nn.functional.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)     # (B, 512, F/4, T/4)
        
        d2 = self.dec2(d3)                  # (B, 128, F/2, T/2)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = nn.functional.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)     # (B, 256, F/2, T/2)
        
        d1 = self.dec1(d2)                  # (B, 64, F, T)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = nn.functional.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)     # (B, 128, F, T)
        
        # Output
        out = self.final(d1)                # (B, 1, F, T)
        
        return out


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for conditional GAN (双通道输入)
    
    输入形状: (batch, 4, freq_bins, time_frames)  # 4 = raw(2) + clean/fake(2)
    输出形状: (batch, 1, H', W')  # Patch-level 判别
    """
    
    def __init__(self, in_channels=4, base_filters=64):
        super(PatchGANDiscriminator, self).__init__()
        
        # 使用卷积层逐步降低空间分辨率
        self.model = nn.Sequential(
            # 第一层：不使用 BatchNorm
            nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第二层
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第三层
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第四层
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出层：输出 Patch 判别结果
            nn.Conv2d(base_filters * 8, 1, kernel_size=4, stride=1, padding=1)
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
    freq_bins = 528  # n_fft=1024, pad到528
    time_frames = 64
    
    # 创建模型
    generator = UNetGenerator(in_channels=2, out_channels=2, base_filters=64)
    discriminator = PatchGANDiscriminator(in_channels=4, base_filters=64)
    
    # 测试输入 (双通道: 实部+虚部)
    raw = torch.randn(batch_size, 2, freq_bins, time_frames)
    clean = torch.randn(batch_size, 2, freq_bins, time_frames)
    
    print("=" * 60)
    print("模型测试 (双通道: 实部+虚部)")
    print("=" * 60)
    
    # Generator 测试
    print(f"\nGenerator:")
    print(f"  输入形状: {raw.shape}")
    fake_clean = generator(raw)
    print(f"  输出形状: {fake_clean.shape}")
    
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
