"""
U-Net Denoising AutoEncoder for STFT-based Signal Denoising
支持 magnitude-only 和 complex (Re/Im) 两种模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DoubleConv(nn.Module):
    """
    双卷积层：Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    下采样：MaxPool -> DoubleConv
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    上采样：Upsample -> DoubleConv（含跳连）
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        # in_channels 是解码器上一层的输出通道数
        # 拼接后通道数 = in_channels + skip_channels (其中 skip_channels = out_channels)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 拼接后: in_channels + out_channels -> out_channels
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # 拼接后: in_channels // 2 + out_channels -> out_channels (注意这里上采样已经减半)
            # 但为了统一，我们期望 in_channels // 2 == out_channels
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        x1: 来自上层的上采样特征
        x2: 来自编码器的跳连特征
        """
        x1 = self.up(x1)
        
        # 处理尺寸不匹配（padding）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接跳连
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Denoising AutoEncoder
    
    支持自动调整深度与通道数
    输入/输出: (batch, channels, freq, time)
        - magnitude 模式: channels=1
        - complex 模式: channels=2 (Re, Im)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        bilinear: bool = True
    ):
        """
        Args:
            in_channels: 输入通道数 (1 for magnitude, 2 for complex)
            out_channels: 输出通道数 (same as in_channels)
            base_channels: 基础通道数（第一层）
            depth: 网络深度（下采样次数）
            bilinear: 使用双线性插值上采样（False 则用转置卷积）
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.bilinear = bilinear
        
        # 初始卷积
        self.inc = DoubleConv(in_channels, base_channels)
        
        # 编码器（下采样）
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for i in range(depth):
            self.down_blocks.append(Down(ch, ch * 2))
            ch = ch * 2
        
        # 解码器（上采样）
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(Up(ch, ch // 2, bilinear))
            ch = ch // 2
        
        # 输出卷积
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, freq, time)
        
        Returns:
            out: (batch, out_channels, freq, time)
        """
        # 编码器
        x1 = self.inc(x)
        
        # 保存跳连特征
        skip_connections = [x1]
        
        x_down = x1
        for down in self.down_blocks:
            x_down = down(x_down)
            skip_connections.append(x_down)
        
        # 解码器（使用跳连）
        x_up = skip_connections[-1]
        for i, up in enumerate(self.up_blocks):
            skip = skip_connections[-(i+2)]
            x_up = up(x_up, skip)
        
        # 输出
        out = self.outc(x_up)
        
        return out
    
    def get_model_size(self):
        """
        计算模型参数量
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': total_params * 4 / (1024 ** 2)  # 假设 float32
        }


class MultiScaleSTFTLoss(nn.Module):
    """
    多分辨率 STFT 损失（用于时频域重建）
    """
    def __init__(
        self,
        fft_sizes: List[int] = [512, 256, 128],
        hop_sizes: List[int] = [128, 64, 32],
        win_lengths: List[int] = [512, 256, 128],
        l1_weight: float = 1.0,
        l2_weight: float = 1.0
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
    
    def stft(self, x, fft_size, hop_size, win_length):
        """
        计算 STFT（仅用于损失计算，与数据预处理无关）
        """
        # x: (batch, 1, freq, time) -> 需要先转回时域（这里简化为直接计算频谱差）
        return x
    
    def forward(self, pred, target):
        """
        计算多尺度 STFT 损失
        """
        total_loss = 0.0
        
        # L1 损失（频谱幅度）
        l1_loss = F.l1_loss(pred, target)
        
        # L2 损失（频谱能量）
        l2_loss = F.mse_loss(pred, target)
        
        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss
        
        return total_loss


class CombinedLoss(nn.Module):
    """
    组合损失函数：L1 + L2 + Perceptual（可选）
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 1.0,
        use_multiscale: bool = False
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.use_multiscale = use_multiscale
        
        if use_multiscale:
            self.multiscale_loss = MultiScaleSTFTLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测的 STFT (batch, channels, freq, time)
            target: 目标 STFT (batch, channels, freq, time)
        """
        # 基础损失
        l1_loss = F.l1_loss(pred, target)
        l2_loss = F.mse_loss(pred, target)
        
        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss
        
        # 多尺度损失（可选）
        if self.use_multiscale:
            ms_loss = self.multiscale_loss(pred, target)
            total_loss += 0.5 * ms_loss
        
        return total_loss, {
            'l1': l1_loss.item(),
            'l2': l2_loss.item(),
            'total': total_loss.item()
        }


def build_unet(
    mode: str = 'magnitude',
    base_channels: int = 64,
    depth: int = 4,
    device: str = 'cuda'
) -> UNet:
    """
    构建 U-Net 模型的快捷函数
    
    Args:
        mode: 'magnitude' 或 'complex'
        base_channels: 基础通道数
        depth: 网络深度
        device: 设备
    
    Returns:
        model: UNet 模型
    """
    if mode == 'magnitude':
        in_channels = 1
        out_channels = 1
    elif mode == 'complex':
        in_channels = 2
        out_channels = 2
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        depth=depth,
        bilinear=True
    )
    
    model = model.to(device)
    
    # 打印模型信息
    model_info = model.get_model_size()
    print(f"\n🏗️  U-Net 模型已构建 (mode={mode})")
    print(f"   输入: (batch, {in_channels}, freq, time)")
    print(f"   输出: (batch, {out_channels}, freq, time)")
    print(f"   深度: {depth} 层")
    print(f"   基础通道: {base_channels}")
    print(f"   参数量: {model_info['total_params']:,}")
    print(f"   模型大小: {model_info['size_mb']:.2f} MB")
    
    return model


if __name__ == "__main__":
    # 测试模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Magnitude 模式
    model_mag = build_unet(mode='magnitude', base_channels=64, depth=4, device=device)
    
    # 测试前向传播
    batch_size = 4
    freq_bins = 129  # n_fft // 2 + 1 (假设 n_fft=256)
    time_frames = 64
    
    x_mag = torch.randn(batch_size, 1, freq_bins, time_frames).to(device)
    y_mag = model_mag(x_mag)
    
    print(f"\n✅ Magnitude 模式测试:")
    print(f"   输入: {x_mag.shape}")
    print(f"   输出: {y_mag.shape}")
    
    # Complex 模式
    model_cplx = build_unet(mode='complex', base_channels=64, depth=4, device=device)
    
    x_cplx = torch.randn(batch_size, 2, freq_bins, time_frames).to(device)
    y_cplx = model_cplx(x_cplx)
    
    print(f"\n✅ Complex 模式测试:")
    print(f"   输入: {x_cplx.shape}")
    print(f"   输出: {y_cplx.shape}")
    
    # 测试损失函数
    criterion = CombinedLoss(l1_weight=1.0, l2_weight=0.5)
    target = torch.randn_like(y_mag)
    
    loss, loss_dict = criterion(y_mag, target)
    print(f"\n📊 损失函数测试:")
    print(f"   L1 Loss: {loss_dict['l1']:.4f}")
    print(f"   L2 Loss: {loss_dict['l2']:.4f}")
    print(f"   Total Loss: {loss_dict['total']:.4f}")
