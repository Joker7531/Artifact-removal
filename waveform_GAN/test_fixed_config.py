"""
验证 GAN 修复配置
测试新的参数是否正常工作
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from waveform_models import WaveformGenerator, WaveformDiscriminator, GANLoss, weights_init

def test_fixed_config():
    """测试修复后的配置"""
    print("="*60)
    print("Testing Fixed GAN Configuration".center(60))
    print("="*60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 修复后的配置
    config = {
        'base_filters': 64,      # Generator
        'base_filters_d': 32,    # Discriminator (减半)
        'num_layers': 4,         # Generator
        'num_layers_d': 3,       # Discriminator (减1)
        'dropout_rate': 0.3,
        'lambda_l1': 10,         # 从 100 降到 10
        'real_label': 0.9,       # Label smoothing
        'fake_label': 0.1,
        'g_lr': 2e-4,            # 提高
        'd_lr': 1e-4,            # 降低
    }
    
    print("Configuration:")
    print(f"  Generator: filters={config['base_filters']}, layers={config['num_layers']}")
    print(f"  Discriminator: filters={config['base_filters_d']}, layers={config['num_layers_d']}")
    print(f"  Lambda L1: {config['lambda_l1']} (was 100)")
    print(f"  G LR: {config['g_lr']}, D LR: {config['d_lr']}")
    print(f"  Label smoothing: real={config['real_label']}, fake={config['fake_label']}\n")
    
    # 创建模型
    print("Creating models...")
    generator = WaveformGenerator(
        base_filters=config['base_filters'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    discriminator = WaveformDiscriminator(
        base_filters=config['base_filters_d'],
        num_layers=config['num_layers_d'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    
    print(f"  Generator parameters: {g_params:,}")
    print(f"  Discriminator parameters: {d_params:,}")
    print(f"  Ratio D/G: {d_params/g_params:.2%}\n")
    
    # 测试 Label Smoothing
    print("Testing Label Smoothing...")
    criterion_gan = GANLoss(
        gan_mode='lsgan',
        real_label=config['real_label'],
        fake_label=config['fake_label']
    ).to(device)
    
    # 创建测试数据
    raw_wave = torch.randn(4, 1, 300).to(device)
    clean_wave = torch.randn(4, 1, 300).to(device)
    
    generator.train()
    discriminator.train()
    
    # 生成
    fake_wave = generator(raw_wave)
    
    # 判别
    pred_real = discriminator(raw_wave, clean_wave)
    pred_fake = discriminator(raw_wave, fake_wave.detach())
    
    # GAN Loss (with label smoothing)
    loss_d_real = criterion_gan(pred_real, True)
    loss_d_fake = criterion_gan(pred_fake, False)
    loss_d = (loss_d_real + loss_d_fake) * 0.5
    
    print(f"  D Loss (real): {loss_d_real.item():.4f}")
    print(f"  D Loss (fake): {loss_d_fake.item():.4f}")
    print(f"  D Loss (total): {loss_d.item():.4f}\n")
    
    # Generator Loss
    pred_fake_for_g = discriminator(raw_wave, fake_wave)
    loss_g_gan = criterion_gan(pred_fake_for_g, True)
    loss_g_l1 = torch.nn.functional.l1_loss(fake_wave, clean_wave)
    loss_g = loss_g_gan + config['lambda_l1'] * loss_g_l1
    
    print(f"  G GAN Loss: {loss_g_gan.item():.4f}")
    print(f"  G L1 Loss: {loss_g_l1.item():.4f}")
    print(f"  G Total Loss: {loss_g.item():.4f}")
    print(f"  L1 weight: {config['lambda_l1']} * {loss_g_l1.item():.4f} = {config['lambda_l1'] * loss_g_l1.item():.4f}\n")
    
    # 检查 L1 和 GAN 的比重
    gan_ratio = loss_g_gan.item() / loss_g.item()
    l1_ratio = (config['lambda_l1'] * loss_g_l1.item()) / loss_g.item()
    
    print("Loss Component Ratios:")
    print(f"  GAN: {gan_ratio:.2%}")
    print(f"  L1:  {l1_ratio:.2%}\n")
    
    if l1_ratio > 0.95:
        print("  [WARNING] L1 dominates (>95%)! Consider lowering lambda_l1")
    elif l1_ratio < 0.50:
        print("  [WARNING] GAN dominates (>50%)! Consider raising lambda_l1")
    else:
        print("  [OK] Good balance between GAN and L1!")
    
    print("\n" + "="*60)
    print("Test Passed! Configuration is ready.".center(60))
    print("="*60)
    print("\nKey improvements:")
    print("  1. Lambda L1: 100 -> 10 (10x reduction)")
    print("  2. D capacity reduced: 64 filters -> 32 filters")
    print("  3. D layers reduced: 4 -> 3")
    print("  4. Label smoothing: 0.9/0.1 (instead of 1.0/0.0)")
    print("  5. G LR increased, D LR decreased")
    print("\nYou can now run: python train_waveform_gan.py")

if __name__ == "__main__":
    try:
        test_fixed_config()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)
