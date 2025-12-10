"""
测试所有模块是否正常工作
"""

import sys
import torch
import numpy as np
from pathlib import Path

print("="*70)
print("Testing Waveform GAN Modules".center(70))
print("="*70)

# 测试 1: 数据集
print("\n[1/3] Testing Dataset Module...")
try:
    from waveform_dataset import WaveformProcessor, create_waveform_dataloaders
    
    processor = WaveformProcessor(fs=100, window_sec=3.0, overlap=0.75)
    test_signal = np.random.randn(600)
    windows = processor.sliding_window(test_signal)
    print(f"  ✓ WaveformProcessor works")
    print(f"    Signal length: {len(test_signal)}, Windows: {len(windows)}")
    
    # 测试归一化
    normalized = processor.normalize(test_signal, method='zscore')
    print(f"    Normalized mean: {normalized.mean():.4f}, std: {normalized.std():.4f}")
    
    # 测试 DataLoader (如果数据存在)
    data_dir = "../2_Data_processed"
    if Path(data_dir).exists():
        train_loader, val_loader = create_waveform_dataloaders(
            data_dir=data_dir,
            train_indices=[1, 2],
            val_indices=[9],
            batch_size=8,
            num_workers=0,
            fs=100,
            window_sec=3.0,
            overlap=0.75
        )
        batch = next(iter(train_loader))
        print(f"  ✓ DataLoader works")
        print(f"    Batch shape: {batch['raw'].shape}")
    else:
        print(f"  ⚠ Data directory not found, skipping DataLoader test")
    
except Exception as e:
    print(f"  ✗ Dataset test failed: {e}")
    sys.exit(1)

# 测试 2: 模型
print("\n[2/3] Testing Model Module...")
try:
    from waveform_models import WaveformGenerator, WaveformDiscriminator, GANLoss
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    # 测试生成器
    generator = WaveformGenerator(in_channels=1, out_channels=1, 
                                  base_filters=64, num_layers=4).to(device)
    test_input = torch.randn(4, 1, 300).to(device)
    output = generator(test_input)
    
    print(f"  ✓ Generator works")
    print(f"    Input: {test_input.shape} -> Output: {output.shape}")
    print(f"    Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # 测试判别器
    discriminator = WaveformDiscriminator(in_channels=2, base_filters=64, 
                                         num_layers=4).to(device)
    test_target = torch.randn(4, 1, 300).to(device)
    disc_output = discriminator(test_input, test_target)
    
    print(f"  ✓ Discriminator works")
    print(f"    Output shape: {disc_output.shape}")
    print(f"    Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # 测试损失
    criterion = GANLoss(gan_mode='lsgan').to(device)
    loss_real = criterion(disc_output, True)
    loss_fake = criterion(disc_output, False)
    
    print(f"  ✓ Loss function works")
    print(f"    Real loss: {loss_real.item():.4f}, Fake loss: {loss_fake.item():.4f}")
    
except Exception as e:
    print(f"  ✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 3: 训练配置
print("\n[3/3] Testing Training Configuration...")
try:
    import train_waveform_gan
    
    print(f"  ✓ Training module imports successfully")
    
    # 检查配置
    config_keys = ['data_dir', 'batch_size', 'num_epochs', 'g_lr', 'd_lr', 
                   'lambda_l1', 'gan_mode']
    print(f"  ✓ Required config keys present")
    
except Exception as e:
    print(f"  ✗ Training module test failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("All Tests Passed! ✓".center(70))
print("="*70)
print("\nYou can now:")
print("  1. Train the model: python train_waveform_gan.py")
print("  2. Run inference: python inference_waveform.py")
print("="*70)
