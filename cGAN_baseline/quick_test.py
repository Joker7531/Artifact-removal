"""
快速验证脚本：测试训练流程是否正常

功能：
1. 加载少量数据
2. 创建模型
3. 运行 2-3 个 epoch
4. 验证所有模块是否正常工作
"""

import os
import sys
import torch
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import prepare_data
from models import UNetGenerator, PatchGANDiscriminator
from utils import cGANTrainer


def quick_test():
    """快速测试训练流程"""
    
    print("="*60)
    print("快速验证测试")
    print("="*60)
    
    # 1. 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1/6] 设备检查: {device}")
    
    # 2. 数据加载测试
    print("\n[2/6] 数据加载测试...")
    try:
        train_dataset, test_dataset = prepare_data(
            npz_path='../../dataset_cz_v2.npz',
            n_fft=256,
            hop_length=128,
            n_frames=32,
            stride=16,  # 更大的stride，减少样本数
            train_ratio=0.8,
            random_seed=42
        )
        print(f"   训练集: {len(train_dataset)} 样本")
        print(f"   测试集: {len(test_dataset)} 样本")
        
        # 测试样本形状
        raw, clean = train_dataset[0]
        print(f"   样本形状: raw={raw.shape}, clean={clean.shape}")
        print("   数据加载: OK")
    except Exception as e:
        print(f"   数据加载失败: {e}")
        return False
    
    # 3. 模型创建测试
    print("\n[3/6] 模型创建测试...")
    try:
        generator = UNetGenerator(in_channels=1, out_channels=1, base_filters=32).to(device)  # 减小模型
        discriminator = PatchGANDiscriminator(in_channels=2, base_filters=32).to(device)
        
        g_params = sum(p.numel() for p in generator.parameters())
        d_params = sum(p.numel() for p in discriminator.parameters())
        print(f"   Generator 参数量: {g_params:,}")
        print(f"   Discriminator 参数量: {d_params:,}")
        print("   模型创建: OK")
    except Exception as e:
        print(f"   模型创建失败: {e}")
        return False
    
    # 4. 前向传播测试
    print("\n[4/6] 前向传播测试...")
    try:
        raw_batch = raw.unsqueeze(0).to(device)
        clean_batch = clean.unsqueeze(0).to(device)
        
        with torch.no_grad():
            fake_clean = generator(raw_batch)
            pred_real = discriminator(raw_batch, clean_batch)
            pred_fake = discriminator(raw_batch, fake_clean)
        
        print(f"   Generator 输出: {fake_clean.shape}")
        print(f"   Discriminator 输出 (real): {pred_real.shape}")
        print(f"   Discriminator 输出 (fake): {pred_fake.shape}")
        print("   前向传播: OK")
    except Exception as e:
        print(f"   前向传播失败: {e}")
        return False
    
    # 5. 训练器创建测试
    print("\n[5/6] 训练器创建测试...")
    try:
        trainer = cGANTrainer(
            generator=generator,
            discriminator=discriminator,
            device=device,
            lr_g=0.0002,
            lr_d=0.0002,
            lambda_l1=100.0,
            lambda_gan=1.0
        )
        print("   训练器创建: OK")
    except Exception as e:
        print(f"   训练器创建失败: {e}")
        return False
    
    # 6. 训练步骤测试（仅 2 个 epoch）
    print("\n[6/6] 训练流程测试（2 个 epoch）...")
    try:
        from torch.utils.data import DataLoader, Subset
        
        # 只使用前 100 个样本
        train_subset = Subset(train_dataset, list(range(min(100, len(train_dataset)))))
        test_subset = Subset(test_dataset, list(range(min(20, len(test_dataset)))))
        
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=8, shuffle=False)
        
        print(f"   使用 {len(train_subset)} 训练样本, {len(test_subset)} 测试样本")
        
        for epoch in range(1, 3):
            print(f"\n   Epoch {epoch}/2:")
            
            # 训练
            train_losses = trainer.train_epoch(train_loader, epoch)
            print(f"      Train - G: {train_losses['g_loss']:.4f}, "
                  f"D: {train_losses['d_loss']:.4f}, "
                  f"L1: {train_losses['g_l1_loss']:.4f}")
            
            # 验证
            val_losses = trainer.validate(test_loader)
            print(f"      Val   - G: {val_losses['g_loss']:.4f}, "
                  f"D: {val_losses['d_loss']:.4f}, "
                  f"L1: {val_losses['g_l1_loss']:.4f}")
        
        print("\n   训练流程: OK")
    except Exception as e:
        print(f"   训练流程失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. 模型保存/加载测试
    print("\n[7/7] 模型保存/加载测试...")
    try:
        test_checkpoint_path = './test_checkpoint.pth'
        trainer.save_checkpoint(test_checkpoint_path, epoch=2)
        print(f"   模型已保存到: {test_checkpoint_path}")
        
        # 重新加载
        loaded_epoch = trainer.load_checkpoint(test_checkpoint_path)
        print(f"   模型已加载，epoch: {loaded_epoch}")
        
        # 清理
        if os.path.exists(test_checkpoint_path):
            os.remove(test_checkpoint_path)
            print("   测试文件已清理")
        
        print("   模型保存/加载: OK")
    except Exception as e:
        print(f"   模型保存/加载失败: {e}")
        return False
    
    print("\n" + "="*60)
    print("所有测试通过！")
    print("="*60)
    print("\n可以开始正式训练：")
    print("  python train.py --num_epochs 100 --batch_size 32")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
