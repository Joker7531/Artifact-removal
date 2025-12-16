"""
测试推理功能（使用训练过程中保存的检查点）

使用方法:
    python test_inference.py
"""

import os
import sys
import torch
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import prepare_data
from models import UNetGenerator


def test_inference():
    """测试推理功能"""
    
    print("="*60)
    print("推理功能测试")
    print("="*60)
    
    # 检查是否有可用的检查点
    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        print(f"\n错误: 检查点目录不存在: {checkpoint_dir}")
        print("请先运行训练:")
        print("  python train.py --num_epochs 10 --batch_size 16")
        return False
    
    # 查找检查点文件
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not checkpoints:
        print(f"\n错误: 未找到检查点文件")
        print("请先运行训练:")
        print("  python train.py --num_epochs 10 --batch_size 16")
        return False
    
    # 使用最新的检查点
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"\n找到检查点: {latest_checkpoint}")
    
    # 1. 加载数据
    print("\n[1/3] 加载数据...")
    try:
        _, test_dataset = prepare_data(
            npz_path='../../dataset_cz_v2.npz',
            n_fft=256,
            hop_length=128,
            n_frames=32,
            stride=8,
            train_ratio=0.8,
            random_seed=42
        )
        print(f"   测试集: {len(test_dataset)} 样本")
    except Exception as e:
        print(f"   数据加载失败: {e}")
        return False
    
    # 2. 加载模型
    print("\n[2/3] 加载模型...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = UNetGenerator(in_channels=1, out_channels=1, base_filters=64).to(device)
        
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
        
        print(f"   模型已加载，训练轮数: {checkpoint['epoch']}")
    except Exception as e:
        print(f"   模型加载失败: {e}")
        return False
    
    # 3. 测试推理
    print("\n[3/3] 测试推理...")
    try:
        raw, clean = test_dataset[0]
        raw = raw.unsqueeze(0).to(device)
        
        with torch.no_grad():
            fake_clean = generator(raw)
        
        print(f"   输入形状: {raw.shape}")
        print(f"   输出形状: {fake_clean.shape}")
        print(f"   输出范围: [{fake_clean.min():.4f}, {fake_clean.max():.4f}]")
        
        # 验证输出是否合理
        if torch.isnan(fake_clean).any():
            print("   警告: 输出包含 NaN")
            return False
        if torch.isinf(fake_clean).any():
            print("   警告: 输出包含 Inf")
            return False
        
        print("   推理测试: OK")
        
    except Exception as e:
        print(f"   推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("推理功能正常！")
    print("="*60)
    print("\n可以运行完整推理和可视化:")
    print(f"  python inference.py --checkpoint {latest_checkpoint} --num_samples 10")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_inference()
    sys.exit(0 if success else 1)
