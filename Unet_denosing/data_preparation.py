"""
数据准备模块：加载、混合、划分 10 组 EEG 数据
支持 CSV 格式数据的 STFT 预处理
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class EEGDataPreparation:
    """
    EEG 数据准备类：负责加载、混合和划分数据
    """
    def __init__(
        self,
        data_dir: str = "2_Data_processed",
        clean_prefix: str = "clean_Cz_",
        noisy_prefix: str = "raw_Cz_",
        num_files: int = 10,
        test_ratio: float = 0.2,
        random_seed: int = 42
    ):
        """
        Args:
            data_dir: 数据目录
            clean_prefix: 干净信号前缀
            noisy_prefix: 带噪信号前缀
            num_files: 文件组数（默认 10 组）
            test_ratio: 测试集比例
            random_seed: 随机种子
        """
        self.data_dir = Path(data_dir)
        self.clean_prefix = clean_prefix
        self.noisy_prefix = noisy_prefix
        self.num_files = num_files
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # 设置随机种子
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
    def load_all_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        加载全部 10 组数据
        
        Returns:
            clean_signals: 干净信号列表
            noisy_signals: 带噪信号列表
        """
        clean_signals = []
        noisy_signals = []
        
        print(f"📂 开始加载 {self.num_files} 组数据...")
        
        for i in range(1, self.num_files + 1):
            clean_file = self.data_dir / f"{self.clean_prefix}{i:02d}.csv"
            noisy_file = self.data_dir / f"{self.noisy_prefix}{i:02d}.csv"
            
            if not clean_file.exists():
                print(f"⚠️  文件不存在: {clean_file}")
                continue
            if not noisy_file.exists():
                print(f"⚠️  文件不存在: {noisy_file}")
                continue
            
            # 加载 CSV（单列数据）
            clean_data = pd.read_csv(clean_file, header=None).values.flatten()
            noisy_data = pd.read_csv(noisy_file, header=None).values.flatten()
            
            # 检查长度一致性
            if len(clean_data) != len(noisy_data):
                print(f"⚠️  长度不匹配: {clean_file} ({len(clean_data)}) vs {noisy_file} ({len(noisy_data)})")
                min_len = min(len(clean_data), len(noisy_data))
                clean_data = clean_data[:min_len]
                noisy_data = noisy_data[:min_len]
            
            clean_signals.append(clean_data)
            noisy_signals.append(noisy_data)
            
            print(f"✅ 第 {i:02d} 组: {len(clean_data)} 样本")
        
        print(f"\n✅ 总共加载 {len(clean_signals)} 组数据")
        return clean_signals, noisy_signals
    
    def segment_signals(
        self,
        clean_signals: List[np.ndarray],
        noisy_signals: List[np.ndarray],
        segment_length: int = 2048,
        overlap_ratio: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将长信号切分为固定长度的片段，并进行混合
        
        Args:
            clean_signals: 干净信号列表
            noisy_signals: 带噪信号列表
            segment_length: 片段长度
            overlap_ratio: 重叠比例（0-1）
            
        Returns:
            clean_segments: (N, segment_length)
            noisy_segments: (N, segment_length)
        """
        clean_all = []
        noisy_all = []
        
        hop_length = int(segment_length * (1 - overlap_ratio))
        
        print(f"\n🔪 切分参数: segment_length={segment_length}, overlap={overlap_ratio}")
        
        for clean_sig, noisy_sig in zip(clean_signals, noisy_signals):
            num_segments = (len(clean_sig) - segment_length) // hop_length + 1
            
            for j in range(num_segments):
                start = j * hop_length
                end = start + segment_length
                
                if end > len(clean_sig):
                    break
                
                clean_all.append(clean_sig[start:end])
                noisy_all.append(noisy_sig[start:end])
        
        clean_segments = np.array(clean_all, dtype=np.float32)
        noisy_segments = np.array(noisy_all, dtype=np.float32)
        
        print(f"✅ 切分完成: {len(clean_segments)} 个片段")
        
        # 完全随机混合
        indices = np.arange(len(clean_segments))
        np.random.shuffle(indices)
        
        clean_segments = clean_segments[indices]
        noisy_segments = noisy_segments[indices]
        
        print(f"✅ 数据已完全混合（random_seed={self.random_seed}）")
        
        return clean_segments, noisy_segments
    
    def split_train_test(
        self,
        clean_segments: np.ndarray,
        noisy_segments: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        划分训练集与测试集
        
        Returns:
            clean_train, clean_test, noisy_train, noisy_test
        """
        clean_train, clean_test, noisy_train, noisy_test = train_test_split(
            clean_segments,
            noisy_segments,
            test_size=self.test_ratio,
            random_state=self.random_seed
        )
        
        print(f"\n📊 数据集划分:")
        print(f"   训练集: {len(clean_train)} 样本")
        print(f"   测试集: {len(clean_test)} 样本")
        print(f"   比例: {1-self.test_ratio:.0%} / {self.test_ratio:.0%}")
        
        return clean_train, clean_test, noisy_train, noisy_test
    
    def save_dataset(
        self,
        clean_train: np.ndarray,
        clean_test: np.ndarray,
        noisy_train: np.ndarray,
        noisy_test: np.ndarray,
        output_dir: str = "dataset_mixed"
    ):
        """
        保存划分好的数据集
        """
        output_path = Path(output_dir)
        
        # 创建目录结构
        for split in ['train', 'test']:
            for category in ['clean', 'noisy']:
                (output_path / split / category).mkdir(parents=True, exist_ok=True)
        
        # 保存为 .npy
        np.save(output_path / 'train' / 'clean' / 'data.npy', clean_train)
        np.save(output_path / 'train' / 'noisy' / 'data.npy', noisy_train)
        np.save(output_path / 'test' / 'clean' / 'data.npy', clean_test)
        np.save(output_path / 'test' / 'noisy' / 'data.npy', noisy_test)
        
        print(f"\n💾 数据已保存至: {output_path}")
        print(f"   train/clean/data.npy: {clean_train.shape}")
        print(f"   train/noisy/data.npy: {noisy_train.shape}")
        print(f"   test/clean/data.npy: {clean_test.shape}")
        print(f"   test/noisy/data.npy: {noisy_test.shape}")


class STFTDataset(Dataset):
    """
    STFT 数据集：实时计算 STFT 并返回频谱
    """
    def __init__(
        self,
        clean_data: np.ndarray,
        noisy_data: np.ndarray,
        n_fft: int = 256,
        hop_length: int = 64,
        mode: str = 'magnitude',  # 'magnitude' 或 'complex'
        normalize: bool = True
    ):
        """
        Args:
            clean_data: 干净信号 (N, signal_length)
            noisy_data: 带噪信号 (N, signal_length)
            n_fft: FFT 点数
            hop_length: 帧移
            mode: 'magnitude' (仅幅度谱) 或 'complex' (实部+虚部)
            normalize: 是否归一化
        """
        self.clean_data = clean_data
        self.noisy_data = noisy_data
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mode = mode
        self.normalize = normalize
        
        # 预计算归一化统计量（基于训练集）
        if normalize:
            sample_stft = self._compute_stft(noisy_data[0])
            self.mean = 0.0
            self.std = 1.0  # 将在第一次调用时计算
    
    def _compute_stft(self, signal: np.ndarray) -> np.ndarray:
        """
        计算 STFT
        
        Returns:
            如果 mode='magnitude': (freq, time)
            如果 mode='complex': (2, freq, time) [Re, Im]
        """
        # 使用 numpy FFT
        stft = np.array([
            np.fft.rfft(signal[i:i+self.n_fft])
            for i in range(0, len(signal) - self.n_fft + 1, self.hop_length)
        ]).T  # (freq, time)
        
        if self.mode == 'magnitude':
            return np.abs(stft)
        elif self.mode == 'complex':
            real = np.real(stft)
            imag = np.imag(stft)
            return np.stack([real, imag], axis=0)  # (2, freq, time)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def __len__(self):
        return len(self.clean_data)
    
    def __getitem__(self, idx):
        clean_sig = self.clean_data[idx]
        noisy_sig = self.noisy_data[idx]
        
        # 计算 STFT
        clean_stft = self._compute_stft(clean_sig)
        noisy_stft = self._compute_stft(noisy_sig)
        
        # 归一化（简单的 min-max 或 z-score）
        if self.normalize:
            # 使用 log-scale 归一化（常用于音频/EEG）
            if self.mode == 'magnitude':
                noisy_stft = np.log1p(noisy_stft)  # log(1 + x)
                clean_stft = np.log1p(clean_stft)
                
                # Z-score 归一化
                noisy_stft = (noisy_stft - noisy_stft.mean()) / (noisy_stft.std() + 1e-8)
                clean_stft = (clean_stft - clean_stft.mean()) / (clean_stft.std() + 1e-8)
            else:
                # 对实部虚部分别归一化
                for c in range(2):
                    noisy_stft[c] = (noisy_stft[c] - noisy_stft[c].mean()) / (noisy_stft[c].std() + 1e-8)
                    clean_stft[c] = (clean_stft[c] - clean_stft[c].mean()) / (clean_stft[c].std() + 1e-8)
        
        # 转为 Tensor 并添加 channel 维度
        if self.mode == 'magnitude':
            noisy_stft = torch.FloatTensor(noisy_stft).unsqueeze(0)  # (1, freq, time)
            clean_stft = torch.FloatTensor(clean_stft).unsqueeze(0)
        else:
            noisy_stft = torch.FloatTensor(noisy_stft)  # (2, freq, time)
            clean_stft = torch.FloatTensor(clean_stft)
        
        return noisy_stft, clean_stft


def create_dataloaders(
    clean_train: np.ndarray,
    clean_test: np.ndarray,
    noisy_train: np.ndarray,
    noisy_test: np.ndarray,
    batch_size: int = 16,
    n_fft: int = 256,
    hop_length: int = 64,
    mode: str = 'magnitude',
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试 DataLoader
    """
    train_dataset = STFTDataset(
        clean_train, noisy_train,
        n_fft=n_fft, hop_length=hop_length, mode=mode, normalize=True
    )
    
    test_dataset = STFTDataset(
        clean_test, noisy_test,
        n_fft=n_fft, hop_length=hop_length, mode=mode, normalize=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n🔄 DataLoader 已创建:")
    print(f"   训练批次: {len(train_loader)} (batch_size={batch_size})")
    print(f"   测试批次: {len(test_loader)}")
    
    return train_loader, test_loader


if __name__ == "__main__":
    # 测试数据准备流程
    prep = EEGDataPreparation(
        data_dir="2_Data_processed",
        num_files=10,
        test_ratio=0.2,
        random_seed=42
    )
    
    # 1. 加载数据
    clean_signals, noisy_signals = prep.load_all_data()
    
    # 2. 切分与混合
    clean_segments, noisy_segments = prep.segment_signals(
        clean_signals, noisy_signals,
        segment_length=2048,
        overlap_ratio=0.5
    )
    
    # 3. 划分训练/测试集
    clean_train, clean_test, noisy_train, noisy_test = prep.split_train_test(
        clean_segments, noisy_segments
    )
    
    # 4. 保存
    prep.save_dataset(clean_train, clean_test, noisy_train, noisy_test)
    
    # 5. 创建 DataLoader
    train_loader, test_loader = create_dataloaders(
        clean_train, clean_test, noisy_train, noisy_test,
        batch_size=16,
        n_fft=256,
        hop_length=64,
        mode='magnitude'
    )
    
    # 测试一个 batch
    noisy_batch, clean_batch = next(iter(train_loader))
    print(f"\n📦 Batch 形状:")
    print(f"   Noisy: {noisy_batch.shape}")
    print(f"   Clean: {clean_batch.shape}")
