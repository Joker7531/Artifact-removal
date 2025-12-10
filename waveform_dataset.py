"""
EEG Waveform Dataset for GAN-based Denoising
直接在时域进行波形重建,无需 STFT 转换
包含数据增强防止过拟合
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataAugmentation:
    """EEG 波形数据增强"""
    
    def __init__(self, 
                 noise_std: float = 0.01,
                 scale_range: Tuple[float, float] = (0.95, 1.05),
                 shift_range: float = 0.05):
        """
        Args:
            noise_std: 添加高斯噪声的标准差
            scale_range: 幅度缩放范围
            shift_range: 时间平移范围(相对于窗口长度)
        """
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_range = shift_range
    
    def add_noise(self, wave: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_std, wave.shape)
        return wave + noise
    
    def scale_amplitude(self, wave: np.ndarray) -> np.ndarray:
        """随机缩放幅度"""
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return wave * scale
    
    def time_shift(self, wave: np.ndarray) -> np.ndarray:
        """时间平移"""
        max_shift = int(len(wave) * self.shift_range)
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift > 0:
            return np.pad(wave[:-shift], (shift, 0), mode='edge')
        elif shift < 0:
            return np.pad(wave[-shift:], (0, -shift), mode='edge')
        return wave
    
    def __call__(self, wave: np.ndarray, prob: float = 0.5) -> np.ndarray:
        """
        应用数据增强
        Args:
            wave: 输入波形
            prob: 每种增强的应用概率
        """
        # 随机应用各种增强
        if np.random.rand() < prob:
            wave = self.add_noise(wave)
        if np.random.rand() < prob:
            wave = self.scale_amplitude(wave)
        if np.random.rand() < prob:
            wave = self.time_shift(wave)
        return wave


class WaveformProcessor:
    """时域波形预处理器"""
    
    def __init__(self, 
                 fs: int = 100,
                 window_sec: float = 3.0,
                 overlap: float = 0.75):
        """
        Args:
            fs: 采样率 (Hz)
            window_sec: 窗口长度 (秒)
            overlap: 重叠比例 (0-1)
        """
        self.fs = fs
        self.window_length = int(window_sec * fs)
        self.overlap = overlap
        self.step = int(self.window_length * (1 - overlap))
    
    def sliding_window(self, data: np.ndarray) -> List[np.ndarray]:
        """
        对 EEG 数据进行滑动窗口切片
        
        Args:
            data: 1D EEG 信号数组
            
        Returns:
            窗口列表
        """
        windows = []
        start = 0
        while start + self.window_length <= len(data):
            windows.append(data[start:start + self.window_length])
            start += self.step
        return windows
    
    def normalize(self, data: np.ndarray, method='zscore', eps: float = 1e-8) -> np.ndarray:
        """
        归一化波形
        
        Args:
            data: 输入数据
            method: 归一化方法 ('zscore', 'minmax', 'robust')
            eps: 防止除零的小常数
            
        Returns:
            归一化后的数据
        """
        if method == 'zscore':
            # Z-score 归一化
            mean = data.mean()
            std = data.std()
            return (data - mean) / (std + eps)
        
        elif method == 'minmax':
            # Min-Max 归一化到 [-1, 1]
            min_val = data.min()
            max_val = data.max()
            if max_val - min_val < eps:
                return np.zeros_like(data)
            return 2 * (data - min_val) / (max_val - min_val + eps) - 1
        
        elif method == 'robust':
            # Robust 归一化 (使用中位数和 IQR)
            median = np.median(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            return (data - median) / (iqr + eps)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")


class EEGWaveformDataset(Dataset):
    """
    EEG Waveform Dataset for GAN
    输入: raw waveform (1D 时序信号)
    标签: clean waveform (1D 时序信号)
    支持数据增强
    """
    
    def __init__(self, 
                 data_dir: str,
                 file_indices: List[int],
                 processor: WaveformProcessor,
                 normalize_method: str = 'zscore',
                 augmentation: DataAugmentation = None,
                 is_train: bool = True):
        """
        Args:
            data_dir: 数据目录路径
            file_indices: 文件索引列表
            processor: 波形处理器实例
            normalize_method: 归一化方法
            augmentation: 数据增强器 (仅训练时使用)
            is_train: 是否为训练模式
        """
        self.data_dir = Path(data_dir)
        self.file_indices = file_indices
        self.processor = processor
        self.normalize_method = normalize_method
        self.augmentation = augmentation if is_train else None
        self.is_train = is_train
        
        # 加载并处理所有数据
        self.samples = []
        self._load_data()
        
    def _load_data(self):
        """加载所有成对的 raw 和 clean 数据"""
        print(f"Loading waveform data from {len(self.file_indices)} file pairs...")
        
        for idx in self.file_indices:
            raw_file = self.data_dir / f"raw_Cz_{idx:02d}.csv"
            clean_file = self.data_dir / f"clean_Cz_{idx:02d}.csv"
            
            if not raw_file.exists() or not clean_file.exists():
                print(f"Warning: Missing file pair {idx:02d}, skipping...")
                continue
            
            # 读取数据
            raw_data = pd.read_csv(raw_file).values.flatten()
            clean_data = pd.read_csv(clean_file).values.flatten()
            
            # 确保长度一致
            min_len = min(len(raw_data), len(clean_data))
            raw_data = raw_data[:min_len]
            clean_data = clean_data[:min_len]
            
            # 滑动窗口切片
            raw_windows = self.processor.sliding_window(raw_data)
            clean_windows = self.processor.sliding_window(clean_data)
            
            # 处理每个窗口对
            for raw_win, clean_win in zip(raw_windows, clean_windows):
                # 归一化
                raw_norm = self.processor.normalize(raw_win, method=self.normalize_method)
                clean_norm = self.processor.normalize(clean_win, method=self.normalize_method)
                
                self.samples.append({
                    'raw': raw_norm.astype(np.float32),
                    'clean': clean_norm.astype(np.float32)
                })
        
        print(f"Loaded {len(self.samples)} waveform samples from {len(self.file_indices)} files")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        raw_wave = sample['raw'].copy()
        clean_wave = sample['clean'].copy()
        
        # 训练时应用数据增强(仅对raw增强，clean保持不变)
        if self.augmentation is not None and self.is_train:
            raw_wave = self.augmentation(raw_wave, prob=0.5)
        
        # 确保数据类型为 float32
        raw_wave = raw_wave.astype(np.float32)
        clean_wave = clean_wave.astype(np.float32)
        
        # 添加通道维度: (L,) -> (1, L)
        raw_tensor = torch.from_numpy(raw_wave).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_wave).unsqueeze(0)
        
        return {
            'raw': raw_tensor,      # (1, L)
            'clean': clean_tensor   # (1, L)
        }


def create_waveform_dataloaders(
    data_dir: str,
    train_indices: List[int],
    val_indices: List[int],
    batch_size: int = 32,
    num_workers: int = 4,
    use_augmentation: bool = True,
    **processor_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证 DataLoader
    
    Args:
        data_dir: 数据目录
        train_indices: 训练集文件索引
        val_indices: 验证集文件索引
        batch_size: batch 大小
        num_workers: 数据加载线程数
        use_augmentation: 是否使用数据增强
        **processor_kwargs: 波形处理器参数
        
    Returns:
        train_loader, val_loader
    """
    # 创建波形处理器
    processor = WaveformProcessor(**processor_kwargs)
    
    # 创建数据增强器
    augmentation = DataAugmentation(
        noise_std=0.01,
        scale_range=(0.95, 1.05),
        shift_range=0.05
    ) if use_augmentation else None
    
    # 创建数据集
    train_dataset = EEGWaveformDataset(
        data_dir=data_dir,
        file_indices=train_indices,
        processor=processor,
        normalize_method='zscore',
        augmentation=augmentation,
        is_train=True
    )
    
    val_dataset = EEGWaveformDataset(
        data_dir=data_dir,
        file_indices=val_indices,
        processor=processor,
        normalize_method='zscore',
        augmentation=None,
        is_train=False
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试代码
    data_dir = "../2_Data_processed"
    
    # 划分训练集和验证集
    train_indices = list(range(1, 9))  # 01-08
    val_indices = [9, 10]              # 09-10
    
    # 创建 DataLoader
    train_loader, val_loader = create_waveform_dataloaders(
        data_dir=data_dir,
        train_indices=train_indices,
        val_indices=val_indices,
        batch_size=32,
        num_workers=0,  # Windows 上建议设为 0
        fs=100,
        window_sec=3.0,
        overlap=0.75
    )
    
    # 测试加载
    print("\n=== Waveform Dataset Test ===")
    for batch in train_loader:
        print(f"Raw shape: {batch['raw'].shape}")      # (B, 1, L)
        print(f"Clean shape: {batch['clean'].shape}")  # (B, 1, L)
        print(f"Raw range: [{batch['raw'].min():.3f}, {batch['raw'].max():.3f}]")
        print(f"Clean range: [{batch['clean'].min():.3f}, {batch['clean'].max():.3f}]")
        break
