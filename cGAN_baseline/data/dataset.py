"""
数据处理模块：按 section_id 划分 session，进行 STFT 变换，构建训练样本

核心功能：
1. 加载 NPZ 数据
2. 按 section_id 将连续数据划分为独立的 session
3. 对每个 session 进行 STFT 变换（仅使用幅度谱）
4. 滑动窗口构建训练样本（T 帧时频图）
5. 样本级别的 train/test 划分
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal
from typing import Tuple, List, Dict


class EEGDataProcessor:
    """
    EEG 数据预处理器：负责将原始 NPZ 数据转换为 STFT 域样本
    """
    
    def __init__(
        self,
        npz_path: str,
        n_fft: int = 256,
        hop_length: int = 128,
        window: str = 'hann',
        sample_rate: int = 250,  # EEG 采样率通常为 250 Hz
    ):
        """
        参数:
            npz_path: NPZ 文件路径
            n_fft: STFT 窗口大小
            hop_length: STFT 跳跃步长
            window: STFT 窗函数类型
            sample_rate: 采样率
        """
        self.npz_path = npz_path
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.sample_rate = sample_rate
        
        # 加载数据
        print(f"加载数据: {npz_path}")
        data = np.load(npz_path)
        self.raw = data['raw']
        self.clean = data['clean']
        self.section_id = data['section_id']
        
        print(f"数据长度: {len(self.raw)}")
        print(f"Section 数量: {len(np.unique(self.section_id))}")
        
    def split_by_section(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        按 section_id 划分数据为独立的 session
        
        返回:
            sessions: {section_id: {'raw': array, 'clean': array}}
        """
        sessions = {}
        unique_sections = np.unique(self.section_id)
        
        for sec_id in unique_sections:
            mask = self.section_id == sec_id
            sessions[int(sec_id)] = {
                'raw': self.raw[mask],
                'clean': self.clean[mask]
            }
            print(f"Section {sec_id}: {np.sum(mask)} 样本点")
        
        return sessions
    
    def compute_stft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 STFT 并返回幅度谱和相位谱
        
        参数:
            signal_data: 时域信号 (N,)
            
        返回:
            magnitude: 幅度谱 (freq_bins, time_frames)
            phase: 相位谱 (freq_bins, time_frames)
        """
        f, t, Zxx = signal.stft(
            signal_data,
            fs=self.sample_rate,
            window=self.window,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length
        )
        
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        return magnitude, phase
    
    def build_samples(
        self,
        sessions: Dict[int, Dict[str, np.ndarray]],
        n_frames: int = 32,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从所有 session 构建滑动窗口样本
        
        参数:
            sessions: 按 section_id 划分的 session 字典
            n_frames: 每个样本的时间帧数（T）
            stride: 滑动窗口步长
            
        返回:
            raw_samples: (N, freq_bins, n_frames)
            clean_samples: (N, freq_bins, n_frames)
        """
        raw_samples_list = []
        clean_samples_list = []
        
        for sec_id, session_data in sessions.items():
            # 计算 STFT
            raw_mag, _ = self.compute_stft(session_data['raw'])
            clean_mag, _ = self.compute_stft(session_data['clean'])
            
            freq_bins, time_frames = raw_mag.shape
            
            # 滑动窗口提取样本
            for i in range(0, time_frames - n_frames + 1, stride):
                raw_patch = raw_mag[:, i:i+n_frames]
                clean_patch = clean_mag[:, i:i+n_frames]
                
                raw_samples_list.append(raw_patch)
                clean_samples_list.append(clean_patch)
            
            print(f"Section {sec_id}: 生成 {(time_frames - n_frames + 1) // stride} 个样本")
        
        raw_samples = np.stack(raw_samples_list, axis=0)  # (N, freq_bins, n_frames)
        clean_samples = np.stack(clean_samples_list, axis=0)
        
        print(f"总样本数: {raw_samples.shape[0]}, 形状: {raw_samples.shape}")
        
        return raw_samples, clean_samples


class STFTPatchDataset(Dataset):
    """
    PyTorch Dataset：用于训练 cGAN 的 STFT patch 数据集
    """
    
    def __init__(
        self,
        raw_samples: np.ndarray,
        clean_samples: np.ndarray,
        normalize: bool = True
    ):
        """
        参数:
            raw_samples: (N, freq_bins, n_frames)
            clean_samples: (N, freq_bins, n_frames)
            normalize: 是否归一化到 [0, 1]
        """
        self.raw = raw_samples
        self.clean = clean_samples
        self.normalize = normalize
        
        if normalize:
            # 计算全局最大值用于归一化
            self.max_val = max(np.max(self.raw), np.max(self.clean))
            print(f"归一化最大值: {self.max_val:.4f}")
        
    def __len__(self):
        return len(self.raw)
    
    def __getitem__(self, idx):
        raw_patch = self.raw[idx]
        clean_patch = self.clean[idx]
        
        if self.normalize:
            raw_patch = raw_patch / self.max_val
            clean_patch = clean_patch / self.max_val
        
        # 转为 Torch 张量，并添加 channel 维度
        raw_tensor = torch.FloatTensor(raw_patch).unsqueeze(0)  # (1, freq_bins, n_frames)
        clean_tensor = torch.FloatTensor(clean_patch).unsqueeze(0)
        
        return raw_tensor, clean_tensor


def prepare_data(
    npz_path: str,
    n_fft: int = 256,
    hop_length: int = 128,
    n_frames: int = 32,
    stride: int = 8,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[STFTPatchDataset, STFTPatchDataset]:
    """
    完整的数据准备流程
    
    参数:
        npz_path: NPZ 文件路径
        n_fft: STFT 窗口大小
        hop_length: STFT 跳跃步长
        n_frames: 每个样本的时间帧数
        stride: 滑动窗口步长
        train_ratio: 训练集比例
        random_seed: 随机种子
        
    返回:
        train_dataset: 训练集 Dataset
        test_dataset: 测试集 Dataset
    """
    # 1. 初始化数据处理器
    processor = EEGDataProcessor(
        npz_path=npz_path,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # 2. 按 section 划分
    sessions = processor.split_by_section()
    
    # 3. 构建样本
    raw_samples, clean_samples = processor.build_samples(
        sessions=sessions,
        n_frames=n_frames,
        stride=stride
    )
    
    # 4. 划分训练集和测试集
    np.random.seed(random_seed)
    n_samples = len(raw_samples)
    indices = np.random.permutation(n_samples)
    
    split_idx = int(n_samples * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    print(f"\n数据集划分:")
    print(f"训练集: {len(train_indices)} 样本")
    print(f"测试集: {len(test_indices)} 样本")
    
    # 5. 创建 Dataset
    train_dataset = STFTPatchDataset(
        raw_samples[train_indices],
        clean_samples[train_indices],
        normalize=True
    )
    
    test_dataset = STFTPatchDataset(
        raw_samples[test_indices],
        clean_samples[test_indices],
        normalize=True
    )
    
    # 共享归一化参数
    test_dataset.max_val = train_dataset.max_val
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    # 测试数据处理流程
    train_ds, test_ds = prepare_data(
        npz_path="../dataset_cz_v2.npz",
        n_fft=256,
        hop_length=128,
        n_frames=32,
        stride=8,
        train_ratio=0.8
    )
    
    print(f"\n样本形状测试:")
    raw, clean = train_ds[0]
    print(f"Raw: {raw.shape}")
    print(f"Clean: {clean.shape}")
