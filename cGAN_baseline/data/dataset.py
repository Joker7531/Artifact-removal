"""
数据处理模块：按 section_id 划分 session，进行 STFT 变换，构建训练样本

核心功能：
1. 加载 NPZ 数据
2. 对信号进行 0.5-40Hz 带通滤波
3. 按 section_id 将连续数据划分为独立的 session
4. 对每个 session 进行 STFT 变换（实部+虚部双通道）
5. 滑动窗口构建训练样本（T 帧时频图）
6. 样本级别的 train/test 划分
7. 频域维度处理：切片到512或补零到528
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
        n_fft: int = 1024,
        hop_length: int = 250,
        window: str = 'hann',
        sample_rate: int = 250,  # EEG 采样率
        lowcut: float = 0.5,     # 带通滤波下限
        highcut: float = 40.0,   # 带通滤波上限
        freq_dim_mode: str = 'pad',  # 'pad' 补零到528, 'crop' 切到512
    ):
        """
        参数:
            npz_path: NPZ 文件路径
            n_fft: STFT 窗口大小
            hop_length: STFT 跳跃步长
            window: STFT 窗函数类型
            sample_rate: 采样率
            lowcut: 带通滤波下限频率（Hz）
            highcut: 带通滤波上限频率（Hz）
            freq_dim_mode: 频域维度处理方式 ('pad' 或 'crop')
        """
        self.npz_path = npz_path
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.sample_rate = sample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.freq_dim_mode = freq_dim_mode
        
        # 计算频域维度
        self.original_freq_bins = n_fft // 2 + 1  # 513 for n_fft=1024
        if freq_dim_mode == 'pad':
            self.processed_freq_bins = 528  # 补零到能被16整除
        elif freq_dim_mode == 'crop':
            self.processed_freq_bins = 512  # 切到能被16整除
        else:
            raise ValueError(f"不支持的 freq_dim_mode: {freq_dim_mode}")
        
        # 加载数据
        print(f"加载数据: {npz_path}")
        data = np.load(npz_path)
        self.raw = data['raw']
        self.clean = data['clean']
        self.section_id = data['section_id']
        
        print(f"数据长度: {len(self.raw)}")
        print(f"Section 数量: {len(np.unique(self.section_id))}")
        print(f"滤波范围: {lowcut}-{highcut} Hz")
        print(f"频域维度: {self.original_freq_bins} -> {self.processed_freq_bins} ({freq_dim_mode})")
        
    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        0.5-40Hz 带通滤波
        
        参数:
            data: 时域信号 (N,)
            
        返回:
            filtered_data: 滤波后的信号 (N,)
        """
        # 设计 Butterworth 带通滤波器（4阶）
        nyq = 0.5 * self.sample_rate
        low = self.lowcut / nyq
        high = self.highcut / nyq
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data)
        
        return filtered_data
        
    def split_by_section(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        按 section_id 划分数据为独立的 session，并进行滤波
        
        返回:
            sessions: {section_id: {'raw': array, 'clean': array}}
        """
        sessions = {}
        unique_sections = np.unique(self.section_id)
        
        for sec_id in unique_sections:
            mask = self.section_id == sec_id
            
            # 提取 session 数据并滤波
            raw_session = self.raw[mask]
            clean_session = self.clean[mask]
            
            # 0.5-40Hz 带通滤波
            raw_filtered = self.bandpass_filter(raw_session)
            clean_filtered = self.bandpass_filter(clean_session)
            
            sessions[int(sec_id)] = {
                'raw': raw_filtered,
                'clean': clean_filtered
            }
            print(f"Section {sec_id}: {np.sum(mask)} 样本点 (已滤波)")
        
        return sessions
    
    def compute_stft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 STFT 并返回实部和虚部
        
        参数:
            signal_data: 时域信号 (N,)
            
        返回:
            real: 实部 (freq_bins, time_frames)
            imag: 虚部 (freq_bins, time_frames)
        """
        f, t, Zxx = signal.stft(
            signal_data,
            fs=self.sample_rate,
            window=self.window,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length
        )
        
        real = np.real(Zxx)
        imag = np.imag(Zxx)
        
        return real, imag
    
    def process_freq_dimension(self, real: np.ndarray, imag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理频域维度：切片或补零
        
        参数:
            real: 实部 (original_freq_bins, time_frames)
            imag: 虚部 (original_freq_bins, time_frames)
            
        返回:
            real_processed: 处理后的实部 (processed_freq_bins, time_frames)
            imag_processed: 处理后的虚部 (processed_freq_bins, time_frames)
        """
        if self.freq_dim_mode == 'crop':
            # 切到 512
            real_processed = real[:self.processed_freq_bins, :]
            imag_processed = imag[:self.processed_freq_bins, :]
        else:  # pad
            # 补零到 528
            pad_size = self.processed_freq_bins - self.original_freq_bins
            real_processed = np.pad(real, ((0, pad_size), (0, 0)), mode='constant')
            imag_processed = np.pad(imag, ((0, pad_size), (0, 0)), mode='constant')
        
        return real_processed, imag_processed
    
    def build_samples(
        self,
        sessions: Dict[int, Dict[str, np.ndarray]],
        n_frames: int = 64,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从所有 session 构建滑动窗口样本（实部+虚部双通道）
        
        参数:
            sessions: 按 section_id 划分的 session 字典
            n_frames: 每个样本的时间帧数（T）
            stride: 滑动窗口步长
            
        返回:
            raw_samples: (N, 2, freq_bins, n_frames) - 通道0: 实部, 通道1: 虚部
            clean_samples: (N, 2, freq_bins, n_frames)
        """
        raw_samples_list = []
        clean_samples_list = []
        
        for sec_id, session_data in sessions.items():
            # 计算 STFT
            raw_real, raw_imag = self.compute_stft(session_data['raw'])
            clean_real, clean_imag = self.compute_stft(session_data['clean'])
            
            # 处理频域维度
            raw_real, raw_imag = self.process_freq_dimension(raw_real, raw_imag)
            clean_real, clean_imag = self.process_freq_dimension(clean_real, clean_imag)
            
            freq_bins, time_frames = raw_real.shape
            
            # 滑动窗口提取样本
            for i in range(0, time_frames - n_frames + 1, stride):
                # 构建双通道样本：[实部, 虚部]
                raw_patch = np.stack([
                    raw_real[:, i:i+n_frames],
                    raw_imag[:, i:i+n_frames]
                ], axis=0)  # (2, freq_bins, n_frames)
                
                clean_patch = np.stack([
                    clean_real[:, i:i+n_frames],
                    clean_imag[:, i:i+n_frames]
                ], axis=0)
                
                raw_samples_list.append(raw_patch)
                clean_samples_list.append(clean_patch)
            
            print(f"Section {sec_id}: 生成 {(time_frames - n_frames + 1) // stride} 个样本")
        
        raw_samples = np.stack(raw_samples_list, axis=0)  # (N, 2, freq_bins, n_frames)
        clean_samples = np.stack(clean_samples_list, axis=0)
        
        print(f"总样本数: {raw_samples.shape[0]}, 形状: {raw_samples.shape}")
        
        return raw_samples, clean_samples


class STFTPatchDataset(Dataset):
    """
    PyTorch Dataset：用于训练 cGAN 的 STFT patch 数据集（实部+虚部双通道）
    """
    
    def __init__(
        self,
        raw_samples: np.ndarray,
        clean_samples: np.ndarray,
        normalize: bool = True
    ):
        """
        参数:
            raw_samples: (N, 2, freq_bins, n_frames) - 通道0: 实部, 通道1: 虚部
            clean_samples: (N, 2, freq_bins, n_frames)
            normalize: 是否标准化
        """
        self.raw = raw_samples
        self.clean = clean_samples
        self.normalize = normalize
        
        if normalize:
            # 分别计算实部和虚部的统计量
            self.real_mean = np.mean(raw_samples[:, 0, :, :])
            self.real_std = np.std(raw_samples[:, 0, :, :])
            self.imag_mean = np.mean(raw_samples[:, 1, :, :])
            self.imag_std = np.std(raw_samples[:, 1, :, :])
            
            print(f"标准化参数 - 实部: mean={self.real_mean:.4f}, std={self.real_std:.4f}")
            print(f"标准化参数 - 虚部: mean={self.imag_mean:.4f}, std={self.imag_std:.4f}")
        
    def __len__(self):
        return len(self.raw)
    
    def __getitem__(self, idx):
        raw_patch = self.raw[idx].copy()  # (2, freq_bins, n_frames)
        clean_patch = self.clean[idx].copy()
        
        if self.normalize:
            # 分别标准化实部和虚部
            raw_patch[0] = (raw_patch[0] - self.real_mean) / (self.real_std + 1e-8)
            raw_patch[1] = (raw_patch[1] - self.imag_mean) / (self.imag_std + 1e-8)
            clean_patch[0] = (clean_patch[0] - self.real_mean) / (self.real_std + 1e-8)
            clean_patch[1] = (clean_patch[1] - self.imag_mean) / (self.imag_std + 1e-8)
        
        # 转为 Torch 张量
        raw_tensor = torch.FloatTensor(raw_patch)  # (2, freq_bins, n_frames)
        clean_tensor = torch.FloatTensor(clean_patch)
        
        return raw_tensor, clean_tensor


def prepare_data(
    npz_path: str,
    n_fft: int = 1024,
    hop_length: int = 250,
    n_frames: int = 64,
    stride: int = 8,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    freq_dim_mode: str = 'pad',
    lowcut: float = 0.5,
    highcut: float = 40.0
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
        freq_dim_mode: 频域维度处理方式 ('pad' 或 'crop')
        lowcut: 带通滤波下限（Hz）
        highcut: 带通滤波上限（Hz）
        
    返回:
        train_dataset: 训练集 Dataset
        test_dataset: 测试集 Dataset
    """
    # 1. 初始化数据处理器
    processor = EEGDataProcessor(
        npz_path=npz_path,
        n_fft=n_fft,
        hop_length=hop_length,
        lowcut=lowcut,
        highcut=highcut,
        freq_dim_mode=freq_dim_mode
    )
    
    # 2. 按 section 划分并滤波
    sessions = processor.split_by_section()
    
    # 3. 构建样本（实部+虚部双通道）
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
    
    # 共享标准化参数
    test_dataset.real_mean = train_dataset.real_mean
    test_dataset.real_std = train_dataset.real_std
    test_dataset.imag_mean = train_dataset.imag_mean
    test_dataset.imag_std = train_dataset.imag_std
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    # 测试数据处理流程
    train_ds, test_ds = prepare_data(
        npz_path="../../dataset_cz_v2.npz",
        n_fft=1024,
        hop_length=250,
        n_frames=64,
        stride=8,
        train_ratio=0.8,
        freq_dim_mode='pad'
    )
    
    print(f"\n样本形状测试:")
    raw, clean = train_ds[0]
    print(f"Raw: {raw.shape}")  # (2, 528, 64)
    print(f"Clean: {clean.shape}")
