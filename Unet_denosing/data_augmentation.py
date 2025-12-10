"""
数据增强策略：用于增加训练数据多样性
"""

import numpy as np
from typing import Tuple


class DataAugmentation:
    """
    EEG/信号数据增强类
    """
    
    @staticmethod
    def add_gaussian_noise(signal: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """
        添加高斯噪声
        
        Args:
            signal: 输入信号
            noise_level: 噪声水平（相对于信号标准差）
        """
        noise = np.random.randn(*signal.shape) * noise_level * np.std(signal)
        return signal + noise
    
    @staticmethod
    def time_shift(signal: np.ndarray, max_shift: int = 100) -> np.ndarray:
        """
        时间平移
        
        Args:
            signal: 输入信号
            max_shift: 最大平移量
        """
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(signal, shift)
    
    @staticmethod
    def amplitude_scale(signal: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        幅度缩放
        
        Args:
            signal: 输入信号
            scale_range: 缩放范围
        """
        scale = np.random.uniform(*scale_range)
        return signal * scale
    
    @staticmethod
    def band_reject(signal: np.ndarray, fs: int = 250, 
                    freq_range: Tuple[float, float] = (45, 55)) -> np.ndarray:
        """
        带阻滤波（模拟工频干扰去除）
        
        Args:
            signal: 输入信号
            fs: 采样率
            freq_range: 阻带频率范围
        """
        from scipy import signal as scipy_signal
        
        nyq = fs / 2
        low = freq_range[0] / nyq
        high = freq_range[1] / nyq
        
        b, a = scipy_signal.butter(4, [low, high], btype='bandstop')
        return scipy_signal.filtfilt(b, a, signal)
    
    @staticmethod
    def time_mask(signal: np.ndarray, num_masks: int = 2, mask_len: int = 50) -> np.ndarray:
        """
        时间域遮罩（类似 SpecAugment）
        
        Args:
            signal: 输入信号
            num_masks: 遮罩数量
            mask_len: 遮罩长度
        """
        masked_signal = signal.copy()
        
        for _ in range(num_masks):
            start = np.random.randint(0, len(signal) - mask_len)
            masked_signal[start:start+mask_len] = 0
        
        return masked_signal
    
    @staticmethod
    def spec_augment(stft: np.ndarray, 
                     freq_mask_param: int = 10, 
                     time_mask_param: int = 10,
                     num_freq_masks: int = 1,
                     num_time_masks: int = 1) -> np.ndarray:
        """
        频谱增强（SpecAugment）
        
        Args:
            stft: STFT 频谱 (freq, time)
            freq_mask_param: 频域遮罩最大宽度
            time_mask_param: 时域遮罩最大宽度
            num_freq_masks: 频域遮罩数量
            num_time_masks: 时域遮罩数量
        """
        aug_stft = stft.copy()
        freq_bins, time_frames = stft.shape
        
        # 频域遮罩
        for _ in range(num_freq_masks):
            f_width = np.random.randint(0, freq_mask_param)
            f_start = np.random.randint(0, freq_bins - f_width)
            aug_stft[f_start:f_start+f_width, :] = 0
        
        # 时域遮罩
        for _ in range(num_time_masks):
            t_width = np.random.randint(0, time_mask_param)
            t_start = np.random.randint(0, time_frames - t_width)
            aug_stft[:, t_start:t_start+t_width] = 0
        
        return aug_stft
    
    @staticmethod
    def mixup(signal1: np.ndarray, signal2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Mixup 增强
        
        Args:
            signal1: 信号1
            signal2: 信号2
            alpha: 混合系数
        """
        lam = np.random.beta(alpha, alpha)
        return lam * signal1 + (1 - lam) * signal2


if __name__ == "__main__":
    # 测试数据增强
    import matplotlib.pyplot as plt
    
    # 生成测试信号
    fs = 250
    t = np.linspace(0, 2, fs * 2)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
    
    aug = DataAugmentation()
    
    # 测试各种增强
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    axes[0, 0].plot(signal)
    axes[0, 0].set_title('Original Signal')
    axes[0, 0].grid(True, alpha=0.3)
    
    noisy_signal = aug.add_gaussian_noise(signal, noise_level=0.2)
    axes[0, 1].plot(noisy_signal)
    axes[0, 1].set_title('+ Gaussian Noise')
    axes[0, 1].grid(True, alpha=0.3)
    
    shifted_signal = aug.time_shift(signal, max_shift=50)
    axes[1, 0].plot(shifted_signal)
    axes[1, 0].set_title('Time Shift')
    axes[1, 0].grid(True, alpha=0.3)
    
    scaled_signal = aug.amplitude_scale(signal)
    axes[1, 1].plot(scaled_signal)
    axes[1, 1].set_title('Amplitude Scale')
    axes[1, 1].grid(True, alpha=0.3)
    
    masked_signal = aug.time_mask(signal, num_masks=2, mask_len=30)
    axes[2, 0].plot(masked_signal)
    axes[2, 0].set_title('Time Mask')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Mixup
    signal2 = np.random.randn(len(signal))
    mixed_signal = aug.mixup(signal, signal2, alpha=0.5)
    axes[2, 1].plot(mixed_signal)
    axes[2, 1].set_title('Mixup')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('augmentation_demo.png', dpi=150)
    plt.show()
    
    print("✅ 数据增强测试完成")
