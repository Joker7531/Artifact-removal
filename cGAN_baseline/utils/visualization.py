"""
评估与可视化模块

核心功能：
1. 重建信号（STFT -> iSTFT）
2. 计算评估指标（MSE, SNR, etc.）
3. 可视化对比（时域波形、幅度谱、PSD）
4. 保存结果图表
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, List
import matplotlib
matplotlib.use('Agg')  # 非交互式后端


class SignalReconstructor:
    """
    信号重建器：从 STFT 幅度谱重建时域信号
    """
    
    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 128,
        window: str = 'hann',
        sample_rate: int = 500
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.sample_rate = sample_rate
    
    def reconstruct(
        self,
        magnitude: np.ndarray,
        phase: np.ndarray = None,
        length: int = None
    ) -> np.ndarray:
        """
        从 STFT 幅度谱重建时域信号
        
        参数:
            magnitude: 幅度谱 (freq_bins, time_frames)
            phase: 相位谱 (freq_bins, time_frames)，如果为 None 则使用零相位
            length: 输出信号长度
            
        返回:
            reconstructed: 重建的时域信号
        """
        if phase is None:
            # 使用零相位（简化版本）
            phase = np.zeros_like(magnitude)
        
        # 构建复数 STFT
        Zxx = magnitude * np.exp(1j * phase)
        
        # iSTFT
        _, reconstructed = signal.istft(
            Zxx,
            fs=self.sample_rate,
            window=self.window,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length
        )
        
        if length is not None:
            reconstructed = reconstructed[:length]
        
        return reconstructed


class EvaluationMetrics:
    """
    评估指标计算
    """
    
    @staticmethod
    def mse(pred, target):
        """均方误差"""
        return np.mean((pred - target) ** 2)
    
    @staticmethod
    def mae(pred, target):
        """平均绝对误差"""
        return np.mean(np.abs(pred - target))
    
    @staticmethod
    def snr(clean, noisy):
        """信噪比 (dB)"""
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean((clean - noisy) ** 2)
        
        if noise_power < 1e-10:
            return 100.0  # 接近完美
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db
    
    @staticmethod
    def compute_psd(signal_data, fs=500, nperseg=256):
        """计算功率谱密度"""
        f, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg)
        return f, psd


class Visualizer:
    """
    可视化工具
    """
    
    def __init__(self, save_dir='./results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_comparison(
        self,
        raw_signal: np.ndarray,
        clean_signal: np.ndarray,
        reconstructed_signal: np.ndarray,
        sample_rate: int = 500,
        save_name: str = 'comparison.png'
    ):
        """
        对比三种信号：raw, clean, reconstructed
        
        参数:
            raw_signal: 原始信号
            clean_signal: 干净信号
            reconstructed_signal: 重建信号
            sample_rate: 采样率
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        time = np.arange(len(raw_signal)) / sample_rate
        
        # 时域对比
        axes[0].plot(time, raw_signal, label='Raw', alpha=0.7)
        axes[0].plot(time, clean_signal, label='Clean (GT)', alpha=0.7)
        axes[0].plot(time, reconstructed_signal, label='Reconstructed', alpha=0.7, linestyle='--')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Time Domain Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 局部放大（前 2 秒）
        zoom_samples = min(2 * sample_rate, len(raw_signal))
        time_zoom = time[:zoom_samples]
        axes[1].plot(time_zoom, raw_signal[:zoom_samples], label='Raw', alpha=0.7)
        axes[1].plot(time_zoom, clean_signal[:zoom_samples], label='Clean (GT)', alpha=0.7)
        axes[1].plot(time_zoom, reconstructed_signal[:zoom_samples], label='Reconstructed', alpha=0.7, linestyle='--')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title('Time Domain Comparison (Zoom: first 2s)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 误差
        error = reconstructed_signal - clean_signal
        axes[2].plot(time, error, color='red', alpha=0.7)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Error')
        axes[2].set_title('Reconstruction Error')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_spectrograms(
        self,
        raw_mag: np.ndarray,
        clean_mag: np.ndarray,
        reconstructed_mag: np.ndarray,
        sample_rate: int = 500,
        hop_length: int = 128,
        save_name: str = 'spectrograms.png'
    ):
        """
        对比 STFT 幅度谱（使用统一的颜色映射范围）
        
        参数:
            raw_mag: Raw STFT 幅度谱 (freq_bins, time_frames)
            clean_mag: Clean STFT 幅度谱
            reconstructed_mag: Reconstructed STFT 幅度谱
            sample_rate: 采样率
            hop_length: STFT 跳跃步长
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 时间和频率轴
        time_frames = raw_mag.shape[1]
        freq_bins = raw_mag.shape[0]
        time = np.arange(time_frames) * hop_length / sample_rate
        freq = np.linspace(0.5, 40, freq_bins)
        
        # 转换为dB
        raw_db = 20 * np.log10(raw_mag + 1e-8)
        clean_db = 20 * np.log10(clean_mag + 1e-8)
        reconstructed_db = 20 * np.log10(reconstructed_mag + 1e-8)
        
        # 计算统一的颜色映射范围（基于三个图的最小值和最大值）
        vmin = min(raw_db.min(), clean_db.min(), reconstructed_db.min())
        vmax = max(raw_db.max(), clean_db.max(), reconstructed_db.max())
        
        # Raw
        im0 = axes[0].imshow(
            raw_db,
            aspect='auto',
            origin='lower',
            extent=[time[0], time[-1], freq[0], freq[-1]],
            cmap='viridis',
            vmin=vmin,
            vmax=vmax
        )
        axes[0].set_title('Raw STFT Magnitude (dB)')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Frequency (Hz)')
        plt.colorbar(im0, ax=axes[0])
        
        # Clean
        im1 = axes[1].imshow(
            clean_db,
            aspect='auto',
            origin='lower',
            extent=[time[0], time[-1], freq[0], freq[-1]],
            cmap='viridis',
            vmin=vmin,
            vmax=vmax
        )
        axes[1].set_title('Clean STFT Magnitude (dB)')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Frequency (Hz)')
        plt.colorbar(im1, ax=axes[1])
        
        # Reconstructed
        im2 = axes[2].imshow(
            reconstructed_db,
            aspect='auto',
            origin='lower',
            extent=[time[0], time[-1], freq[0], freq[-1]],
            cmap='viridis',
            vmin=vmin,
            vmax=vmax
        )
        axes[2].set_title('Reconstructed STFT Magnitude (dB)')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Frequency (Hz)')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_psd(
        self,
        raw_signal: np.ndarray,
        clean_signal: np.ndarray,
        reconstructed_signal: np.ndarray,
        sample_rate: int = 500,
        save_name: str = 'psd.png'
    ):
        """
        对比功率谱密度
        
        参数:
            raw_signal: 原始信号
            clean_signal: 干净信号
            reconstructed_signal: 重建信号
            sample_rate: 采样率
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # 计算 PSD
        f_raw, psd_raw = EvaluationMetrics.compute_psd(raw_signal, fs=sample_rate)
        f_clean, psd_clean = EvaluationMetrics.compute_psd(clean_signal, fs=sample_rate)
        f_recon, psd_recon = EvaluationMetrics.compute_psd(reconstructed_signal, fs=sample_rate)
        
        # 绘制
        ax.semilogy(f_raw, psd_raw, label='Raw', alpha=0.7)
        ax.semilogy(f_clean, psd_clean, label='Clean (GT)', alpha=0.7)
        ax.semilogy(f_recon, psd_recon, label='Reconstructed', alpha=0.7, linestyle='--')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (V^2/Hz)')
        ax.set_title('Power Spectral Density Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(self, history: dict, save_name: str = 'training_history.png'):
        """
        绘制训练历史曲线
        
        参数:
            history: 训练历史字典
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['g_loss']) + 1)
        
        # Generator Loss
        axes[0, 0].plot(epochs, history['g_loss'], label='G Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Generator Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Discriminator Loss
        axes[0, 1].plot(epochs, history['d_loss'], label='D Loss', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Generator Components
        axes[1, 0].plot(epochs, history['g_complex_loss'], label='G Complex Loss')
        axes[1, 0].plot(epochs, history['g_mag_loss'], label='G Magnitude Loss')
        axes[1, 0].plot(epochs, history['g_gan_loss'], label='G GAN Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Generator Loss Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Discriminator Components
        axes[1, 1].plot(epochs, history['d_real_loss'], label='D Real Loss')
        axes[1, 1].plot(epochs, history['d_fake_loss'], label='D Fake Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Discriminator Loss Components')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    print("评估与可视化模块")
    print("测试功能...")
    
    # 测试信号重建
    reconstructor = SignalReconstructor()
    test_signal = np.random.randn(1000)
    
    # STFT
    f, t, Zxx = signal.stft(test_signal, fs=250, nperseg=256, noverlap=128)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # iSTFT
    reconstructed = reconstructor.reconstruct(magnitude, phase)
    
    print(f"原始信号长度: {len(test_signal)}")
    print(f"重建信号长度: {len(reconstructed)}")
    print(f"重建误差 (MSE): {EvaluationMetrics.mse(test_signal[:len(reconstructed)], reconstructed):.6f}")
