"""
U-Net 推理与可视化脚本
支持单文件推理、批量推理和结果可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from scipy import signal as scipy_signal

from unet_model import build_unet
from data_preparation import STFTDataset


class UNetInference:
    """
    U-Net 推理类
    """
    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        device: str = 'cuda'
    ):
        """
        Args:
            model_path: 模型权重路径
            config_path: 配置文件路径（可选）
            device: 设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        if config_path is None:
            config_path = Path(model_path).parent / 'config.json'
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            print(f"⚠️  未找到配置文件，使用默认参数")
            self.config = {
                'mode': 'magnitude',
                'base_channels': 64,
                'depth': 4,
                'n_fft': 256,
                'hop_length': 64
            }
        
        # 构建模型
        self.model = build_unet(
            mode=self.config['mode'],
            base_channels=self.config['base_channels'],
            depth=self.config['depth'],
            device=self.device
        )
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        print(f"✅ 模型已加载: {model_path}")
        print(f"   模式: {self.config['mode']}")
        print(f"   设备: {self.device}")
    
    def _signal_to_stft(self, signal: np.ndarray) -> torch.Tensor:
        """
        信号转 STFT
        
        Args:
            signal: (signal_length,)
        
        Returns:
            stft_tensor: (1, channels, freq, time)
        """
        n_fft = self.config['n_fft']
        hop_length = self.config['hop_length']
        
        # 计算 STFT
        stft = np.array([
            np.fft.rfft(signal[i:i+n_fft])
            for i in range(0, len(signal) - n_fft + 1, hop_length)
        ]).T  # (freq, time)
        
        if self.config['mode'] == 'magnitude':
            stft_mag = np.abs(stft)
            
            # 归一化（与训练时一致）
            stft_mag = np.log1p(stft_mag)
            stft_mag = (stft_mag - stft_mag.mean()) / (stft_mag.std() + 1e-8)
            
            stft_tensor = torch.FloatTensor(stft_mag).unsqueeze(0).unsqueeze(0)  # (1, 1, freq, time)
            phase = np.angle(stft)  # 保存相位用于重建
            
            return stft_tensor, phase
        
        elif self.config['mode'] == 'complex':
            real = np.real(stft)
            imag = np.imag(stft)
            
            # 归一化
            for arr in [real, imag]:
                arr = (arr - arr.mean()) / (arr.std() + 1e-8)
            
            stft_tensor = torch.FloatTensor(np.stack([real, imag], axis=0)).unsqueeze(0)  # (1, 2, freq, time)
            
            return stft_tensor, None
    
    def _stft_to_signal(self, stft_tensor: torch.Tensor, phase: np.ndarray = None) -> np.ndarray:
        """
        STFT 转信号
        
        Args:
            stft_tensor: (1, channels, freq, time)
            phase: 相位（仅 magnitude 模式需要）
        
        Returns:
            signal: (signal_length,)
        """
        stft_np = stft_tensor.squeeze(0).cpu().numpy()
        
        n_fft = self.config['n_fft']
        hop_length = self.config['hop_length']
        
        if self.config['mode'] == 'magnitude':
            # 反归一化（需要保存训练时的统计量，这里简化处理）
            # stft_mag = stft_np[0]  # (freq, time)
            # stft_mag = np.expm1(stft_mag)  # 反 log1p
            
            # 使用相位重建复数谱
            if phase is None:
                raise ValueError("Magnitude 模式需要提供相位")
            
            stft_mag = stft_np[0]
            stft_mag = np.expm1(stft_mag)
            stft_complex = stft_mag * np.exp(1j * phase)
        
        elif self.config['mode'] == 'complex':
            real = stft_np[0]
            imag = stft_np[1]
            stft_complex = real + 1j * imag
        
        # iSTFT（重叠相加法）
        freq_bins, time_frames = stft_complex.shape
        signal_length = (time_frames - 1) * hop_length + n_fft
        
        signal = np.zeros(signal_length)
        window_sum = np.zeros(signal_length)
        
        for t in range(time_frames):
            frame = np.fft.irfft(stft_complex[:, t], n=n_fft)
            start = t * hop_length
            signal[start:start+n_fft] += frame
            window_sum[start:start+n_fft] += 1
        
        # 归一化（避免除零）
        signal = np.divide(signal, window_sum, where=window_sum > 0)
        
        return signal
    
    @torch.no_grad()
    def denoise_signal(self, noisy_signal: np.ndarray) -> np.ndarray:
        """
        对单个信号进行降噪
        
        Args:
            noisy_signal: (signal_length,)
        
        Returns:
            clean_signal: (signal_length,)
        """
        # 转 STFT
        stft_tensor, phase = self._signal_to_stft(noisy_signal)
        stft_tensor = stft_tensor.to(self.device)
        
        # 推理
        pred_stft = self.model(stft_tensor)
        
        # 转回信号
        clean_signal = self._stft_to_signal(pred_stft, phase)
        
        # 截取到原始长度
        clean_signal = clean_signal[:len(noisy_signal)]
        
        return clean_signal
    
    def denoise_batch(self, noisy_signals: np.ndarray) -> np.ndarray:
        """
        批量降噪
        
        Args:
            noisy_signals: (N, signal_length)
        
        Returns:
            clean_signals: (N, signal_length)
        """
        clean_signals = []
        
        for i, noisy_sig in enumerate(noisy_signals):
            clean_sig = self.denoise_signal(noisy_sig)
            clean_signals.append(clean_sig)
            
            if (i + 1) % 10 == 0:
                print(f"   进度: {i+1}/{len(noisy_signals)}")
        
        return np.array(clean_signals)


def visualize_results(
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    denoised_signal: np.ndarray,
    save_path: str = None,
    n_fft: int = 256,
    hop_length: int = 64,
    fs: int = 250
):
    """
    可视化降噪结果
    
    Args:
        noisy_signal: 带噪信号
        clean_signal: 真实干净信号（可选）
        denoised_signal: 降噪后信号
        save_path: 保存路径
        n_fft: FFT 点数
        hop_length: 帧移
        fs: 采样率
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    # 时间轴
    time = np.arange(len(noisy_signal)) / fs
    
    # 计算 STFT（用于可视化）
    def compute_stft_vis(sig):
        f, t, Zxx = scipy_signal.stft(sig, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length)
        return f, t, np.abs(Zxx)
    
    # === 第一行：时域波形 ===
    axes[0, 0].plot(time, noisy_signal, label='Noisy', alpha=0.7)
    axes[0, 0].set_title('Noisy Signal (Time Domain)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    if clean_signal is not None:
        axes[0, 1].plot(time, clean_signal, label='Clean (Ground Truth)', color='green', alpha=0.7)
        axes[0, 1].set_title('Clean Signal (Time Domain)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(time, denoised_signal, label='Denoised', color='red', alpha=0.7)
    axes[0, 2].set_title('Denoised Signal (Time Domain)')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Amplitude')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # === 第二行：频谱图 ===
    f_noisy, t_noisy, Zxx_noisy = compute_stft_vis(noisy_signal)
    axes[1, 0].pcolormesh(t_noisy, f_noisy, 20 * np.log10(Zxx_noisy + 1e-8), shading='gouraud', cmap='viridis')
    axes[1, 0].set_title('Noisy STFT')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_xlabel('Time (s)')
    
    if clean_signal is not None:
        f_clean, t_clean, Zxx_clean = compute_stft_vis(clean_signal)
        axes[1, 1].pcolormesh(t_clean, f_clean, 20 * np.log10(Zxx_clean + 1e-8), shading='gouraud', cmap='viridis')
        axes[1, 1].set_title('Clean STFT')
        axes[1, 1].set_ylabel('Frequency (Hz)')
        axes[1, 1].set_xlabel('Time (s)')
    
    f_denoised, t_denoised, Zxx_denoised = compute_stft_vis(denoised_signal)
    axes[1, 2].pcolormesh(t_denoised, f_denoised, 20 * np.log10(Zxx_denoised + 1e-8), shading='gouraud', cmap='viridis')
    axes[1, 2].set_title('Denoised STFT')
    axes[1, 2].set_ylabel('Frequency (Hz)')
    axes[1, 2].set_xlabel('Time (s)')
    
    # === 第三行：对比 ===
    if clean_signal is not None:
        # 时域对比
        axes[2, 0].plot(time, clean_signal, label='Clean', color='green', alpha=0.7)
        axes[2, 0].plot(time, denoised_signal, label='Denoised', color='red', alpha=0.7, linestyle='--')
        axes[2, 0].set_title('Time Domain Comparison')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Amplitude')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 误差
        error = clean_signal - denoised_signal
        axes[2, 1].plot(time, error, color='purple', alpha=0.7)
        axes[2, 1].set_title(f'Reconstruction Error (MSE={np.mean(error**2):.4f})')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Error')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 功率谱密度
        f_clean_psd, psd_clean = scipy_signal.welch(clean_signal, fs=fs, nperseg=n_fft)
        f_denoised_psd, psd_denoised = scipy_signal.welch(denoised_signal, fs=fs, nperseg=n_fft)
        
        axes[2, 2].semilogy(f_clean_psd, psd_clean, label='Clean', color='green', alpha=0.7)
        axes[2, 2].semilogy(f_denoised_psd, psd_denoised, label='Denoised', color='red', alpha=0.7, linestyle='--')
        axes[2, 2].set_title('Power Spectral Density')
        axes[2, 2].set_xlabel('Frequency (Hz)')
        axes[2, 2].set_ylabel('PSD')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
    else:
        # 仅显示降噪前后对比
        axes[2, 0].plot(time, noisy_signal, label='Noisy', alpha=0.7)
        axes[2, 0].plot(time, denoised_signal, label='Denoised', color='red', alpha=0.7, linestyle='--')
        axes[2, 0].set_title('Denoising Comparison')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Amplitude')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 可视化已保存: {save_path}")
    
    plt.show()


def main(args):
    # 创建推理器
    inferencer = UNetInference(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )
    
    # 加载测试数据
    if args.test_clean_path and args.test_noisy_path:
        print(f"\n📂 加载测试数据...")
        clean_signals = np.load(args.test_clean_path)
        noisy_signals = np.load(args.test_noisy_path)
        
        print(f"   Clean: {clean_signals.shape}")
        print(f"   Noisy: {noisy_signals.shape}")
    else:
        # 从 dataset_mixed 加载
        clean_signals = np.load('dataset_mixed/test/clean/data.npy')
        noisy_signals = np.load('dataset_mixed/test/noisy/data.npy')
    
    # 推理
    print(f"\n🚀 开始推理...")
    
    if args.num_samples > 0:
        num_samples = min(args.num_samples, len(noisy_signals))
        noisy_signals = noisy_signals[:num_samples]
        clean_signals = clean_signals[:num_samples]
    
    denoised_signals = inferencer.denoise_batch(noisy_signals)
    
    # 计算指标
    mse = np.mean((clean_signals - denoised_signals) ** 2)
    mae = np.mean(np.abs(clean_signals - denoised_signals))
    
    # SNR 改善
    noise_before = clean_signals - noisy_signals
    noise_after = clean_signals - denoised_signals
    
    snr_before = 10 * np.log10(np.mean(clean_signals ** 2) / (np.mean(noise_before ** 2) + 1e-8))
    snr_after = 10 * np.log10(np.mean(clean_signals ** 2) / (np.mean(noise_after ** 2) + 1e-8))
    snr_improvement = snr_after - snr_before
    
    print(f"\n📊 评估结果:")
    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")
    print(f"   SNR (Before): {snr_before:.2f} dB")
    print(f"   SNR (After): {snr_after:.2f} dB")
    print(f"   SNR Improvement: {snr_improvement:.2f} dB")
    
    # 保存结果
    if args.save_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / 'denoised_signals.npy', denoised_signals)
        print(f"\n💾 降噪结果已保存: {output_dir / 'denoised_signals.npy'}")
    
    # 可视化
    if args.visualize:
        vis_dir = Path(args.output_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 可视化前几个样本
        num_vis = min(args.num_visualize, len(noisy_signals))
        
        for i in range(num_vis):
            save_path = vis_dir / f'sample_{i:03d}.png'
            
            visualize_results(
                noisy_signal=noisy_signals[i],
                clean_signal=clean_signals[i],
                denoised_signal=denoised_signals[i],
                save_path=str(save_path),
                n_fft=inferencer.config['n_fft'],
                hop_length=inferencer.config['hop_length'],
                fs=args.sample_rate
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='U-Net Denoising Inference')
    
    parser.add_argument('--model_path', type=str, default='checkpoints_unet/best.pth', 
                        help='模型权重路径')
    parser.add_argument('--config_path', type=str, default=None, 
                        help='配置文件路径（默认从 checkpoint_dir 自动查找）')
    parser.add_argument('--test_clean_path', type=str, default=None, 
                        help='测试集干净信号路径')
    parser.add_argument('--test_noisy_path', type=str, default=None, 
                        help='测试集带噪信号路径')
    parser.add_argument('--num_samples', type=int, default=0, 
                        help='推理样本数（0 表示全部）')
    parser.add_argument('--device', type=str, default='cuda', 
                        choices=['cuda', 'cpu'], help='设备')
    
    parser.add_argument('--save_results', action='store_true', 
                        help='保存降噪结果')
    parser.add_argument('--output_dir', type=str, default='results_unet', 
                        help='输出目录')
    
    parser.add_argument('--visualize', action='store_true', 
                        help='可视化结果')
    parser.add_argument('--num_visualize', type=int, default=5, 
                        help='可视化样本数')
    parser.add_argument('--sample_rate', type=int, default=250, 
                        help='采样率（用于可视化）')
    
    args = parser.parse_args()
    
    main(args)
