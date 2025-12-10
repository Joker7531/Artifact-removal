"""
Inference and Evaluation for Waveform GAN
用于加载训练好的模型进行波形去噪推理
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from waveform_models import WaveformGenerator
from waveform_dataset import WaveformProcessor


class WaveformDenoiser:
    """波形去噪推理器"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path: 训练好的模型 checkpoint 路径
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # 创建模型
        self.generator = WaveformGenerator(
            in_channels=1,
            out_channels=1,
            base_filters=self.config['base_filters'],
            num_layers=self.config['num_layers']
        )
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.to(self.device)
        self.generator.eval()
        
        # 创建波形处理器
        self.processor = WaveformProcessor(
            fs=self.config['fs'],
            window_sec=self.config['window_sec'],
            overlap=self.config['overlap']
        )
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Trained for {checkpoint['epoch']+1} epochs")
        print(f"Using device: {self.device}")
    
    def denoise_signal(self, raw_eeg, normalize_method='zscore'):
        """
        对原始 EEG 信号进行去噪
        
        Args:
            raw_eeg: 1D numpy array, 原始 EEG 信号
            normalize_method: 归一化方法
            
        Returns:
            denoised_eeg: 去噪后的 EEG 信号
        """
        # 滑动窗口切片
        windows = self.processor.sliding_window(raw_eeg)
        
        if len(windows) == 0:
            print("Warning: Signal too short for windowing")
            return raw_eeg
        
        denoised_windows = []
        normalization_params = []
        
        with torch.no_grad():
            for window in windows:
                # 保存归一化参数用于反归一化
                if normalize_method == 'zscore':
                    mean = window.mean()
                    std = window.std()
                    normalization_params.append({'mean': mean, 'std': std})
                elif normalize_method == 'minmax':
                    min_val = window.min()
                    max_val = window.max()
                    normalization_params.append({'min': min_val, 'max': max_val})
                
                # 归一化
                window_norm = self.processor.normalize(window, method=normalize_method)
                
                # 转为 tensor (1, 1, L)
                input_tensor = torch.from_numpy(window_norm).float().unsqueeze(0).unsqueeze(0).to(self.device)
                
                # 生成去噪波形
                output = self.generator(input_tensor)
                denoised_norm = output.squeeze().cpu().numpy()
                
                # 反归一化
                if normalize_method == 'zscore':
                    denoised_window = denoised_norm * normalization_params[-1]['std'] + normalization_params[-1]['mean']
                elif normalize_method == 'minmax':
                    min_val = normalization_params[-1]['min']
                    max_val = normalization_params[-1]['max']
                    denoised_window = (denoised_norm + 1) / 2 * (max_val - min_val) + min_val
                else:
                    denoised_window = denoised_norm
                
                denoised_windows.append(denoised_window)
        
        # 重叠相加重建完整信号
        denoised_eeg = self._overlap_add(denoised_windows)
        
        return denoised_eeg
    
    def _overlap_add(self, windows):
        """重叠相加重建信号"""
        window_length = self.processor.window_length
        step = self.processor.step
        
        # 计算总长度
        total_length = (len(windows) - 1) * step + window_length
        reconstructed = np.zeros(total_length)
        weights = np.zeros(total_length)
        
        # 重叠相加
        for i, window in enumerate(windows):
            start = i * step
            end = start + window_length
            reconstructed[start:end] += window
            weights[start:end] += 1
        
        # 加权平均
        reconstructed = np.divide(reconstructed, weights, 
                                 out=np.zeros_like(reconstructed), 
                                 where=weights!=0)
        
        return reconstructed
    
    def evaluate(self, raw_file, clean_file=None, save_dir='results/inference'):
        """
        评估去噪效果
        
        Args:
            raw_file: 原始 EEG 文件路径
            clean_file: 干净 EEG 文件路径 (可选)
            save_dir: 结果保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取原始信号
        raw_eeg = pd.read_csv(raw_file).values.flatten()
        
        # 去噪
        print(f"Denoising {raw_file}...")
        denoised_eeg = self.denoise_signal(raw_eeg, normalize_method='zscore')
        
        # 截取到相同长度
        min_len = min(len(raw_eeg), len(denoised_eeg))
        raw_eeg = raw_eeg[:min_len]
        denoised_eeg = denoised_eeg[:min_len]
        
        # 可视化
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        
        time_axis = np.arange(min_len) / self.config['fs']
        
        # 原始信号
        axes[0].plot(time_axis, raw_eeg, alpha=0.7, linewidth=0.5)
        axes[0].set_title('Raw EEG Signal', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Amplitude (µV)', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, time_axis[-1]])
        
        # 去噪信号
        axes[1].plot(time_axis, denoised_eeg, alpha=0.7, linewidth=0.5, color='green')
        axes[1].set_title('Denoised EEG Signal (GAN Generated)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Amplitude (µV)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, time_axis[-1]])
        
        # 如果有 clean 信号,显示对比
        if clean_file and Path(clean_file).exists():
            clean_eeg = pd.read_csv(clean_file).values.flatten()[:min_len]
            
            axes[2].plot(time_axis, clean_eeg, alpha=0.6, linewidth=0.7, 
                        label='Ground Truth', color='blue')
            axes[2].plot(time_axis, denoised_eeg, alpha=0.6, linewidth=0.7, 
                        label='GAN Denoised', color='green', linestyle='--')
            axes[2].set_title('Comparison: Ground Truth vs GAN Denoised', 
                            fontsize=14, fontweight='bold')
            axes[2].set_ylabel('Amplitude (µV)', fontsize=11)
            axes[2].set_xlabel('Time (s)', fontsize=11)
            axes[2].legend(fontsize=10)
            axes[2].grid(True, alpha=0.3)
            axes[2].set_xlim([0, time_axis[-1]])
            
            # 计算指标
            mse = np.mean((clean_eeg - denoised_eeg) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(clean_eeg - denoised_eeg))
            
            # 相关系数
            correlation = np.corrcoef(clean_eeg, denoised_eeg)[0, 1]
            
            # SNR 改善
            noise_raw = raw_eeg - clean_eeg
            noise_denoised = denoised_eeg - clean_eeg
            snr_raw = 10 * np.log10(np.var(clean_eeg) / np.var(noise_raw))
            snr_denoised = 10 * np.log10(np.var(clean_eeg) / np.var(noise_denoised))
            snr_improvement = snr_denoised - snr_raw
            
            print(f"\nEvaluation Metrics:")
            print(f"  MSE:  {mse:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE:  {mae:.6f}")
            print(f"  Correlation: {correlation:.6f}")
            print(f"  SNR (Raw):      {snr_raw:.2f} dB")
            print(f"  SNR (Denoised): {snr_denoised:.2f} dB")
            print(f"  SNR Improvement: {snr_improvement:.2f} dB")
            
            # 保存指标
            metrics = {
                'file': str(raw_file),
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'correlation': float(correlation),
                'snr_raw': float(snr_raw),
                'snr_denoised': float(snr_denoised),
                'snr_improvement': float(snr_improvement)
            }
            
            metrics_file = save_dir / 'metrics.txt'
            with open(metrics_file, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
        else:
            axes[2].axis('off')
        
        plt.tight_layout()
        plot_file = save_dir / f'{Path(raw_file).stem}_denoised.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Results saved to {plot_file}")
        
        # 保存去噪信号
        output_file = save_dir / f'{Path(raw_file).stem}_denoised.csv'
        pd.DataFrame(denoised_eeg).to_csv(output_file, index=False, header=False)
        print(f"Denoised signal saved to {output_file}")
        
        return denoised_eeg


def main():
    parser = argparse.ArgumentParser(description='Waveform GAN Denoising Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--raw_file', type=str, default='../2_Data_processed/raw_Cz_10.csv',
                       help='Path to raw EEG file')
    parser.add_argument('--clean_file', type=str, default='../2_Data_processed/clean_Cz_10.csv',
                       help='Path to clean EEG file (optional)')
    parser.add_argument('--save_dir', type=str, default='results/inference',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 创建去噪器
    denoiser = WaveformDenoiser(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # 评估
    denoiser.evaluate(
        raw_file=args.raw_file,
        clean_file=args.clean_file if Path(args.clean_file).exists() else None,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    # 如果直接运行,使用默认参数
    import sys
    if len(sys.argv) == 1:
        print("Running with default parameters...")
        
        checkpoint_path = 'checkpoints/best_model.pth'
        if not Path(checkpoint_path).exists():
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            print("Please train the model first using train_waveform_gan.py")
            sys.exit(1)
        
        denoiser = WaveformDenoiser(
            checkpoint_path=checkpoint_path,
            device='cuda'
        )
        
        # 测试验证集数据
        for idx in [9, 10]:
            raw_file = f'../2_Data_processed/raw_Cz_{idx:02d}.csv'
            clean_file = f'../2_Data_processed/clean_Cz_{idx:02d}.csv'
            
            if Path(raw_file).exists():
                print(f"\n{'='*60}")
                print(f"Processing file {idx:02d}")
                print('='*60)
                denoiser.evaluate(
                    raw_file=raw_file,
                    clean_file=clean_file,
                    save_dir=f'results/inference/test_{idx:02d}'
                )
    else:
        main()
