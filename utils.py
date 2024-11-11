import os
import shutil
import argparse
import torch
import numpy as np
from scipy.signal import stft
import scipy.signal as sig
import torch.nn.functional as F

from matplotlib import pyplot as plt
import librosa

import wandb

def prepare_discriminator(config):
    from discriminator import MultiBandSTFTDiscriminator, SSDiscriminatorBlock
    disc_type = config['discriminator']['type']

    if disc_type == "MultiBandSTFTDiscriminator":
        disc_config = config["discriminator"]['MultiBandSTFTDiscriminator_config']
        discriminator = SSDiscriminatorBlock(
            sd_num=len(disc_config['n_fft_list']),
            C=disc_config['C'],
            n_fft_list=disc_config['n_fft_list'],
            hop_len_list=disc_config['hop_len_list'],
            sd_mode='BS',
            band_split_ratio=disc_config['band_split_ratio']
        )
    else:
        raise ValueError(f"Unsupported discriminator type: {disc_type}")

    # Print information about the loaded model
    print("########################################")
    print(f"Discriminator Type: {disc_type}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    print("########################################")

    return discriminator

def prepare_generator(config, MODEL_MAP):
    gen_type = config['generator']['type']
    if gen_type not in MODEL_MAP:
        raise ValueError(f"Unsupported generator type: {gen_type}")
    
    ModelClass = MODEL_MAP[gen_type]
    
    # Retrieve the parameters for the generator from the config
    model_params = {k: v for k, v in config['generator'].items() if k not in ['type']}
    
    # Print information about the loaded model
    print("########################################")
    print(f"Instantiating {gen_type} Generator with parameters:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    print(f"  type: {gen_type}")
    generator = ModelClass(**model_params)
    print(f"Generator Parameters: {sum(p.numel() for p in generator.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    print("########################################")
    
    return generator


def wandb_log(loglist, epoch, note):
    for key, val in loglist.items():
        if isinstance(val, torch.Tensor):
            item = val.cpu().detach().numpy()
        else:
            item = val
        try:
            if isinstance(item, float):
                log = item
            elif isinstance(item, plt.Figure):
                log = wandb.Image(item)
                plt.close(item)
            elif item.ndim in [2, 3]:  # 이미지 데이터
                log = wandb.Image(item, caption=f"{note.capitalize()} {key.capitalize()} Epoch {epoch}")
            elif item.ndim == 1:  # 오디오 데이터
                log = wandb.Audio(item, sample_rate=48000, caption=f"{note.capitalize()} {key.capitalize()} Epoch {epoch}")
            else:
                log = item
        except Exception as e:
            print(f"Failed to log {key}: {e}")
            log = item

        wandb.log({
            f"{note.capitalize()} {key.capitalize()}": log,
        }, step=epoch)

def save_checkpoint(generator, discriminator, optim_g, optim_d, epoch, lsd_h, config):
    checkpoint_dir = config['train']['ckpt_save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_lsdH_{lsd_h:.3f}.pth")
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optim_g.state_dict(),  # Save optimizer state
        'optimizer_D_state_dict': optim_d.state_dict(),  # Save optimizer state
        'lsd_h': lsd_h,
    }, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, device, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    start_epoch = checkpoint['epoch'] 
    # best_pesq = checkpoint['pesq_score']
    best_lsdh = checkpoint['lsd_h']
    
    if 'optimizer_G_state_dict' in checkpoint and 'optimizer_D_state_dict' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    # Optimizer states are loaded but still on CPU, need to move to GPU
    for state in optimizer_G.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    for state in optimizer_D.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return start_epoch, best_lsdh


def count_params(model, milion=True):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if milion:
        print(f"Trainable parameters: {trainable_params/1000_000:.2f}M")
    else:
        print(f"Trainable parameters: {trainable_params:.2f}")


""" 주어진 디렉토리에서 지정된 확장자를 가진 모든 오디오 파일의 절대 경로를 반환합니다. """
def get_audio_paths(paths: list, file_extensions=['.wav', '.flac']):
    audio_paths = []
    if isinstance(paths, str):
        paths = [paths]
        
    for path in paths:
        for root, dirs, files in os.walk(path):
            audio_paths += [os.path.join(root, file) for file in files if os.path.splitext(file)[-1].lower() in file_extensions]
                        
    audio_paths.sort(key=lambda x: os.path.split(x)[-1])
    
    return audio_paths

def get_filename(path):
    return os.path.splitext(os.path.basename(path))  

def lsd_batch(x_batch, y_batch, fs=16000, frame_size=0.02, frame_shift=0.02, start=0, cutoff_freq=0, nfft=512):
    frame_length = int(frame_size * fs)
    frame_step = int(frame_shift * fs)

    if fs == 48000:
        frame_length = 2048
        frame_step = 2048
        nfft = 2048

    if isinstance(x_batch, np.ndarray):
        x_batch = torch.from_numpy(x_batch)
        y_batch = torch.from_numpy(y_batch)
   
    if x_batch.dim()==1:
        batch_size = 1
    ## 1 x 32000
    elif x_batch.dim()==2:
        x_batch=x_batch.unsqueeze(1)
    batch_size, _, signal_length = x_batch.shape
   
    if y_batch.dim()==1:
        y_batch=y_batch.reshape(batch_size,1,-1)
    elif y_batch.dim()==2:
        y_batch=y_batch.unsqueeze(1)
   
    # X and Y Size
    x_len = x_batch.shape[-1]
    y_len = y_batch.shape[-1]
    minlen = min(x_len, y_len)
    x_batch = x_batch[:,:,:minlen]
    y_batch = y_batch[:,:,:minlen]

    lsd_values = []
    for i in range(batch_size):
        x = x_batch[i, 0, :].numpy()
        y = y_batch[i, 0, :].numpy()
 
        # STFT
        ## nfft//2 +1: freq len
        f_x, t_x, Zxx_x = stft(x, fs, nperseg=frame_length, noverlap=frame_length - frame_step, nfft=nfft)
        f_y, t_y, Zxx_y = stft(y, fs, nperseg=frame_length, noverlap=frame_length - frame_step, nfft=nfft)
       
        # Power spec
        power_spec_x = np.abs(Zxx_x) ** 2
        power_spec_y = np.abs(Zxx_y) ** 2
       
        # Log Power Spec
        log_spec_x = np.log10(power_spec_x + 1e-10)  # eps
        log_spec_y = np.log10(power_spec_y + 1e-10)

        if start or cutoff_freq:
            freq_len = log_spec_x.shape[0]
            max_freq = fs // 2
            start = int(start / max_freq * freq_len)
            freq_idx = int(cutoff_freq / max_freq * freq_len)
            log_spec_x = log_spec_x[start:freq_idx,:]
            log_spec_y = log_spec_y[start:freq_idx,:]

        #Spectral Mean
        lsd = np.sqrt(np.mean((log_spec_x - log_spec_y) ** 2, axis=0))
       
        #Frame mean
        mean_lsd = np.mean(lsd)
        lsd_values.append(mean_lsd)
   
    # Batch mean
    batch_mean_lsd = np.mean(lsd_values)
    # return log_spec_x, log_spec_y
    return batch_mean_lsd

## 언젠가는 분석해볼 것
def lsd(self, est, target):
        lsd = torch.log10(target**2 / ((est + 1e-12) ** 2) + 1e-12) ** 2
        lsd = torch.mean(torch.mean(lsd, dim=3) ** 0.5, dim=2)
        return lsd[..., None, None]

def draw_spec(x,
              figsize=(10, 6), title='', n_fft=2048,
              win_len=1024, hop_len=256, sr=16000, cmap='inferno',
              vmin=-50, vmax=40, use_colorbar=True,
              ylim=None,
              title_fontsize=10,
              label_fontsize=8,
                return_fig=False,
                save_fig=False, save_path=None):
    fig = plt.figure(figsize=figsize)
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
    stft = 20 * np.log10(np.clip(np.abs(stft), a_min=1e-8, a_max=None))

    plt.imshow(stft,
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
               origin='lower', extent=[0, len(x) / sr, 0, sr//2])

    if use_colorbar:
        plt.colorbar()

    plt.xlabel('Time (s)', fontsize=label_fontsize)
    plt.ylabel('Frequency (Hz)', fontsize=label_fontsize)

    if ylim is None:
        ylim = (0, sr / 2)
    plt.ylim(*ylim)

    plt.title(title, fontsize=title_fontsize)
    
    if save_fig and save_path:
        plt.savefig(f"{save_path}.png")
    
    if return_fig:
        plt.close()
        return fig
    else:
        # plt.close()
        plt.show()
        return stft

from torch.utils.data import Subset
def SmallDataset(dataset, num_samples):
    total_samples = len(dataset)
    indices = list(range(total_samples))
    random.shuffle(indices)
    random_indices = indices[:num_samples]
    
    subset = Subset(dataset, random_indices)
    return subset

from scipy.signal import firwin, lfilter, freqz
def lpf(y, sr=16000, cutoff=500, plot_resp=False, window='hamming', figsize=(10,2)):
    """ 
    Applies FIR filter
    cutoff freq: cutoff freq in Hz
    """
    nyquist = 0.5 * sr
    normalized_cutoff = cutoff / nyquist
    taps = firwin(numtaps=200, cutoff=normalized_cutoff, window=window)
    y_lpf = lfilter(taps, 1.0, y)
    # y_lpf = np.convolve(y, taps, mode='same')
    
    # plt.plot(taps)
    # plt.show()
    if plot_resp:
        w, h = freqz(taps, worN=8000)
        plt.figure(figsize=figsize)
        plt.plot(0.5*sr*w/np.pi, np.abs(h), 'b')
        plt.title("FIR Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain')
        plt.xlim(0, sr/2)
        plt.grid()
        plt.show()

    return y_lpf

from pystoi import stoi
def py_stoi(clean_audio, processed_audio, sample_rate=16000):
    # Calculate STOI
    stoi_score = stoi(clean_audio, processed_audio, sample_rate, extended=False)
    return stoi_score


# adapted from
# https://github.com/kan-bayashi/ParallelWaveGAN/tree/master/parallel_wavegan
"""PQMF module.

This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.

.. _`Near-perfect-reconstruction pseudo-QMF banks`:
    https://ieeexplore.ieee.org/document/258122

"""
class PQMF(torch.nn.Module):
    def __init__(self, N=4, taps=62, cutoff=0.15, beta=9.0):
        super(PQMF, self).__init__()

        self.N = N
        self.taps = taps
        self.cutoff = cutoff
        self.beta = beta
        
        # scipy FIR filter coefficients
        QMF = sig.firwin(taps + 1, cutoff, window=('kaiser', beta))
        H = np.zeros((N, len(QMF)))
        G = np.zeros((N, len(QMF)))
        for k in range(N):
            constant_factor = (2 * k + 1) * (np.pi /
                                             (2 * N)) * (np.arange(taps + 1) -
                                                         ((taps - 1) / 2))  # TODO: (taps - 1) -> taps
            phase = (-1)**k * np.pi / 4
            # Analysis Filter
            H[k] = 2 * QMF * np.cos(constant_factor + phase)
            # Synthesis Filter
            G[k] = 2 * QMF * np.cos(constant_factor - phase)

        H = torch.from_numpy(H[:, None, :]).float()
        G = torch.from_numpy(G[None, :, :]).float()

        self.register_buffer("H", H)
        self.register_buffer("G", G)

        updown_filter = torch.zeros((N, N, N)).float()
        for k in range(N):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.N = N

        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def forward(self, x):
        return self.analysis(x)

    def analysis(self, x):
        return F.conv1d(x, self.H, padding=self.taps // 2, stride=self.N)

    def synthesis(self, x):
        x = F.conv_transpose1d(x,
                               self.updown_filter * self.N,
                               stride=self.N)
        x = F.conv1d(x, self.G, padding=self.taps // 2)
        return x

import random
def set_seed(seed):
    # Set seed for CPU and CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for Python's built-in random module
    random.seed(seed)

    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

## ViSQOL
class ViSQOL:
    def __init__(self, sample_rate=16000, use_speech_scoring=True, svr_model_name="libsvm_nu_svr_model.txt"):
        from visqol import visqol_lib_py
        from visqol.pb2 import visqol_config_pb2, similarity_result_pb2 
        
        # Configuration
        self.config = visqol_config_pb2.VisqolConfig()
        self.config.audio.sample_rate = sample_rate
        self.config.options.use_speech_scoring = use_speech_scoring

        # Model path
        self.svr_model_path = svr_model_name
        self.config.options.svr_model_path = os.path.join(os.path.dirname(visqol_lib_py.__file__), "model", self.svr_model_path)

        # Initialize ViSQOL
        self.api = visqol_lib_py.VisqolApi()
        self.api.Create(self.config)

    def load_audio(self, file_path):
        # Load audio file using librosa
        audio, _ = librosa.load(file_path, sr=self.config.audio.sample_rate)
        return audio.astype(np.float64)

    def measure(self, ref_audio, deg_audio):
        # Ensure the audio data is in float64 format
        ref_audio = ref_audio.astype(np.float64)
        deg_audio = deg_audio.astype(np.float64)

        # Measure similarity
        similarity_result = self.api.Measure(ref_audio, deg_audio)

        # Return the MOS-LQO score
        return similarity_result.moslqo


def main():
    parser = argparse.ArgumentParser(description="Path for train test split")

if __name__ == "__main__":
    main()

