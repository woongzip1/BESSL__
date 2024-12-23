import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.signal.windows import hann
# from soundstream.balancer import *


class LossCalculator:
    def __init__(self, config, discriminator):
        self.discriminator = discriminator
        self.lambda_adv_loss = config['loss']['lambda_adv_loss']
        self.lambda_fm_loss = config['loss']['lambda_fm_loss']
        self.lambda_mel_loss = config['loss']['lambda_mel_loss']
        self.lambda_commitment_loss = config['loss']['lambda_commitment_loss']
        self.lambda_codebook_loss = config['loss']['lambda_codebook_loss']
        self.ms_mel_loss_config = config['loss']['ms_mel_loss_config']

    def compute_generator_loss(self, hr, bwe, commitment_loss, codebook_loss):
        g_loss_dict, g_loss_report = self.discriminator.g_loss(hr, bwe, adv_loss_type='hinge')
        ms_mel_loss_value = ms_mel_loss(hr, bwe, **self.ms_mel_loss_config)
        loss_G = (
            self.lambda_adv_loss * g_loss_dict.get('adv_g', 0) +
            self.lambda_fm_loss * g_loss_dict.get('fm', 0) +
            self.lambda_mel_loss * ms_mel_loss_value +
            self.lambda_commitment_loss * commitment_loss +
            self.lambda_codebook_loss * codebook_loss
        )
        return loss_G, ms_mel_loss_value, g_loss_dict, g_loss_report

    def compute_discriminator_loss(self, hr, bwe):
        d_loss_dict, d_loss_report = self.discriminator.d_loss(hr, bwe, adv_loss_type='hinge')
        import pdb
        loss_D = d_loss_dict.get('adv_d', 0)
        # pdb.set_trace()
        # loss_D = sum(d_loss_dict.values())

        return loss_D, d_loss_dict, d_loss_report
    
def ms_mel_loss(x, x_hat, n_fft_list=[32, 64, 128, 256, 512, 1024, 2048], hop_ratio=0.25, 
                mel_bin_list=[5, 10, 20, 40, 80, 160, 320], fmin=0, fmax=None, sr=44100, mel_power=1.0, 
                eps=1e-5, reduction='sum', **kwargs):
    """
    Multi-scale spectral energy distance loss
    References:
        Kumar, Rithesh, et al. "High-Fidelity Audio Compression with Improved RVQGAN." NeurIPS, 2023.
    Args:
        x (torch.Tensor) [B, ..., T]: ground truth waveform
        x_hat (torch.Tensor) [B, ..., T]: generated waveform
        n_fft_list (List of int): list that contains n_fft for each scale
        hop_ratio (float): hop_length = n_fft * hop_ratio
        mel_bin_list (List of int): list that contains the number of mel bins for each scale
        sr (int): sampling rate
        fmin (float): minimum frequency for mel-filterbank calculation
        fmax (float): maximum frequency for mel-filterbank calculation
        mel_power (float): power to raise magnitude to before taking log
    Returns:
    """
    
    assert len(n_fft_list) == len(mel_bin_list)

    loss = 0
    
    for n_fft, mel_bin in zip(n_fft_list, mel_bin_list):
        sig_to_spg = T.Spectrogram(n_fft=n_fft, win_length=n_fft, hop_length=int(n_fft * hop_ratio), 
                                    window_fn=hann, wkwargs={"sym": False},\
                                    power=1.0, normalized=False, center=True).to(x.device)
        spg_to_mel = T.MelScale(n_mels=mel_bin, sample_rate=sr, n_stft=n_fft//2+1, f_min=fmin, f_max=fmax, norm="slaney", mel_scale="slaney").to(x.device)  
        x_mel = spg_to_mel(sig_to_spg(x))  # [B, C, mels, T]
        x_hat_mel = spg_to_mel(sig_to_spg(x_hat))
        
        log_term = torch.sum(torch.abs(x_mel.clamp(min=eps).pow(mel_power).log10() - x_hat_mel.clamp(min=eps).pow(mel_power).log10()))
        
        if reduction == 'mean':
            log_term /= torch.numel(x_mel)
        elif reduction == 'sum':
            log_term /= x_mel.shape[0]
        
        loss += log_term
        
    return loss