from matplotlib import pyplot as plt
import torchaudio as ta
import torch
import sys
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torchaudio.compliance.kaldi as kaldi
from utils import *

class CustomDataset(Dataset):
    """ PATH = [dir1, dir2 , ...] 

    path_dir_wb=[ "/mnt/hdd/Dataset/FSD50K_16kHz", 
                "/mnt/hdd/Dataset/MUSDB18_HQ_16kHz_mono"],
    path_dir_nb=["/mnt/hdd/Dataset/FSD50K_16kHz_codec",
                 "/mnt/hdd/Dataset/MUSDB18_MP3_8k"],
                 """
    def __init__(self, path_dir_nb, path_dir_wb, seg_len=0.9, sr=48000, mode="train"):
        assert isinstance(path_dir_nb, list), "PATH must be a list"

        self.seg_len = seg_len
        self.mode = mode
        self.sr = sr

        paths_wav_wb = []
        paths_wav_nb = []
        self.labels = []
        self.path_lengths = {}

        # number of dataset -> ['path1','path2']
        for i in range(len(path_dir_nb)):
            self.path_dir_nb = path_dir_nb[i]
            self.path_dir_wb = path_dir_wb[i]

            wb_files = get_audio_paths(self.path_dir_wb, file_extensions='.wav')
            nb_files = get_audio_paths(self.path_dir_nb, file_extensions='.wav')
            paths_wav_wb.extend(wb_files)
            paths_wav_nb.extend(nb_files)

            # Assign labels based on path1, path2
            self.labels.extend([i] * len(wb_files)) 
            self.path_lengths[f'idx{i}len'] = len(wb_files)
            print(f"Index:{i} with {len(wb_files)} samples")

        print(f"LR {len(paths_wav_nb)} and HR {len(paths_wav_wb)} file numbers loaded!")

        if len(paths_wav_wb) != len(paths_wav_nb):
            sys.exit(f"Error: LR {len(paths_wav_nb)} and HR {len(paths_wav_wb)} file numbers are different!")

        # make filename wb-nb        
        self.filenames = [(path_wav_wb, path_wav_nb) for path_wav_wb, path_wav_nb in zip(paths_wav_wb, paths_wav_nb)]
        print(f"{mode}: {len(self.filenames)} files loaded")

    def get_class_counts(self):
        return [self.path_lengths[f'idx{i}len'] for i in range(len(self.path_lengths))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path_wav_wb, path_wav_nb = self.filenames[idx]
        # get label with current nb-wb pair
        label = self.labels[idx]

        wav_nb, sr_nb = ta.load(path_wav_nb)
        wav_wb, sr_wb = ta.load(path_wav_wb)

        wav_wb = wav_wb.view(1, -1)
        wav_nb = wav_nb.view(1, -1)

        if self.seg_len > 0 and self.mode == "train":
            duration = int(self.seg_len * self.sr) #43200
            sig_len = wav_nb.shape[-1] # 43200 or 48000

            if sig_len < duration:
                "crop out 100 nb and repeat"
                print('short')
                t_start = 0
                t_end = t_start + duration # 43200
                wav_nb = wav_nb[...,100:]
                wav_wb = wav_wb[...,:-100]
                wav_nb = wav_nb.repeat(1, t_end // sig_len + 1)[..., :duration]
                wav_wb = wav_wb.repeat(1, t_end // sig_len + 1)[..., :duration]
                
            elif sig_len > 43200:
                "crop out 100"
                print('long')
                t_start = np.random.randint(low=0, high=np.max([1, sig_len - duration - 200]), size=1)[0]
                t_end = t_start + duration
                wav_nb = wav_nb[...,100+t_start:100+t_end]
                wav_wb = wav_wb[...,t_start:t_end]
                
            elif sig_len == 43200:
                wav_nb = wav_nb[...,:]
                wav_wb = wav_wb[...,:duration]
            else:
                ValueError(f"wrong nb length for {path_wav_nb}")

            
            # # t_start = np.random.randint(low=0, high=np.max([1, sig_len - duration - 2]), size=1)[0] # random start
            # # if t_start % 2 == 1:
            #     # t_start -= 1
            # t_end = t_start + duration

            # #### repeat for short signals
            # wav_nb = wav_nb.repeat(1, t_end // sig_len + 1)[..., t_start:t_end]
            # wav_wb = wav_wb.repeat(1, t_end // sig_len + 1)[..., t_start:t_end]
            
            # #### Length ensure
            # wav_nb = self.ensure_length(wav_nb, sr_nb * self.seg_len)
            # wav_wb = self.ensure_length(wav_wb, sr_wb * self.seg_len)

        elif self.mode == "val":
            ### Need to be modified
            # min_len = min(wav_wb.shape[-1], wav_nb.shape[-1])
            # wav_nb = self.ensure_length(wav_nb, sr_nb * self.seg_len)
            # wav_wb = self.ensure_length(wav_wb, sr_nb * self.seg_len)
            # wav_nb = self.set_maxlen(wav_nb, max_lensec=5.12)
            # wav_wb = self.set_maxlen(wav_wb, max_lensec=5.12)
            # print('val')
            pass
        else:
            sys.exit(f"unsupported mode! (train/val)")

        # Compute Spectrogram from wideband
        spec = self.get_log_spectrogram(wav_wb)
        spec = self.normalize_spec(spec)

        # Extract Subbands from WB spectrogram
        spec = self.extract_subband(spec, start=6, end=31)

        return wav_wb, wav_nb, spec, get_filename(path_wav_wb)[0], label

    @staticmethod
    def ensure_length(wav, target_length):
        target_length = int(target_length)
        if wav.shape[1] < target_length:
            pad_size = target_length - wav.shape[1]
            wav = F.pad(wav, (0, pad_size))
        elif wav.shape[1] > target_length:
            wav = wav[:, :target_length]
        return wav
        
    def set_maxlen(self, wav, max_lensec):
        sr = self.sr
        max_len = int(max_lensec * sr)
        if wav.shape[1] > max_len:
            # print(wav.shape, max_len)
            wav = wav[:, :max_len]
        return wav

    @staticmethod
    def get_log_spectrogram(waveform):
        n_fft = 2048
        hop_length = 2048 
        win_length = 2048

        spectrogram = ta.transforms.Spectrogram(
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length, 
            power=2.0
        )(waveform)

        # return spectrogram[:, :]  
        log_spectrogram = ta.transforms.AmplitudeToDB()(spectrogram)
        spec_length = waveform.shape[-1] // 2048
        return log_spectrogram[..., :spec_length]  

    def normalize_spec(self, spec):
        norm_mean = -42.61
        norm_std = 25.79
        spec = (spec - norm_mean) / (norm_std * 2)
        return spec
    
    def extract_subband(self, spec, start=6, end=31):
        """ Get spectrogram Inputs and extract range of subbands : [start:end] """
        
        C,F,T = spec.shape
        num_subband = 32
        freqbin_size = F // num_subband
        dc_line = spec[:,0,:].unsqueeze(1)

        f_start = 1 + freqbin_size * start
        f_end = 1 + freqbin_size * (end+1)

        extracted_spec = spec[:,f_start:f_end,:]
        if f_start == 1:
            extracted_spec = torch.cat((dc_line, extracted_spec),dim=1)

        # print(f_start/1024 * 24000, f_end/1024 * 24000)
        return extracted_spec

