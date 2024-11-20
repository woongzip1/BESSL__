from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2
from visqol import visqol_lib_py

from tqdm import tqdm
import sys
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from models.SEANet import SEANet
from models.SEANet_TFiLM import SEANet_TFiLM
from models.SEANet_TFiLM_nok import SEANet_TFiLM as SEANet_TFiLM_nok
from models.SEANet_TFiLM_nok_modified import SEANet_TFiLM as SEANet_TFiLM_nokmod
from models.SEANet_TFiLM_RVQ import SEANet_TFiLM as SEANet_TFiLM_RVQ

from dataset import CustomDataset
from utils import draw_spec, lsd_batch, prepare_generator
from main import load_config 
import soundfile as sf

from pesq import pesq

MODEL_MAP = {
    "SEANet_TFiLM": SEANet_TFiLM,
    "SEANet": SEANet,
    "SEANet_TFiLM_nokmod": SEANet_TFiLM_nokmod,
    "SEANet_TFiLM_RVQ": SEANet_TFiLM_RVQ,
}

# DEVICE = 'cpu'
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model, checkpoint_path):
    model = model.to(DEVICE)
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['generator_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    return model

def visualize_combined_spectrogram(wb_spec, nb_spec, recon_spec, index, name, output_dir, vmin=-50, vmax=40, sr=48000, 
                                   save_separate=False, plot_colorbar=True):
    """
    Wideband, Narrowband, and Reconstructed 스펙트로그램을 하나의 큰 그림에 합쳐서 저장하거나
    세 개의 스펙트로그램을 각각 따로 저장할 수 있는 옵션 추가.
    주파수 축을 48kHz 샘플링에 맞게 24kHz까지 표시.
    """
    freq_bins = wb_spec.shape[0]
    max_freq = sr // 2
    freq_range = np.linspace(0, max_freq, freq_bins)

    os.makedirs(output_dir, exist_ok=True)

    if save_separate:
        specs = [(wb_spec, "Ground Truth (GT)"), (nb_spec, "Narrowband (NB)"), (recon_spec, "Reconstructed")]
        for i, (spec, title) in enumerate(specs):
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(spec, aspect='auto', origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_yticks(np.linspace(0, freq_bins - 1, num=6))
            ax.set_yticklabels([f"{int(f)}" for f in np.linspace(0, max_freq, num=6)])

            if plot_colorbar:
                fig.colorbar(im, ax=ax)

            output_path = os.path.join(output_dir, f"{name}_{i}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Separate figure saved: {output_path}", end="\r")
    else:
        fig, axes = plt.subplots(3, 1, figsize=(18, 15))

        # Plot WB spectrogram
        im = axes[0].imshow(wb_spec, aspect='auto', origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground Truth (GT)")
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].set_yticks(np.linspace(0, freq_bins - 1, num=6))
        axes[0].set_yticklabels([f"{int(f)}" for f in np.linspace(0, max_freq, num=6)])

        # Plot NB spectrogram
        axes[1].imshow(nb_spec, aspect='auto', origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1].set_title("Narrowband (NB)")
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].set_yticks(np.linspace(0, freq_bins - 1, num=6))
        axes[1].set_yticklabels([f"{int(f)}" for f in np.linspace(0, max_freq, num=6)])

        # Plot Reconstructed spectrogram
        axes[2].imshow(recon_spec, aspect='auto', origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[2].set_title("Reconstructed")
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Frequency (Hz)')
        axes[2].set_yticks(np.linspace(0, freq_bins - 1, num=6))
        axes[2].set_yticklabels([f"{int(f)}" for f in np.linspace(0, max_freq, num=6)])

        if plot_colorbar:
            fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

        output_path = os.path.join(output_dir, f"combined_{name}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)  # Close the figure to avoid display in interactive environments
        print(f"Combined figure saved for index {index}: {output_path}", end="\r")

from visqol.pb2 import visqol_config_pb2
from visqol import visqol_lib_py
import librosa
import numpy as np
import os
import torch

class ViSQOL:
    def __init__(self, sample_rate=48000, use_speech_scoring=False, svr_model_name="libsvm_nu_svr_model.txt"):
        self.config = visqol_config_pb2.VisqolConfig()
        self.config.audio.sample_rate = sample_rate
        self.config.options.use_speech_scoring = use_speech_scoring
        self.config.options.svr_model_path = os.path.join(os.path.dirname(visqol_lib_py.__file__), "model", svr_model_name)

        self.api = visqol_lib_py.VisqolApi()
        self.api.Create(self.config)

    def measure(self, ref_audio, deg_audio):
        ref_audio = ref_audio.astype(np.float64)
        deg_audio = deg_audio.astype(np.float64)
        similarity_result = self.api.Measure(ref_audio, deg_audio)
        return similarity_result.moslqo

def calculate_visqol_score(ref_audio, deg_audio):
    visqol = ViSQOL(sample_rate=48000, use_speech_scoring=False, svr_model_name="libsvm_nu_svr_model.txt")
    mos_lqo = visqol.measure(ref_audio, deg_audio)
    return mos_lqo

###################################################################################################

def main(config, path_wb, path_nb, output_dir, save_files, reduce=False, measure_visqol=False):
    import warnings
    warnings.filterwarnings("ignore", message="MessageFactory class is deprecated.*")
    dataset = CustomDataset(path_dir_nb=path_nb, path_dir_wb=path_wb, seg_len=1, mode="val", start_index=config['dataset']['start_index'], high_index=31)

    os.makedirs(output_dir, exist_ok=True)

    model = prepare_generator(config, MODEL_MAP)
    model = load_model(model, config['train']['ckpt_path'])

    torch.manual_seed(42)
    np.random.seed(42)
    NUMSAMPLES = 16
    indices = torch.randperm(len(dataset))
    if reduce:
        indices = torch.randperm(len(dataset))[0:NUMSAMPLES]

    metrics = {
        "LSD_Low": [],
        "LSD_Mid": [],
        "LSD_High": [],
    }
    if measure_visqol:
        metrics["ViSQOL"]=[]
        
    metric_calculations = [
        ("LSD_Low", lambda wb, recon: lsd_batch(wb, recon, fs=48000, start=0, cutoff_freq=4240)),
        ("LSD_Mid", lambda wb, recon: lsd_batch(wb, recon, fs=48000, start=4240, cutoff_freq=12000)),
        ("LSD_High", lambda wb, recon: lsd_batch(wb, recon, fs=48000, start=4240, cutoff_freq=24000)),
    ]

    Visqol = ViSQOL(sample_rate=48000, use_speech_scoring=False, svr_model_name="libsvm_nu_svr_model.txt")
    bar = tqdm(range(len(dataset)))
    for idx in bar:
        # idx = idx.item()
        wb, nb, spec, name, label = dataset[idx]
        with torch.no_grad():
            recon = model(nb.to(DEVICE), spec.to(DEVICE)) 
            if config['generator']['type'] == "SEANet_TFiLM_RVQ":
                recon = recon[0] # rvq model has two outputs
            
        for metric_name, calc_func in metric_calculations:
            metrics[metric_name].append(calc_func(wb.to('cpu').numpy(), recon.to('cpu').numpy()))

        if measure_visqol:
            # visqol_score = calculate_visqol_score(wb.to('cpu').numpy().squeeze(), recon.to('cpu').numpy().squeeze())
            visqol_score = Visqol.measure(wb.to('cpu').numpy().squeeze(), recon.to('cpu').numpy().squeeze()) # recon
            metrics["ViSQOL"].append(visqol_score)


        from utils import lpf
        recon_lpf = lpf(recon.cpu().squeeze(), sr=48000, cutoff=4240)
        recon_lpf2 = lpf(recon.cpu().squeeze(), sr=48000, cutoff=12000)
        gt_lpf = lpf(wb.cpu().squeeze(), sr=48000, cutoff=4240)
        
        if save_files:
            sf.write(f"{output_dir}/{name}_gt.wav", wb.cpu().squeeze(), format="WAV", samplerate=48000)
            sf.write(f"{output_dir}/{name}.wav", recon.cpu().squeeze(), format="WAV", samplerate=48000)
            sf.write(f"{output_dir}/{name}_lpf.wav", recon_lpf.squeeze(), format="WAV", samplerate=48000)
            sf.write(f"{output_dir}/{name}_12lpf.wav", recon_lpf2.squeeze(), format="WAV", samplerate=48000)
            sf.write(f"{output_dir}/{name}_gtlpf.wav", gt_lpf.squeeze(), format="WAV", samplerate=48000)
            sf.write(f"{output_dir}/{name}_core.wav", nb.squeeze(), format="WAV", samplerate=48000)
        # print(idx,'\r')

        wb_spec = draw_spec(wb.cpu().squeeze().numpy(), sr=48000, return_fig=False, vmin=-50, vmax=40)
        nb_spec = draw_spec(nb.cpu().squeeze().numpy(), sr=48000, return_fig=False, vmin=-50, vmax=40)
        recon_spec = draw_spec(recon.cpu().squeeze().numpy(), sr=48000, return_fig=False, vmin=-50, vmax=40)
        visualize_combined_spectrogram(wb_spec, nb_spec, recon_spec, index=idx, name=name, output_dir=output_dir, vmin=-50, vmax=40, save_separate=False, plot_colorbar=False)

    for metric_name, values in metrics.items():
        avg_value = sum(values) / len(values)
        print(f"Average {metric_name}: {avg_value:.4f}")
        # sys.stdout.flush()  

    log_file_path = f"{output_dir}/log.txt"
    with open(log_file_path, "w") as log_file:  # "w" mode creates a new file or overwrites an existing one
        for metric_name, values in metrics.items():
            avg_value = sum(values) / len(values)
            log_file.write(f"Average {metric_name}: {avg_value:.4f}\n")
            # log_file.flush()  

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with specified config.")
    ## ckpts/P4EXP1/P4_EXP1.yaml
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--device", type=str, required=True, default='cuda')
    args = parser.parse_args()
    
    # Load common configuration
    config = load_config(args.config)
    DEVICE = args.device
    
    # Define datasets
    datasets = [
        {
            "name": "Audio",
            "path_wb": ["/home/woongjib/Projects/Dataset_BESSL/USAC44_mono_48k"],
            "path_nb": ["/home/woongjib/Projects/Dataset_BESSL/USAC44_mono_48k_core"],
            "output_dir": config['eval']['eval_dir_audio']
        },

        {
            "name": "Speech",
            "path_wb": ["/home/woongjib/Projects/Dataset_BESSL/DAPS_gt_small"],
            "path_nb": ["/home/woongjib/Projects/Dataset_BESSL/DAPS_core_small"],
            "output_dir": config['eval']['eval_dir_speech']
        },

        # {
        #     "name": "SBR",
        #     "path_wb": ["/home/woongjib/Projects/USAC44_mono_48k"],
        #     "path_nb": ["/home/woongjib/Projects/USAC44_mono_48k_HEAAC16"],
        #     "output_dir": "outputs/SBR"
        # }
    ]

    # Run main for each dataset
    for dataset in datasets:
        print(f"\nProcessing {dataset['name']} Dataset:")
        main(config, dataset["path_wb"], dataset["path_nb"], dataset["output_dir"], save_files=True, reduce=False, measure_visqol=True)




