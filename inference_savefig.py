import torch
import numpy as np
import os
from tqdm import tqdm
import torchaudio as ta
import matplotlib.pyplot as plt
from models.SEANet import SEANet
from models.SEANet_TFiLM import SEANet_TFiLM
from models.SEANet_TFiLM_nok import SEANet_TFiLM as SEANet_TFiLM_nok

from dataset import CustomDataset
from utils import draw_spec, lsd_batch
import torch.nn.functional as F
import soundfile as sf




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
    # 샘플링 레이트에 맞춰 주파수 범위를 설정 (Nyquist frequency는 sr / 2)
    freq_bins = wb_spec.shape[0]
    max_freq = sr // 2
    freq_range = np.linspace(0, max_freq, freq_bins)

    os.makedirs(output_dir, exist_ok=True)

    if save_separate:
        # 각 스펙트로그램을 개별적으로 저장
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

            # 저장
            output_path = os.path.join(output_dir, f"{name}_{i}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Separate figure saved: {output_path}", end="\r")
    else:
        # 하나의 큰 그림으로 합쳐서 저장
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

        # 하나의 colorbar 추가 (세 번째 스펙트로그램 기준)
        if plot_colorbar:
            fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

        # 저장
        output_path = os.path.join(output_dir, f"combined_{name}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)  # Close the figure to avoid display in interactive environments
        print(f"Combined figure saved for index {index}: {output_path}", end="\r")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    ################### Data
    # Dataloader 설정
    # path_wb = [
    #             "/mnt/hdd/Dataset/FSD50K_48kHz/FSD50K.eval_audio", 
    #             "/mnt/hdd/Dataset/MUSDB18_HQ_mono_48kHz/test", 
    #             ]
    # path_nb = [
    #             "/mnt/hdd/FSD50K_CORE_fir_crop/FSD50K.eval_audio", 
    #             "/mnt/hdd/MUSDB18_CORE_fir_crop/test", 
    #             ]

    # dataset = CustomDataset(path_dir_nb=path_nb, path_dir_wb=path_wb, seg_len=1, mode="val")
 
    # 1 sec
    path_wb = [
                "/mnt/hdd/Dataset_BESSL/FSD50K_WB_SEGMENT/FSD50K.eval_audio", 
                "/mnt/hdd/Dataset_BESSL/MUSDB_WB_SEGMENT/test", 
                ]
    path_nb = [
                "/mnt/hdd/Dataset_BESSL/FSD50K_NB_SEGMENT/FSD50K.eval_audio", 
                "/mnt/hdd/Dataset_BESSL/MUSDB_NB_SEGMENT/test", 
                ]
    
    # # For HE-AAC
    # path_wb = ["/home/woongjib/Projects/USAC44_mono_48k"]
    # path_nb = ["/home/woongjib/Projects/USAC44_mono_48k_HEAAC16_Crop"]

    # HE-AAC
    path_wb = ["/home/woongjib/Projects/USAC44_mono_48k"]
    path_nb = ["/home/woongjib/Projects/USAC44_mono_48k_HEAAC16_LPF_Crop"]

    dataset = CustomDataset(path_dir_nb=path_nb, path_dir_wb=path_wb, seg_len=1, mode="val")

    ################### Model
    # TFiLM 64
    model = SEANet_TFiLM(kmeans_model_path="/home/woongjib/Projects/BESSL__/kmeans/K64_MAE.pkl")
    model = load_model(model, "/home/woongjib/Projects/BESSL__/ckpt_K64/epoch_41_lsdH_0.441.pth")
    
    # No K
    model = SEANet_TFiLM_nok(kmeans_model_path=None)
    model = load_model(model, "/home/woongjib/Projects/BESSL__/ckpt_nok/epoch_13_lsdH_0.430.pth")

    # Blind
    # model = SEANet()
    # model = load_model(model, "/home/woongjib/Projects/BESSL__/ckpt_baseline/epoch_26_lsdH_0.550.pth")

    # Output 디렉토리 설정
    output_dir = "output_samples_tempor"
    # output_dir = "output_samples_noK_"
    os.makedirs(output_dir, exist_ok=True)

    # Random
    torch.manual_seed(42)
    np.random.seed(42)
    NUMSAMPLES = 3
    indices = torch.randperm(len(dataset))[:NUMSAMPLES]  # 랜덤하게 20개 샘플 추출
    print(indices)
    lsd_list = []
    lsd_highlist = []

    import warnings
    warnings.filterwarnings("ignore", message=".*nperseg = .*")
    warnings.filterwarnings("ignore", message=".*cudnnException.*")

    # pbar = tqdm(indices, dynamic_ncols=False)
    for idx in indices:
        idx = idx.item()
        wb, nb, spec, name, label = dataset[idx]
        print(f"Processing: {name}", end="\r")

        with torch.no_grad():
            # Narrowband 데이터를 입력으로 받아 Reconstruction 수행
            recon = model(nb.to(DEVICE), spec.to(DEVICE)).detach()

        # LSD 계산
        lsd = lsd_batch(wb.to('cpu').numpy(), recon.to('cpu').numpy(), fs=48000)
        print(lsd)
        lsd_list.append(lsd)        
        # LSD 계산
        lsd_high = lsd_batch(wb.to('cpu').numpy(), recon.to('cpu').numpy(), fs=48000, start=4500, cutoff_freq=24000)
        lsd_highlist.append(lsd_high)

        # draw_spec을 사용해 스펙트로그램을 생성하고 저장
        wb_spec = draw_spec(wb.cpu().squeeze().numpy(), sr=48000, return_fig=False, vmin=-50, vmax=40)
        nb_spec = draw_spec(nb.cpu().squeeze().numpy(), sr=48000, return_fig=False, vmin=-50, vmax=40)
        recon_spec = draw_spec(recon.cpu().squeeze().numpy(), sr=48000, return_fig=False, vmin=-50, vmax=40)

        # 하나의 큰 그림으로 합쳐서 저장
        visualize_combined_spectrogram(wb_spec, nb_spec, recon_spec, index=idx, name=name, output_dir=output_dir, vmin=-50, vmax=40,
                                       save_separate=False, plot_colorbar=False)

        # recon_spec = recon_spec.squeeze()
        sf.write(f"{output_dir}/{name}.wav", recon.cpu().squeeze(), format="WAV", samplerate=48000)
        sf.write(f"{output_dir}/{name}_gt.wav", wb.cpu().squeeze(), format="WAV", samplerate=48000)
        sf.write(f"{output_dir}/{name}_nb.wav", nb.cpu().squeeze(), format="WAV", samplerate=48000)
        # sf.write(f"{output_dir}/{name}.wav", nb.cpu().squeeze(), format="WAV", samplerate=48000)

        # ta.save(f"output_samples/{idx}.wav", recon_spec, sample_rate=48000)
        # sf.write(f"{output_dir}/{name}_GT.wav", wb.cpu().squeeze(), format="WAV",samplerate=48000)

    
    # LSD 및 LSD High 평균 계산 및 출력
    average_lsd = sum(lsd_list) / len(lsd_list)
    average_lsd_high = sum(lsd_highlist) / len(lsd_highlist)
    print(f"\nAverage LSD for the 20 samples: {average_lsd:.4f}")
    print(f"Average LSD High for the 20 samples: {average_lsd_high:.4f}")

if __name__ == "__main__":
    main()
