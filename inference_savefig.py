import torch
import numpy as np
import os
from tqdm import tqdm
import torchaudio as ta
import matplotlib.pyplot as plt

from models.SEANet import SEANet
from models.SEANet_TFiLM import SEANet_TFiLM
from models.SEANet_TFiLM_nok import SEANet_TFiLM as SEANet_TFiLM_nok
from models.SEANet_TFiLM_nok_modified import SEANet_TFiLM as SEANet_TFiLM_nokmod
from models.SEANet_TFiLM_RVQ import SEANet_TFiLM as SEANet_TFiLM_RVQ

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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"

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
   
    # # For HE-AAC
    path_wb = ["/home/woongjib/Projects/USAC44_mono_48k"]
    path_nb = ["/home/woongjib/Projects/USAC44_mono_48k_HEAAC16_Crop"]
    
    # USACMonoDataset
    # path_wb = ["/home/woongjib/Projects/USAC44_mono_48k"]
    # path_nb = ["/home/woongjib/Projects/USAC44_mono_48k_HEAAC16_LPF_Crop"]
    dataset = CustomDataset(path_dir_nb=path_nb, path_dir_wb=path_wb, seg_len=1, mode="val", high_index=31)

    ################### Model
    # # TFiLM 64
    # model = SEANet_TFiLM(kmeans_model_path="/home/woongjib/Projects/BESSL__/kmeans/K64_MAE.pkl")
    # model = load_model(model, "/home/woongjib/Projects/BESSL__/ckpts/ckpt_BESSL_AAC13.5/ckpt_K64/epoch_41_lsdH_0.441.pth")
    
    # # No K
    # model = SEANet_TFiLM_nok(kmeans_model_path=None)
    # model = load_model(model, "/home/woongjib/Projects/BESSL__/ckpts/ckpt_BESSL_AAC13.5/ckpt_nok/epoch_13_lsdH_0.430.pth")
    
    # No K Modified FT
    # model = SEANet_TFiLM_nokmod(kmeans_model_path=None, in_channels=64)
    # model = load_model(model, "/home/woongjib/Projects/BESSL__/ckpts/ckpt_D64FT_T4/epoch_10_lsdH_0.426.pth")
    
    # No K mod
    # model = SEANet_TFiLM_nokmod(kmeans_model_path=None, in_channels=64)
    # model = load_model(model, "/home/woongjib/Projects/BESSL__/ckpts/ckpt_D64m_data/epoch_44_lsdH_0.349.pth")

    # No K with different min_dim
    # model = SEANet_TFiLM_nokmod(min_dim=16, visualize=False, in_channels=32)
    # model = load_model(model, "/home/woongjib/Projects/BESSL__/ckpts/ckpt_D32_md16_extend/epoch_10_lsdH_0.320.pth")
    # output_dir = "outputs/D32md16_extend_"

    # md 20
    # model = SEANet_TFiLM_nokmod(min_dim=8, visualize=False, in_channels=64)
    # model = load_model(model, "/home/woongjib/Projects/BESSL__/ckpts/ckpt_D64_md8_extend/epoch_34_lsdH_0.348.pth")
    # output_dir = "outputs/D64min8_extend"

    # Blind
    # model = SEANet()
    # model = load_model(model, "/home/woongjib/Projects/BESSL__/ckpt_baseline/epoch_26_lsdH_0.550.pth")

    # RVQ - EXP3
    # model = SEANet_TFiLM_RVQ(in_channels=32, min_dim=16)
    # model = load_model(model, "/home/woongjib/Projects/BESSL__/ckpts/EXP3/epoch_5_lsdH_0.575.pth")
    # output_dir = "outputs/EXP3"
    rvq = False

    output_dir = 'outputs/temptemp'
    # os dir
    os.makedirs(output_dir, exist_ok=True)

    # Random
    torch.manual_seed(42)
    np.random.seed(42)
    datasetlen = len(dataset)
    # NUMSAMPLES = 8
    # indices = torch.randperm(len(dataset))[:NUMSAMPLES]  # 랜덤하게 20개 샘플 추출
    indices = torch.randperm(len(dataset))  # 랜덤하게 20개 샘플 추출

    print(indices)
    lsd_list = []
    lsd_highlist = []
    lsd_highlist2 = []

    import warnings
    warnings.filterwarnings("ignore", message=".*nperseg = .*")
    warnings.filterwarnings("ignore", message=".*cudnnException.*")

    # pbar = tqdm(indices, dynamic_ncols=False)
    for idx in indices:
        idx = idx.item()
        wb, nb, spec, name, label = dataset[idx]
        print(f"Processing: {name}", end="\r")

        # with torch.no_grad():
        #     if not rvq:
        #         recon = model(nb.to(DEVICE), spec.to(DEVICE)).detach()
        #     else:
        #         recon,_,_ = model(nb.to(DEVICE), spec.to(DEVICE))

        recon = nb
        
        # LSD 계산
        lsd = lsd_batch(wb.to('cpu').numpy(), recon.to('cpu').numpy(), fs=48000) 
        print(lsd)
        lsd_list.append(lsd)        
        # LSD 계산
        lsd_high = lsd_batch(wb.to('cpu').numpy(), recon.to('cpu').numpy(), fs=48000, start=4500, cutoff_freq=12000)
        lsd_highlist.append(lsd_high)

        lsd_high2 = lsd_batch(wb.to('cpu').numpy(), recon.to('cpu').numpy(), fs=48000, start=0, cutoff_freq=4500)
        lsd_highlist2.append(lsd_high2)

        # draw_spec을 사용해 스펙트로그램을 생성하고 저장
        wb_spec = draw_spec(wb.cpu().squeeze().numpy(), sr=48000, return_fig=False, vmin=-50, vmax=40)
        nb_spec = draw_spec(nb.cpu().squeeze().numpy(), sr=48000, return_fig=False, vmin=-50, vmax=40)
        recon_spec = draw_spec(recon.cpu().squeeze().numpy(), sr=48000, return_fig=False, vmin=-50, vmax=40)

        # 하나의 큰 그림으로 합쳐서 저장
        visualize_combined_spectrogram(wb_spec, nb_spec, recon_spec, index=idx, name=name, output_dir=output_dir, vmin=-50, vmax=40,
                                       save_separate=False, plot_colorbar=False)

        from utils import lpf
        recon_lpf = lpf(recon.cpu().squeeze(), sr=48000, cutoff=4500) # lpf
        recon_lpf2 = lpf(recon.cpu().squeeze(), sr=48000, cutoff=12000) # lpf
        gt_lpf = lpf(wb.cpu().squeeze(), sr=48000, cutoff=4500) # lpf
        
        # sf.write(f"{output_dir}/{name}_lpf.wav", recon_lpf.squeeze(), format="WAV", samplerate=48000)
        # sf.write(f"{output_dir}/{name}_12lpf.wav", recon_lpf2.squeeze(), format="WAV", samplerate=48000)
        # sf.write(f"{output_dir}/{name}_gtlpf.wav", gt_lpf.squeeze(), format="WAV", samplerate=48000)
        # sf.write(f"{output_dir}/{name}.wav", recon.cpu().squeeze(), format="WAV", samplerate=48000)
        # sf.write(f"{output_dir}/{name}_gt.wav", wb.cpu().squeeze(), format="WAV", samplerate=48000)
        # sf.write(f"{output_dir}/{name}_nb.wav", nb.cpu().squeeze(), format="WAV", samplerate=48000)
    
    # LSD 및 LSD High 평균 계산 및 출력
    average_lsd = sum(lsd_list) / len(lsd_list)
    average_lsd_high = sum(lsd_highlist) / len(lsd_highlist)
    average_lsd_high_high = sum(lsd_highlist2) / len(lsd_highlist2)

    print(f"\nAverage LSD for the 20 samples: {average_lsd:.4f}")
    print(f"Average LSD High for the 20 samples: {average_lsd_high:.4f}")

    log_file = f"{output_dir}/log.txt"
    with open(log_file, 'a') as f:
        f.write(f"Average LSD: {average_lsd:.4f}\n")
        f.write(f"Average LSD High: {average_lsd_high:.4f}\n")
        f.write(f"Average LSD low: {average_lsd_high_high:.4f}\n")

if __name__ == "__main__":
    main()
