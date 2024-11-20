import os
import librosa
import numpy as np
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2, similarity_result_pb2
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, message=".*MessageFactory class is deprecated.*")

class ViSQOL:
    def __init__(self, sample_rate=16000, use_speech_scoring=True, svr_model_name="libsvm_nu_svr_model.txt"):
        # Configuration
        self.config = visqol_config_pb2.VisqolConfig()
        self.config.audio.sample_rate = sample_rate
        self.config.options.use_speech_scoring = use_speech_scoring

        # Model path
        self.svr_model_path = svr_model_name
        # self.config.options.svr_model_path = os.path.join(os.path.dirname(visqol_lib_py.__file__), "model", self.svr_model_path)
        # /libsvm_nu_svr_model.txt
        self.config.options.svr_model_path = "/home/woongjib/anaconda3/envs/env2/lib/python3.12/site-packages/visqol/model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
        # Initialize ViSQOL
        # print(self.config.options.svr_mode_path)
        
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

def calculate_average_visqol_score(gt_dir, bwe_dir, mode='speech'):
    if mode == 'speech':
        visqol = ViSQOL(sample_rate=16000, use_speech_scoring=True, svr_model_name="libsvm_nu_svr_model.txt")
    elif mode == 'audio':
        visqol = ViSQOL(sample_rate=48000, use_speech_scoring=False, svr_model_name="lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite")
        
    gt_files = sorted(os.listdir(gt_dir))
    bwe_files = sorted(os.listdir(bwe_dir))
    
    gt_files = gt_files[:10]
    bwe_files = bwe_files[:10]
    
    if len(gt_files) != len(bwe_files):
        raise ValueError("The number of files in GT and BWE directories do not match.")
    
    mos_lqo_scores = []
    
    for gt_file, bwe_file in tqdm(zip(gt_files, bwe_files), total=len(gt_files)):
        gt_path = os.path.join(gt_dir, gt_file)
        bwe_path = os.path.join(bwe_dir, bwe_file)
        
        ref_audio = visqol.load_audio(gt_path)
        deg_audio = visqol.load_audio(bwe_path)
        
        mos_lqo = visqol.measure(ref_audio, deg_audio)
        mos_lqo_scores.append(mos_lqo)
    
    avg_mos_lqo = np.mean(mos_lqo_scores)
    return avg_mos_lqo, visqol

# Example usage
if __name__ == "__main__":
    gt_directory = "/mnt/hdd/Results/model_EH_baseline/GT"
    # bwe_directory = "/mnt/hdd/Results/model_1/BWE"
    # average_mos_lqo = calculate_average_visqol_score(gt_directory, bwe_directory)
    # print(f'Average MOS-LQO: {average_mos_lqo:.3f}')

gt_directory = "/mnt/hdd/Dataset_BESSL_p2/Dataset_gt/VCTK_CORE_M4a/48k/p225"
core_directory = "/home/woongjib/Projects/Dataset_BESSL/Dataset_core/VCTK_CORE_M4a/48k/p225"

# gt_directory = "/home/woongjib/Projects/Dataset_BESSL/Dataset_gt_crop/VCTK_CORE_M4a/48k/p225"
# core_directory = "/home/woongjib/Projects/Dataset_BESSL/Dataset_core_crop/VCTK_CORE_M4a/48k/p225"
# lpf_directory = "/home/woongjib/Projects/Dataset_BESSL/Dataset_LPF_crop/VCTK_CORE_M4a/48k/p225"
# gt_directory = sorted(os.listdir())
# core_directory = sorted(os.listdir("/home/woongjib/Projects/Dataset_BESSL/Dataset_core_crop/VCTK_CORE_M4a/48k/p225"))
# gt_directory = gt_directory[:10]
# core_directory = core_directory[:10]

print(gt_directory)
print(core_directory)
avg1, visqol = calculate_average_visqol_score(gt_directory, core_directory, mode='audio')
print(avg1)
# avg2 = calculate_average_visqol_score(gt_directory, core_directory)
# avg2 = calculate_average_visqol_score(gt_directory, lpf_directory)
# print(avg1, avg2)