from matplotlib import pyplot as plt
import torchaudio as ta
import torch
import sys
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
from utils import *
from FeatureExtractor.model import AutoEncoder
from dataset import CustomDataset
from einops import rearrange
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle

def run_kmeans_and_save(embeddings, n_clusters_list, model_name='MAE' ):
    num_patches = 10  # Assuming 8 patches
    patch_size = 512  # Each patch has 768 dimensions

    for n_clusters in n_clusters_list:
        print(f"Running k-means for {n_clusters} clusters...")

        patch_kmeans_models = []  # To store k-means models for each patch

        for idx in range(num_patches):
            # Extract the patch-specific embeddings
            # BT x 5120 -> BT x 512 extract
            patch_embeddings = embeddings[:, idx * patch_size:(idx + 1) * patch_size]
            
            # Initialize and fit k-means for the current patch
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
            patch_embeddings = patch_embeddings.cpu()
            kmeans.fit(patch_embeddings.numpy())

            # Store the k-means model
            patch_kmeans_models.append(kmeans)
            print(f"\n Patch {idx} - Labels shape for {n_clusters} clusters:", kmeans.labels_.shape)

        # Save the k-means models for all patches in one file
        save_path = f"kmeans/K{n_clusters}_{model_name}.pkl"
        with open(save_path, 'wb') as file:
            # Length 8 Kmeans Codebook List
            pickle.dump(patch_kmeans_models, file)

        print(f"\t Saved k-means models for {n_clusters} clusters to {save_path}")

def main():
    ## Feature Extractor
    model = AutoEncoder()
    model.load_checkpoint("weights/epoch_100_mse_0.009.pth")
    model = model.encoder

    ## DataLoader
    path_wb = [
                "/mnt/hdd/Dataset_BESSL/FSD50K_WB_SEGMENT/", 
                "/mnt/hdd/Dataset_BESSL/MUSDB_WB_SEGMENT/", 
                ]
    path_nb = [
                "/mnt/hdd/Dataset_BESSL/FSD50K_NB_SEGMENT/", 
                "/mnt/hdd/Dataset_BESSL/MUSDB_NB_SEGMENT/", 
                ]

    dataset = CustomDataset(path_dir_nb=path_nb, path_dir_wb=path_wb, seg_len=0.9, mode="train")
    dataset = SmallDataset(dataset, 60000) #590000
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=16, prefetch_factor=True)

    embs = []
    ### features : 512 dim for 10 subbands
    ### features : B x 512 x 10 x 28
    ### dataloader -> spectrogram -> Feature Encode -> kmeans
    # B x 5120 x 28(frames) feature
    for wb, nb, spec, name, label in dataloader:
        # extract features: b x 512 x 10 x t
        features = model(spec)
        features = rearrange(features, "b d f t -> (b t) (d f)")
        
        # print(features.shape)    
        embs.append(features)

    data_tensor = torch.cat(embs, dim=0)
    print("data tensor shape:", data_tensor.shape)
    # (B T) x 5120

    run_kmeans_and_save(data_tensor,[1,4,16,64,256])

if __name__ == "__main__":
    main()

