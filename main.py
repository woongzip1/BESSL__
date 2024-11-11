import torch
import yaml
import gc
import warnings
import random
from utils import *
from tqdm import tqdm
import wandb
from datetime import datetime
import argparse
from torch.utils.data import DataLoader, random_split
from trainer import Trainer 

# from MelGAN import Discriminator_MelGAN
# from MBSTFTD import MultiBandSTFTDiscriminator
from models.SEANet_TFiLM import SEANet_TFiLM
from models.SEANet_TFiLM_nok_modified import SEANet_TFiLM as SEANet_TFiLM_nokmod
from models.SEANet_TFiLM_RVQ import SEANet_TFiLM as SEANet_TFiLM_RVQ
from models.SEANet import SEANet
from dataset import CustomDataset

DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"DEVICE: {DEVICE}")

# Dictionary for models and configuration
MODEL_MAP = {
    "SEANet_TFiLM": SEANet_TFiLM,
    "SEANet": SEANet,
    "SEANet_TFiLM_nokmod": SEANet_TFiLM_nokmod,
    "SEANet_TFiLM_RVQ": SEANet_TFiLM_RVQ,
}

def parse_args():
    parser = argparse.ArgumentParser(description="BESSL Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main(if_log_to_wandb):
    args = parse_args()
    config = load_config(args.config)
    torch.manual_seed(42)
    random.seed(42)
    
    if if_log_to_wandb: # if log
        wandb.init(project='BESSL_p4', entity='woongzip1', config=config, name=config['run_name'], notes=config['run_name'])
    
    # Load datasets
    train_dataset = CustomDataset(path_dir_nb=config['dataset']['nb_train'], path_dir_wb=config['dataset']['wb_train'], 
                                  seg_len=config['dataset']['seg_len'], mode="train", enhance=config['dataset']['enhance'])
    val_dataset = CustomDataset(path_dir_nb=config['dataset']['nb_test'], path_dir_wb=config['dataset']['wb_test'], 
                                seg_len=config['dataset']['seg_len'], mode="val", high_index=31)
    
    # Optionally split train data
    if config['dataset']['ratio'] < 1:
        train_size = int(config['dataset']['ratio'] * len(train_dataset))
        _, train_dataset = random_split(train_dataset, [len(train_dataset) - train_size, train_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['dataset']['batch_size'],
                            #   batch_size=1,
                              num_workers=config['dataset']['num_workers'],
                              prefetch_factor=2, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config['dataset']['num_workers'], 
                            prefetch_factor=2, persistent_workers=True, pin_memory=True)
    
    # Model selection
    generator = prepare_generator(config, MODEL_MAP)
    discriminator = prepare_discriminator(config)

    # Optimizers
    optim_G = torch.optim.Adam(generator.parameters(), lr=config['optim']['learning_rate'], betas=(config['optim']['B1'], config['optim']['B2']))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=config['optim']['learning_rate'], betas=(config['optim']['B1'], config['optim']['B2']))

    # Trainer initialization
    trainer = Trainer(generator, discriminator, train_loader, val_loader, optim_G, optim_D, config, DEVICE, if_log_step=False, if_log_to_wandb=if_log_to_wandb)
    
    # Train
    trainer.train(num_epochs=config['train']['max_epochs'])

if __name__ == "__main__":
    main(if_log_to_wandb=False)
