import torch
import yaml
# import mir_eval
import gc
import warnings
from utils import *

from tqdm import tqdm
import wandb
from pesq import pesq
from pystoi import stoi
import random
from datetime import datetime
import argparse
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
# from SEANet_v2 import SEANet_ver2
from MelGAN import Discriminator_MelGAN
from MBSTFTD import MultiBandSTFTDiscriminator

from dataset import CustomDataset

from models.SEANet_TFiLM import SEANet_TFiLM
from models.SEANet_TFiLM_nok_modified import SEANet_TFiLM as SEANet_TFiLM_nokmod
from models.SEANet_TFiLM_RVQ import SEANet_TFiLM as SEANet_TFiLM_RVQ
from models.SEANet import SEANet

DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"DEVICE: {DEVICE}")

## Dictionary to store all models and information
TPARAMS = {}
NOTES = 'BESSL_K256_local'
START_DATE = NOTES +'_' + datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_MAP = {
    "SEANet_TFiLM": SEANet_TFiLM,
    "SEANet": SEANet,
    "SEANet_TFiLM_nokmod": SEANet_TFiLM_nokmod,
    "SEANet_TFiLM_RVQ": SEANet_TFiLM_RVQ,
}

def parse_args():
    parser = argparse.ArgumentParser(description="BESSL Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    return args

#########################
def compute_loss(discriminator, bwe, hr, params, commitment_loss, codebook_loss):
    # Compute generator loss
    loss_GAN, loss_FM, loss_mel = discriminator.loss_G(bwe, hr)
    loss_G = (
        params['lambda_mel_loss'] * loss_mel +
        params['lambda_fm_loss'] * loss_FM +
        params['lambda_adv_loss'] * loss_GAN + 
        params['lambda_commitment_loss'] * commitment_loss + 
        params['lambda_codebook_loss'] * codebook_loss
    )
    return loss_G, loss_GAN, loss_FM, loss_mel


def train_step(train_parameters):
    train_parameters['generator'].train()
    train_parameters['discriminator'].train()
    result = {
        'loss_G_total': 0, 'loss_D': 0,
        'FM_loss': 0, 'GAN_loss': 0, 'Mel_loss': 0,
        'codebook_loss': 0, 'commitment_loss': 0,
    }

    train_bar = tqdm(train_parameters['train_dataloader'], desc="Train", position=1, leave=False, disable=False)

    # Train DataLoader Loop in one epoch
    for hr, lr, cond, _, _ in train_bar:
        lr, hr, cond = lr.to(DEVICE), hr.to(DEVICE), cond.to(DEVICE)
        if train_parameters['lambda_codebook_loss'] != 0:
            bwe, commitment_loss, codebook_loss = train_parameters['generator'](lr, cond)
        else:
            bwe = train_parameters['generator'](lr, cond)
            commitment_loss, codebook_loss = 0, 0
            
        # Train Generator     
        loss_G, loss_GAN, loss_FM, loss_mel = compute_loss(train_parameters['discriminator'], bwe, hr, train_parameters, commitment_loss, codebook_loss)
        # loss_G = loss_GAN + 100*loss_FM + loss_mel
        train_parameters['optim_G'].zero_grad()
        loss_G.mean().backward()
        train_parameters['optim_G'].step()

        # Train Discriminator
        train_parameters['optim_D'].zero_grad()
        loss_D = train_parameters['discriminator'].loss_D(bwe, hr)
        loss_D.mean().backward()
        train_parameters['optim_D'].step()
        
        # Log
        result['loss_G_total'] += loss_G.detach().item()
        result['loss_D'] += loss_D.detach().item()
        result['GAN_loss'] += loss_GAN.detach().item()
        result['FM_loss'] += loss_FM.detach().item()  # FM loss 추가
        result['Mel_loss'] += loss_mel.detach().item()  # Mel loss 추가

        if train_parameters['lambda_codebook_loss'] != 0:
                result['commitment_loss']  += commitment_loss.detach().item()
                result['codebook_loss']  += codebook_loss.detach().item()
        else: 
            pass

        train_bar.set_postfix({
            # 'Loss G': f'{loss_G.item():.2f}',
            # 'Loss D': f'{loss_D.item():.2f}',
            'CB Loss': f'{codebook_loss.detach().item():.2f}',
            'FM Loss': f'{loss_FM.item():.2f}',
            'Mel Loss': f'{loss_mel.item():.2f}'
        })

    train_bar.close()
    for key in result.keys():
        result[key] /= len(train_parameters['train_dataloader'])

    return result

def test_step(test_parameters, store_lr_hr=False):
    test_parameters['generator'].eval()
    test_parameters['discriminator'].eval()
    
    result = {
        'loss_G_total': 0, 'loss_D': 0,
        'FM_loss': 0, 'GAN_loss': 0, 'Mel_loss': 0,
        'codebook_loss': 0, 'commitment_loss': 0,
        'LSD': 0, 'LSD_H': 0
    }

    test_bar = tqdm(test_parameters['val_dataloader'], desc='Validation', position=1, leave=False, disable=False)

    # Test DataLoader Loop
    with torch.no_grad():
        for i, (hr, lr, cond, _, _) in enumerate(test_bar):
            lr, hr, cond = lr.to(DEVICE), hr.to(DEVICE), cond.to(DEVICE)
            
            if test_parameters['lambda_codebook_loss'] != 0:
                bwe, commitment_loss, codebook_loss = test_parameters['generator'](lr, cond)
            else:
                bwe = test_parameters['generator'](lr, cond)
                commitment_loss, codebook_loss = 0, 0

            loss_G, loss_GAN, loss_FM, loss_mel = compute_loss(test_parameters['discriminator'], bwe, hr, test_parameters, commitment_loss, codebook_loss)
            loss_D = test_parameters['discriminator'].loss_D(bwe, hr)
            
            # Log
            result['loss_G_total'] += loss_G.item()
            result['loss_D'] += loss_D.item()
            result['GAN_loss'] += loss_GAN.item()
            result['FM_loss'] += loss_FM.item()
            result['Mel_loss'] += loss_mel.item()
            result['commitment_loss'] += commitment_loss.item() if commitment_loss else 0
            result['codebook_loss'] += codebook_loss.item() if codebook_loss else 0

            # Compute metric
            batch_lsd = lsd_batch(x_batch=hr.cpu(), y_batch=bwe.cpu(), fs=48000, )
            batch_lsd_h = lsd_batch(x_batch=hr.cpu(), y_batch=bwe.cpu(), fs=48000, start=4500, cutoff_freq=24000)
            result['LSD'] += batch_lsd
            result['LSD_H'] += batch_lsd_h

            test_bar.set_postfix({
                # 'Loss G': f'{loss_G.item():.2f}',
                # 'Loss D': f'{loss_D.item():.2f}',
                'FM Loss': f'{loss_FM.item():.2f}',
                'Mel Loss': f'{loss_mel.item():.2f}',
                'LSD_H': f'{batch_lsd_h:.2f}'
            })

            # Optional: Store spectrograms for specific indices
            if i in [3, 9, 15, 23]: # 15 10 22
                result[f'audio_bwe_{i}'] = bwe.squeeze().cpu().numpy()
                result[f'audio_recon_{i}'] = draw_spec(bwe.squeeze().cpu().numpy(),win_len=2048, sr=48000, use_colorbar=False, hop_len=1024, return_fig=True)

                if store_lr_hr:
                    result[f'audio_hr_{i}'] = hr.squeeze().cpu().numpy()
                    # result[f'audio_input_{i}'] = draw_spec(lr.squeeze().cpu().numpy(),win_len=2048, sr=48000, use_colorbar=False, hop_len=1024, return_fig=True)
                    result[f'audio_target_{i}'] = draw_spec(hr.squeeze().cpu().numpy(),win_len=2048, sr=48000, use_colorbar=False, hop_len=1024, return_fig=True)

            del lr, hr, bwe
            # gc.collect()
        test_bar.close()
        
        for key, value in result.items():
            if isinstance(value, (int, float)):
                if key not in ['LSD', 'LSD_H']:
                    result[key] /= len(test_parameters['val_dataloader'])
        result['LSD'] /= len(test_parameters['val_dataloader'])
        result['LSD_H'] /= len(test_parameters['val_dataloader'])
        
        return result

def main():
    ################ Read Config Files
    torch.manual_seed(42)
    random.seed(42)
    
    args = parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    print(START_DATE)
    
    wandb.init(project='BESSL_p3',
           entity='woongzip1',
           config=config,
           name=config['run_name'],
           # mode='disabled',
           notes=config['run_name'])

    train_dataset = CustomDataset(path_dir_nb=config['dataset']['nb_train'], 
                                  path_dir_wb=config['dataset']['wb_train'], seg_len=config['dataset']['seg_len'], mode="train", enhance=config['dataset']['enhance'])
    test_dataset = CustomDataset(path_dir_nb=config['dataset']['nb_test'], 
                                 path_dir_wb=config['dataset']['wb_test'], seg_len=config['dataset']['seg_len'], mode="val", high_index=31)

    # Split train into train/test
    if config['dataset']['ratio'] < 1:
        train_size = int(config['dataset']['ratio'] * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, _ = random_split(train_dataset, [train_size,test_size])
        
    # Small dataset
    # train_dataset = SmallDataset(train_dataset, 100)
    # test_dataset = SmallDataset(test_dataset, 3000) 

    print(f'Train Dataset size: {len(train_dataset)} | Validation Dataset size: {len(test_dataset)}\n')
    TPARAMS['train_dataloader'] = DataLoader(train_dataset, batch_size = config['dataset']['batch_size'], 
                                            # sampler = train_sampler,
                                            num_workers=config['dataset']['num_workers'], 
                                            prefetch_factor=2, persistent_workers=True, pin_memory=True
                                            )
    TPARAMS['val_dataloader'] = DataLoader(test_dataset, batch_size = 1, shuffle=False,
                                            num_workers=config['dataset']['num_workers'], 
                                            prefetch_factor=2, persistent_workers=True, pin_memory=True
                                            )

    print(f"DataLoader Loaded!: {len(TPARAMS['train_dataloader'])} | {len(TPARAMS['val_dataloader'])}")

    ################ Load Models
    warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated")
    warnings.filterwarnings("ignore", category=UserWarning, message="At least one mel filterbank has")
    warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated")
    warnings.filterwarnings("ignore", message=".*cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*")

    TPARAMS['generator'] = prepare_generator(config, MODEL_MAP)
    
    # Prepare discriminator
    disc_type = config['model']['discriminator']
    if disc_type == "MSD":
        TPARAMS['discriminator'] = Discriminator_MelGAN()
    elif disc_type == "MBSTFTD":
        discriminator_config = config['model']['MultiBandSTFTDiscriminator_config']
        TPARAMS['discriminator'] = MultiBandSTFTDiscriminator(
            C=discriminator_config['C'],
            n_fft_list=discriminator_config['n_fft_list'],
            hop_len_list=discriminator_config['hop_len_list'],
            bands=discriminator_config['band_split_ratio'],
            ms_mel_loss_config=config['model']['ms_mel_loss_config']
        )
        TPARAMS['lambda_mel_loss'] = config['loss']['lambda_mel_loss']
        TPARAMS['lambda_fm_loss'] = config['loss']['lambda_fm_loss']
        TPARAMS['lambda_adv_loss'] = config['loss']['lambda_adv_loss']
        TPARAMS['lambda_commitment_loss'] = config['loss']['lambda_commitment_loss']
        TPARAMS['lambda_codebook_loss'] = config['loss']['lambda_codebook_loss']
    else:
        raise ValueError(f"Unsupported discriminator type: {disc_type}")
    print("########################################")

    ################ Load Optimizers
    TPARAMS['optim_G'] = torch.optim.Adam(TPARAMS['generator'].parameters(), lr=config['optim']['learning_rate'], 
                                          betas=(config['optim']['B1'],config['optim']['B2']))
    TPARAMS['optim_D'] = torch.optim.Adam(TPARAMS['discriminator'].parameters(), config['optim']['learning_rate'], 
                                          betas=(config['optim']['B1'],config['optim']['B2']))
    TPARAMS['scheduler_G'] = optim.lr_scheduler.ExponentialLR(
                            TPARAMS['optim_G'], gamma=config['optim']['scheduler_gamma'])
    TPARAMS['scheduler_D'] = optim.lr_scheduler.ExponentialLR(
                            TPARAMS['optim_D'], gamma=config['optim']['scheduler_gamma'])
    ################ Load Checkpoint if available
    start_epoch = 0

    if config['train']['ckpt']:
        checkpoint_path = config['train']['ckpt_path']
        if os.path.isfile(checkpoint_path):
            start_epoch, best_lsdh = load_checkpoint(TPARAMS['generator'], TPARAMS['discriminator'],
                                                     TPARAMS['optim_G'], TPARAMS['optim_D'], DEVICE, checkpoint_path)
        else:
            print(f"Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")

    ################ Training Loop
    print(f'Train Start! Start Epoch:{start_epoch+1}')
    torch.manual_seed(42) # 42 seed
    BAR = tqdm(range(start_epoch+1, config['train']['max_epochs'] + 1), position=0, leave=True)
    TPARAMS['generator'].to(DEVICE)
    TPARAMS['discriminator'].to(DEVICE)
    
    best_lsdh = 1e10
    store_lr_hr = True # flag
    for epoch in BAR:
        # set_seed(epoch+42)
        TPARAMS['current_epoch'] = epoch
        train_result = train_step(TPARAMS)
        wandb_log(train_result, epoch, 'train')

        if epoch % config['train']['val_epoch'] == 0:
            # Validation step
            val_result = test_step(TPARAMS, store_lr_hr)
            wandb_log(val_result, epoch, 'val')

            if store_lr_hr:
                store_lr_hr = False
            # Best LSD Model
            if val_result['LSD_H'] < best_lsdh:
                # best_lsdh = val_result['LSD_H'] # save all time
                save_checkpoint(TPARAMS['generator'], TPARAMS['discriminator'], TPARAMS['optim_G'], 
                                TPARAMS['optim_D'], epoch, val_result['LSD_H'], config)

            TPARAMS['scheduler_G'].step()
            TPARAMS['scheduler_D'].step()

            desc = (f"Epoch [{epoch}/{config['train']['max_epochs']}] "
                    f"Loss G: {train_result['loss_G_total']:.2f}, "
                    f"FM Loss: {train_result['FM_loss']:.2f}, "
                    f"Mel Loss: {train_result['Mel_loss']:.2f}, "
                    f"Loss D: {train_result['loss_D']:.2f}, "
                    f"Val Loss G: {val_result['loss_G_total']:.2f}, "
                    f"Val FM Loss: {val_result['FM_loss']:.2f}, "
                    f"Val Mel Loss: {val_result['Mel_loss']:.2f}, "
                    f"Val Loss D: {val_result['loss_D']:.2f}, "
                    f"LSD: {val_result['LSD']:.2f}"
                    )
            
        else:
            desc = (f"Epoch [{epoch}/{config['train']['max_epochs']}] "
                    f"Loss G: {train_result['loss_G_total']:.2f}, "
                    f"FM Loss: {train_result['FM_loss']:.2f}, "
                    f"Mel Loss: {train_result['Mel_loss']:.2f}, "
                    f"Loss D: {train_result['loss_D']:.2f}"
                    )
        BAR.set_description(desc)

    gc.collect()


if __name__ == "__main__":
    main()