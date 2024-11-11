import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
import os
import gc
from utils import wandb_log, draw_spec, lsd_batch

from matplotlib import pyplot as plt
import numpy as np
from loss_ import LossCalculator
from loss import loss_backward


class Trainer:
    def __init__(self, generator, discriminator, train_loader, val_loader, optim_G, optim_D, config, device, if_log_step=False, if_log_to_wandb=True):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.config = config
        self.device = device
        self.if_log_step = if_log_step
        self.if_log_to_wandb = if_log_to_wandb
        
        self.loss_calculator = LossCalculator(config, self.discriminator)
        self.lambda_commitment_loss = config['loss']['lambda_commitment_loss']
        self.lambda_codebook_loss = config['loss']['lambda_codebook_loss']
        self.lambda_mel_loss = config['loss']['lambda_mel_loss']
        self.lambda_fm_loss = config['loss']['lambda_fm_loss']
        self.lambda_adv_loss = config['loss']['lambda_adv_loss']

    def unified_log(self, log_dict, stage, epoch=None):
        """
        Unified logging function for wandb that handles different data types.

        Args:
            log_dict (dict): Dictionary containing log keys and values.
            stage (str): Training/validation stage ('train'/'val').
            epoch (int, optional): Epoch number for logging. Defaults to None.
        """
        if self.if_log_to_wandb:
            for key, value in log_dict.items():
                if isinstance(value, torch.Tensor):
                    item = value.cpu().detach().numpy()
                else:
                    item = value
                
                try:
                    if isinstance(item, float) or isinstance(item, int):
                        log = item
                    elif isinstance(item, plt.Figure):  # Handling Matplotlib figures if applicable
                        log = wandb.Image(item)
                        plt.close(item)
                    elif isinstance(item, np.ndarray) and item.ndim in [2, 3]:  # Assuming this is an image
                        log = wandb.Image(item, caption=f"{stage.capitalize()} {key.capitalize()} Epoch {epoch}")
                    elif isinstance(item, np.ndarray) and item.ndim == 1:  # Assuming this is an audio signal
                        log = wandb.Audio(item, sample_rate=48000, caption=f"{stage.capitalize()} {key.capitalize()} Epoch {epoch}")
                    else:
                        log = item  # Default logging for non-special cases
                except Exception as e:
                    print(f"Failed to log {key}: {e}")
                    log = item  # Log as-is if an exception occurs

                wandb.log({
                    f"{stage}/{key}": log,
                }, step=epoch if epoch is not None else None)

    # Temporarily unavailable
    # def log_results(self, result, stage, epoch=None):
    #     if epoch is not None:
    #         for key, value in result.items():
    #             wandb.log({f"{stage}/{key}": value}, step=epoch)
    #     else:
    #         wandb.log(result)
   
    def _forward_pass(self, lr, cond):
        if self.lambda_codebook_loss != 0:
            return self.generator(lr, cond)
        return self.generator(lr, cond), 0, 0

    def train_step(self, hr, lr, cond):
        self.generator.train()
        self.discriminator.train()
       
        bwe, commitment_loss, codebook_loss = self._forward_pass(lr, cond)
        loss_G, ms_mel_loss_value, g_loss_dict, g_loss_report = self.loss_calculator.compute_generator_loss(hr, bwe, commitment_loss, codebook_loss)  
        
        # Train generator
        self.optim_G.zero_grad()
        loss_dict_g = {
            'adv_g': self.lambda_adv_loss * g_loss_dict.get('adv_g', 0),
            'fm': self.lambda_fm_loss * g_loss_dict.get('fm', 0),
            'ms_mel_loss': self.lambda_mel_loss * ms_mel_loss_value,
            'commitment_loss': self.lambda_commitment_loss * commitment_loss,
            'codebook_loss': self.lambda_codebook_loss * codebook_loss
        }
        # loss_backward(loss_dict=loss_dict_g, loss_name_list=list(loss_dict_g.keys()), loss_weight_dict={name: 1 for name in loss_dict_g}, balancer=None)
        loss_G.backward() # mean?
        self.optim_G.step()

        # Train discriminator
        loss_D, d_loss_dict, d_loss_report = self.loss_calculator.compute_discriminator_loss(hr, bwe)
        self.optim_D.zero_grad()
        loss_D.backward()
        loss_dict_d = {'adv_d': loss_D}
        # loss_backward(loss_dict=loss_dict_d, loss_name_list=list(loss_dict_d.keys()), loss_weight_dict={name: 1 for name in loss_dict_d}, balancer=None)
        self.optim_D.step()

        # print(ms_mel_loss_value.item())
        step_result = {
            'loss_G': loss_G.item(),
            'ms_mel_loss': ms_mel_loss_value.item(),
            # 'loss_D': loss_D.item(),
            **{f'G_{k}': v.item() if isinstance(v, torch.Tensor) else v for k, v in g_loss_dict.items()},
            **{f'D_{k}': v.item() if isinstance(v, torch.Tensor) else v for k, v in d_loss_dict.items()},
            **{f'G_report_{k}': v for k, v in g_loss_report.items()},  
            **{f'D_report_{k}': v for k, v in d_loss_report.items()},  
            'commitment_loss': commitment_loss.item() if commitment_loss else 0,
            'codebook_loss': codebook_loss.item() if codebook_loss else 0
            }
        import pdb
        # pdb.set_trace()
        if self.if_log_step:
            self.unified_log(step_result, 'train')
        return step_result

    def validate(self, epoch):
        self.generator.eval()
        self.discriminator.eval()
        result = {key: 0 for key in ['loss_G', 'loss_D', 'ms_mel_loss', 'commitment_loss', 'codebook_loss', 'LSD', 'LSD_H']}

        with torch.no_grad():
            for i, (hr, lr, cond, _, _) in enumerate(tqdm(self.val_loader, desc='Validation')):
                lr, hr, cond = lr.to(self.device), hr.to(self.device), cond.to(self.device)
                bwe, commitment_loss, codebook_loss = self._forward_pass(lr, cond)

                loss_G, ms_mel_loss_value, g_loss_dict, g_loss_report = self.loss_calculator.compute_generator_loss(hr, bwe, commitment_loss, codebook_loss)
                loss_D, d_loss_dict, d_loss_report = self.loss_calculator.compute_discriminator_loss(hr, bwe)

                # Compute LSD and LSD_H metrics
                batch_lsd = lsd_batch(x_batch=hr.cpu(), y_batch=bwe.cpu(), fs=48000)
                batch_lsd_h = lsd_batch(x_batch=hr.cpu(), y_batch=bwe.cpu(), fs=48000, start=4500, cutoff_freq=24000)
                result['LSD'] += batch_lsd
                result['LSD_H'] += batch_lsd_h

                # Aggregate results
                result['loss_G'] += loss_G.item()
                result['ms_mel_loss'] += ms_mel_loss_value.item()
                result['loss_D'] += loss_D.item()
                result['commitment_loss'] += commitment_loss.item() if commitment_loss else 0
                result['codebook_loss'] += codebook_loss.item() if codebook_loss else 0

                # # Optionally log reports
                # if i == 0:  
                #     self.unified_log({
                #         **{f'G_report_{k}': v for k, v in g_loss_report.items()},
                #         **{f'D_report_{k}': v for k, v in d_loss_report.items()}
                #     }, 'val', epoch)

                # Example of logging additional data for validation
                if i in [0,1,2]:  # Log for the first example as an illustration
                    self.unified_log({
                        f'audio_bwe_{i}': bwe.squeeze().cpu().numpy(),
                        f'audio_hr_{i}': hr.squeeze().cpu().numpy(),
                        f'spec_bwe_{i}': draw_spec(bwe.squeeze().cpu().numpy(),win_len=2048, sr=48000, use_colorbar=False, hop_len=1024, return_fig=True),
                        f'spec_lr_{i}': draw_spec(lr.squeeze().cpu().numpy(),win_len=2048, sr=48000, use_colorbar=False, hop_len=1024, return_fig=True),
                        f'spec_hr_{i}': draw_spec(hr.squeeze().cpu().numpy(),win_len=2048, sr=48000, use_colorbar=False, hop_len=1024, return_fig=True),
                    }, 'val', epoch)

        for key in result:
            result[key] /= len(self.val_loader)
        return result

    def save_checkpoint(self, epoch, val_result, save_path):
        os.makedirs(save_path, exist_ok=True)
        filename = f"epoch_{epoch}_lsdh_{val_result['LSD_H']:.4f}.pth"
        save_path = os.path.join(save_path, filename)
        
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optim_G_state_dict': self.optim_G.state_dict(),
            'optim_D_state_dict': self.optim_D.state_dict(),
        }, save_path)

    def train(self, num_epochs):
        best_lsdh = float('inf')
        # torch.autograd.set_detect_anomaly(True)

        for epoch in range(1,num_epochs+1):
            train_result={}
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{num_epochs}')

            for hr, lr, cond, _, _ in progress_bar:
                hr, lr, cond = hr.to(self.device), lr.to(self.device), cond.to(self.device)
                step_result = self.train_step(hr, lr, cond)

                # Sum up step losses for logging purposes
                for key, value in step_result.items():
                    train_result[key] = train_result.get(key, 0) + value

                # Update tqdm description with specific losses
                progress_bar.set_postfix({
                    'loss_G': step_result.get('loss_G', 0),
                    'mel_loss': step_result.get('ms_mel_loss', 0)
                })
                    
            for key in train_result: # mean epoch loss
                train_result[key] /= len(self.train_loader)
            print(train_result)
            # print(len(self.train_loader))
            self.unified_log(train_result, 'train', epoch=epoch)

            val_result = self.validate(epoch)
            # print(val_result)
            self.unified_log(val_result, 'val', epoch=epoch)
            
            if val_result['LSD_H'] < best_lsdh:
                # best_lsdh = val_result['LSD_H']
                print(f"Ckpt saved at {self.config['train']['ckpt_save_dir']} with LSDH {val_result['LSD_H']:.4f}")
                self.save_checkpoint(epoch, val_result, save_path=self.config['train']['ckpt_save_dir'])

