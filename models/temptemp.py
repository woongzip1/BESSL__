import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary
import torch.nn.functional as F
import pickle
from einops import rearrange

from FeatureExtractor.model import AutoEncoder
import pickle
from transformers import HubertModel, AutoProcessor, Wav2Vec2Model, WavLMModel, AutoModel
# from AudioMAE.feature_encoder import AudioMAEEncoder

"""
****** Temporal FiLMing ******

n_channels: Number of Conv feature map channels

do not input N x L x C

x: Conv Feature Map (N x C x L)
cond: SSL Condition (N x L/320 x 1024)
        --> N x #FramePatches x F

output:  modulated feature map (N x C x L)

"""

""" Total 1.27 M Parameters """
class SEANet_TFiLM(nn.Module):
    
    def __init__(self, min_dim=8, kmeans_model_path=None, visualize=False, **kwargs):
        # from AudioMAE.models_mae import AudioMAEEncoder

        super().__init__()
        
        self.visualize = visualize

        # Load Kmeans model
        # with open(kmeans_model_path, 'rb') ass file:
            # self.kmeans = pickle.load(file)

        ## First Conv Layer
        self.conv_in = Conv1d(
            in_channels = 1,
            out_channels = min_dim,
            kernel_size = 7,
            stride = 1
        )

        # Crop factor to match the signal length
        self.downsampling_factor = 2048
        self.encoder = nn.ModuleList([
                                    EncBlock(min_dim*2, 2),
                                    EncBlock(min_dim*4, 2),
                                    EncBlock(min_dim*8, 8),
                                    EncBlock(min_dim*16, 8)                                        
                                    ])
        
        self.conv_bottle1 = Conv1d(
                            in_channels=min_dim*16,
                            out_channels = min_dim*16//4,
                            kernel_size = 7, 
                            stride = 1,
                            )
                        
        self.conv_bottle2 = Conv1d(
                            in_channels=min_dim*16//4,
                            out_channels = min_dim*16,
                            kernel_size = 7,
                            stride = 1,
                            )
        
        self.decoder = nn.ModuleList([
                                    DecBlock(min_dim*8, 8),
                                    DecBlock(min_dim*4, 8),
                                    DecBlock(min_dim*2, 2),
                                    DecBlock(min_dim, 2),
                                    ])
        
        self.conv_out = Conv1d(
            in_channels = min_dim,
            out_channels = 1,
            kernel_size = 7,
            stride = 1,
        )
        
    def length_adjustment(self, x, cond):
        """
        Adjusts the length of the input signals x and cond to be a multiple of the downsampling factor.
        Returns the adjusted signals and the fragment that was trimmed off.
        """

        # cond : 1024 x 128 -> 1 x 1 x 1024 x 128
        # print(x.shape, cond.shape, "X, Cond")
        if cond.dim() == 2:
            cond = cond.unsqueeze(0).unsqueeze(0)
        elif cond.dim() == 3:
            cond = cond.unsqueeze(1)
        fragment = torch.randn(0).to(x.device)
        downsampling_factor = self.downsampling_factor

        if x.dim() == 3:  # N x 1 x L
            sig_len = x.shape[2]
            if sig_len % downsampling_factor != 0:
                new_len = sig_len // downsampling_factor * downsampling_factor
                fragment = x[:, :, new_len:].clone().to(x.device)
                x = x[:, :, :new_len]

        if x.dim() == 2:
            sig_len = x.shape[1]
            if sig_len % downsampling_factor != 0:
                new_len = sig_len // downsampling_factor * downsampling_factor
                fragment = x[:, new_len:].clone().to(x.device)
                x = x[:, :new_len]

        while len(x.size()) < 3:
            x = x.unsqueeze(-2)

        return x, cond, fragment

    def forward(self, x, cond):
        x, cond, fragment = self.length_adjustment(x, cond) # Length Adjustment
        patch_len = x.shape[2] // (2048)

        ################## Forward
        # Conv
        skip = [x]
        x = self.conv_in(x)
        skip.append(x)

        if self.visualize: 
            print(x.shape, "Conv Feature: B x F x L")

        # Enc
        film_list = [self.FiLM_e1, self.FiLM_e2, self.FiLM_e3, self.FiLM_e4]
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)
            # print("\t x.shape", x.shape)
            skip.append(x)

        # Bottleneck
        x = self.conv_bottle1(x) 
        x = self.conv_bottle2(x) 

        # Dec
        skip = skip[::-1]
        film_list_d = [self.FiLM_d1, self.FiLM_d2, self.FiLM_d3, self.FiLM_d4]
        for l in range(len(self.decoder)):
            x = x + skip[l]
            x = self.decoder[l](x)
            # print("\t x.shape", x.shape)
        x = x + skip[4]
        x = self.conv_out(x)
        x = x + skip[5]

        # Length Adjustment: Append the fragment back
        if len(fragment.size()) == 2:
            fragment = fragment.unsqueeze(-2)

        x = torch.cat((x, fragment), dim=-1)

        return x

class EncBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()
        

        self.res_units = nn.ModuleList([
                                    ResUnit(out_channels//2, 1),
                                    ResUnit(out_channels//2, 3),
                                    ResUnit(out_channels//2, 9)                                        
                                    ])
        
        self.conv = nn.Sequential(
                    nn.ELU(),
                    Pad((2 * stride - 1, 0)),
                    nn.Conv1d(in_channels = out_channels//2,
                                       out_channels = out_channels,
                                       kernel_size = 2 * stride,
                                       stride = stride, padding = 0),
                    )  
        
        
    def forward(self, x):
        
        for res_unit in self.res_units:
            x = res_unit(x)
        x = self.conv(x)

        return x
        
    
class DecBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        ## Channel Reduction & Upsampling
        self.conv = ConvTransposed1d(
                                 in_channels = out_channels*2, 
                                 out_channels = out_channels, 
                                 kernel_size = 2*stride, stride= stride,
                                 dilation = 1,
                                 )
        
        
        self.res_units = nn.ModuleList([
                                    ResUnit(out_channels, 1),
                                    ResUnit(out_channels, 3),
                                    ResUnit(out_channels, 9)                                       
                                    ])
               
        self.stride = stride
        

    def forward(self, x):
        x = self.conv(x)
        for res_unit in self.res_units:
            x = res_unit(x)
        return x
    
    
class ResUnit(nn.Module):
    def __init__(self, channels, dilation = 1):
        super().__init__()
        

        self.conv_in = Conv1d(
                                 in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 3, stride= 1,
                                 dilation = dilation,
                                 )
        
        self.conv_out = Conv1d(
                                in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 1, stride= 1,
                                 )
        
        self.conv_shortcuts = Conv1d(
                                in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 1, stride= 1,
                                 )
        
    
        
    def forward(self, x):
        y = self.conv_in(x)
        y = self.conv_out(y)
        x = self.conv_shortcuts(x)
        return x + y
        
    
class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation = 1, groups = 1):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels = in_channels, 
            out_channels = out_channels,
            kernel_size= kernel_size, 
            stride= stride, 
            dilation = dilation,
            groups = groups
        )
        self.conv = nn.utils.weight_norm(self.conv)
        
        self.pad = Pad(((kernel_size-1)*dilation, 0)) 
        self.activation = nn.ELU()
            

    def forward(self, x):

        x = self.pad(x)
        x = self.conv(x)
        x = self.activation(x)
        
        return x

class ConvTransposed1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, dilation = 1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride =stride,
            dilation = dilation
        )
        self.conv = nn.utils.weight_norm(self.conv)
        
        self.activation = nn.ELU()
        self.pad = dilation * (kernel_size - 1) - dilation * (stride - 1)
        
    def forward(self, x):
        x = self.conv(x)
        x = x[..., :-self.pad]
        x = self.activation(x)
        return x
    
class Pad(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
    
    def forward(self, x):
        return F.pad(x, pad=self.pad)   