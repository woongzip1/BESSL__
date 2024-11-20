import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary
import torch.nn.functional as F
import pickle
from einops import rearrange

from FeatureExtractor.model import AutoEncoder
from FeatureExtractor.model_new import AutoEncoder_new

import pickle
# from AudioMAE.feature_encoder import AudioMAEEncoder

from quantize import ResidualVectorQuantize
# from vector_quantize_pytorch import ResidualVQ

"""
****** Temporal FiLMing ******

n_channels: Number of Conv feature map channels

do not input N x L x C

x: Conv Feature Map (N x C x L)
cond: SSL Condition (N x L/320 x 1024)
        --> N x #FramePatches x F

output:  modulated feature map (N x C x L)

"""
class FiLMLayer(nn.Module):
    def __init__(self, n_channels=10, subband_num=3, visualize=False):
        """ What does melpatch_num do ? """
        super().__init__()
        self.n_channels = n_channels # channels of conv layer
        # 320: subband dim 
        self.subband_num = subband_num
        # 32: output of FeatureReduction layer
        self.film_gen = nn.Linear(32*self.subband_num, 2*n_channels)
        self.visualize = visualize

    def forward(self, x, condition):

        ## Start from -> N C L
        ## Condi -> N L C
        subblock_num = condition.size(1)

        if self.visualize: print(condition.shape, "Cond Shape")
        cond = self.film_gen(condition)
        if self.visualize: print(cond.shape, "FiLM Generated Shape")
        cond = rearrange(cond, 'n l c -> n c l')
        # Extract (ch x subblock_num) gamma and beta 
        gamma = cond[:,:self.n_channels,:].unsqueeze(-1)
        beta = cond[:,self.n_channels:,:].unsqueeze(-1)

        ## Reshape X
        if self.visualize: 
            print("Subblock_num (Frame num):", subblock_num)
            print(x.shape,"-->", end=' ')

        x = rearrange(x, "n c (b t) -> n c b t", b=subblock_num)

        if self.visualize: 
            print(x.shape)
            print(cond.shape, "SSL Projected Shape")
            print(beta.shape, "BETA Shape")
            print(x.shape, "X Shape")

        # Linear Modulation
        x = gamma * x + beta
        x = rearrange(x, 'n c b t -> n c (b t)')
        if self.visualize:
            print(x.shape, "Concat X Shape")
        return x

"""
# x as a conv feature map shape (B x C x L)
x = torch.rand(3, 4, 16000)
cond = torch.rand(3, #frames, #features)
model = FiLMLayer(n_channels=4, visualize=True)
y = model(x,cond)
print(y.shape)
"""

"""
Input Shape: B x Patch x 6144
Output Shape: B x Patch x 256 
"""
class FeatureReduction(nn.Module):
    def __init__(self, subband_num=10, D=512):
        super(FeatureReduction, self).__init__()

        self.subband_num = subband_num
        self.patch_len = D #  8D for feature encoder
        self.layers = nn.ModuleList([nn.Linear(self.patch_len,32) for _ in range(self.subband_num)])
        # 8 Feature Encoder 512x10 -> 32x10 dim
    
    def forward(self, embeddings):
        # embedding: B x Patch x 6144 (5120)
        ## input must be: B x T x (D F)

        outs = []
        for idx, layer in enumerate(self.layers):
            patch_embeddings = embeddings[:, :, idx*self.patch_len:(idx+1)*self.patch_len] # extract per subband
            # print(idx, patch_embeddings.shape)
            out = layer(patch_embeddings)
            outs.append(out)
            # print(out.shape)
        final_output = torch.cat(outs, dim=-1)

        return final_output

""" Total 1.27 M Parameters """
class SEANet_TFiLM(nn.Module):
    
    def __init__(self, min_dim=8, kmeans_model_path=None, visualize=False, 
                 subband_num=27, fe_weight_path=None, train_enc=True, in_channels=16, **kwargs):
        # from AudioMAE.models_mae import AudioMAEEncoder

        super().__init__()
        
        self.visualize = visualize

        # Load Kmeans model
        # with open(kmeans_model_path, 'rb') ass file:
            # self.kmeans = pickle.load(file)
                
        self.min_dim = min_dim
        
        ## Load SSL model
        # self.ssl_model = AutoEncoder()
        self.ssl_model = AutoEncoder_new(in_channels=in_channels)
        ##### load check points and do not freeze the model
        
        # self.ssl_model.load_checkpoint(fe_weight_path)
        self.ssl_model = self.ssl_model.encoder

        # Freeze SSL Parameters            
        for param in self.ssl_model.parameters():
            param.requires_grad = train_enc

        self.rvq = ResidualVectorQuantize(
            input_dim=864,        #
            n_codebooks=10,         # 
            codebook_size=1024,     # 
            codebook_dim=8,       # 
            quantizer_dropout=0.5  # 
        )
        
    #     self.rvq = ResidualVQ(dim=832,                 # Input dimension (same as embedding dimension)
    #     num_quantizers=8,         # Number of codebooks (quantizers)
    #     codebook_size=512,        # Codebook size (512 entries per codebook)
    #     kmeans_init=True,         # Use k-means initialization
    #     kmeans_iters=10,          # Number of k-means iterations for initialization
    #     threshold_ema_dead_code=2 # Replace rarely used codes with new centroids
    # )
        

        # Feature Extracted SSL Layers
        self.subband_num = subband_num
        self.EmbeddingReduction = FeatureReduction(self.subband_num, D=in_channels*8)
        
        self.FiLM_e1 = FiLMLayer(subband_num=self.subband_num, n_channels=self.min_dim*2, visualize=self.visualize)
        self.FiLM_e2 = FiLMLayer(subband_num=self.subband_num, n_channels=self.min_dim*4, visualize=self.visualize)
        self.FiLM_e3 = FiLMLayer(subband_num=self.subband_num, n_channels=self.min_dim*8, visualize=self.visualize)
        self.FiLM_e4 = FiLMLayer(subband_num=self.subband_num, n_channels=self.min_dim*16, visualize=self.visualize)

        self.FiLM_b1 = FiLMLayer(subband_num=self.subband_num, n_channels=self.min_dim*4, visualize=self.visualize)
        self.FiLM_b2 = FiLMLayer(subband_num=self.subband_num, n_channels=self.min_dim*16, visualize=self.visualize)

        self.FiLM_d1 = FiLMLayer(subband_num=self.subband_num, n_channels=self.min_dim*8)
        self.FiLM_d2 = FiLMLayer(subband_num=self.subband_num, n_channels=self.min_dim*4)
        self.FiLM_d3 = FiLMLayer(subband_num=self.subband_num, n_channels=self.min_dim*2)
        self.FiLM_d4 = FiLMLayer(subband_num=self.subband_num, n_channels=self.min_dim)

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

        # print(cond.shape)
        embedding = self.ssl_model(cond)
        embedding = rearrange(embedding, 'b d f t -> b t (d f)') # Bx512x26xT -> BxTx(26*512)
        embedding = self.EmbeddingReduction(embedding) # BxTx(26*512) -> BxTx(26*32)
        
        if self.visualize: 
            print("Input Signal Length:", x.shape[2], " Fragment:", fragment.shape)
            print("Patch len:", patch_len)
            print(embedding.shape, "EMBEDDING SHAPE")
            print(x.shape, "input shape")

        ################## RVQ Module ##################
        embedding = rearrange(embedding, 'b t f -> b f t')
        # print("*** Before VQ***", embedding.shape)
        embedding, codes, latents, commitment_loss, codebook_loss = self.rvq(embedding)
        # print("*** After VQ***", embedding.shape)
        embedding = rearrange(embedding, 'b f t -> b t f')
        ################################################

        ################## Forward ##################
        skip = [x]
        x = self.conv_in(x)
        skip.append(x)

        if self.visualize: 
            print(embedding.shape, "EMBEDDING: B x L x F")
            print(x.shape, "After 1st Conv Feature: B x F x L")

        # Enc
        film_list = [self.FiLM_e1, self.FiLM_e2, self.FiLM_e3, self.FiLM_e4]
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)
            x = film_list[i](x, embedding)
            # print("\t x.shape", x.shape)
            skip.append(x)

        # Bottleneck
        x = self.conv_bottle1(x) 
        x = self.FiLM_b1(x, embedding)
        x = self.conv_bottle2(x) 
        x = self.FiLM_b2(x, embedding)

        # Dec
        skip = skip[::-1]
        film_list_d = [self.FiLM_d1, self.FiLM_d2, self.FiLM_d3, self.FiLM_d4]
        for l in range(len(self.decoder)):
            x = x + skip[l]
            x = self.decoder[l](x)
            x = film_list_d[l](x, embedding)
            # print("\t x.shape", x.shape)
        x = x + skip[4]
        x = self.conv_out(x)
        x = x + skip[5]

        # Length Adjustment: Append the fragment back
        if len(fragment.size()) == 2:
            fragment = fragment.unsqueeze(-2)

        x = torch.cat((x, fragment), dim=-1)

        return x, commitment_loss, codebook_loss

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