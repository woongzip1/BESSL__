import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from FeatureExtractor.causal_conv import CausalConv2d
from torchinfo import summary

class AutoEncoder_new(nn.Module):
    """ New AutoEncoder for 26 subband feature extraction """
    def __init__(self, in_channels=16):
        super(AutoEncoder_new, self).__init__()
        
        self.encoder = ResNet18(in_channels=in_channels)
        self.decoder = Decoder_new(bottleneck_shape=in_channels * 8)

        self.initialize_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def initialize_weights(self):
        # Iterate through all layers and apply Xavier Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("**** CHECKPOINT LOADED for Feature Encoder! **** ")
                    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, isfinal=False):
        super(BasicBlock, self).__init__()
        self.conv1 = weight_norm(CausalConv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False))
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 =  weight_norm(CausalConv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=False))
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample  # Used for downsampling (channel modificiation)
        self.isfinal = isfinal

    def forward(self, x):
        # print("input shape", x.shape)
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        
        # Skip ReLU for Final output
        if not self.isfinal:
            out = self.relu(out)

        return out
    ##### Modified ResNet for Feature Extraction
class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=64):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.bottleneckdim = in_channels
        self.conv1 = weight_norm(CausalConv2d(1, self.bottleneckdim, kernel_size=(7,7), stride=(2,1), bias=False))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2,1), padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, self.bottleneckdim, layers[0])
        self.layer2 = self._make_layer(block, self.bottleneckdim*2, layers[1], stride=(2,1))
        self.layer3 = self._make_layer(block, self.bottleneckdim*4, layers[2], stride=(2,1))
        self.layer4 = self._make_layer(block, self.bottleneckdim*8, layers[3], stride=(2,1), isfinal=True)

    def _make_layer(self, block, out_channels, blocks, stride=1, isfinal=False):
        downsample = None
        if stride != 1 : # Downsampling layer needs channel modification
            downsample = CausalConv2d(self.in_channels, out_channels,
                             kernel_size=1, stride=stride, bias=False)
                        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, isfinal=isfinal))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
def ResNet18(in_channels=64):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels)

## Conv-ReLU-Conv with Residual Connection
class ResBlock(nn.Module):
    def __init__(self, n_ch):
        super(ResBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=1, padding=1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = weight_norm(nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=1, padding=1))

    def forward(self, x, final=False):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        x += identity
        if final:
            out = x
        else:
            out = self.relu(x)
        return out
    
class Decoder_new(nn.Module):
    def __init__(self, bottleneck_shape=512):
        super(Decoder_new, self).__init__()

        self.bottleneck_shape = bottleneck_shape
        
        self.c1 = weight_norm(nn.ConvTranspose2d(self.bottleneck_shape, self.bottleneck_shape//2, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
        self.conv1 = ResBlock(self.bottleneck_shape//2)

        self.c2 = weight_norm(nn.ConvTranspose2d(self.bottleneck_shape//2, self.bottleneck_shape//4, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
        self.conv2 = ResBlock(self.bottleneck_shape//4)
        
        self.c3 = weight_norm(nn.ConvTranspose2d(self.bottleneck_shape//4, self.bottleneck_shape//8, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
        self.conv3 = ResBlock(self.bottleneck_shape//8)
        
        self.c4 = weight_norm(nn.ConvTranspose2d(self.bottleneck_shape//8, self.bottleneck_shape//16, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
        self.conv4 = ResBlock(self.bottleneck_shape//16)
        
        self.c5 = weight_norm(nn.ConvTranspose2d(self.bottleneck_shape//16, 1, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
        self.conv5 = ResBlock(1)

    def forward(self, x):
        x = self.c1(x)
        x = self.conv1(x)
        x = self.c2(x)
        x = self.conv2(x)
        x = self.c3(x)
        x = self.conv3(x)
        x = self.c4(x)
        x = self.conv4(x)
        x = self.c5(x)
        x = self.conv5(x, final=True)
        return x
    
    

