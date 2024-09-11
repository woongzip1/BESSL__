import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torchinfo import summary
from FeatureExtractor.model_encoder import ResNet18

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

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.c1 = weight_norm(nn.ConvTranspose2d(512, 256, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
        self.conv1 = ResBlock(256)

        self.c2 = weight_norm(nn.ConvTranspose2d(256, 128, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
        self.conv2 = ResBlock(128)
        
        self.c3 = weight_norm(nn.ConvTranspose2d(128, 64, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
        self.conv3 = ResBlock(64)
        
        self.c4 = weight_norm(nn.ConvTranspose2d(64, 32, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
        self.conv4 = ResBlock(32)
        
        self.c5 = weight_norm(nn.ConvTranspose2d(32, 1, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
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

if __name__ == "__main__":
# Example input tensor with shape B x 512 x 10 x 80
    encoder = ResNet18()
    input = torch.rand(8,1,32*10, 40)
    out = encoder(input)

    model = Decoder()
    print(summary(model, input_data=out))
