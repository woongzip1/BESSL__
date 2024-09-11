import torch
import torch.nn as nn
import torch.nn.functional as F
import math

""" Causal Convolutions """

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + 1 - self.stride[0]
 
    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)

class CausalConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Calculate padding for temporal dimension (T)
        self.temporal_padding = self.dilation[1] * (self.kernel_size[1] - 1) - (self.stride[1] - 1)
        
        # Calculate total padding for frequency dimension (F)
        total_f_padding = self.dilation[0] * (self.kernel_size[0] - 1) - (self.stride[0] - 1)
        
        # Split total padding into top and bottom (asymmetrical padding if needed)
        self.frequency_padding_top = math.ceil(total_f_padding / 2)
        self.frequency_padding_bottom = math.floor(total_f_padding / 2)
        
    def forward(self, x):
        # Apply padding: F (top and bottom), T (only to the left)
        # print(f"Temporal Padding (T): {self.temporal_padding}")
        # print(f"Frequency Padding (F): top={self.frequency_padding_top}, bottom={self.frequency_padding_bottom}")
        x = F.pad(x, [self.temporal_padding, 0, self.frequency_padding_top, self.frequency_padding_bottom])
        return self._conv_forward(x, self.weight, self.bias)

if __name__ == "__main__":
    in_channels = 3  # Number of input channels
    out_channels = 64  # Number of output channels
    kernel_size = (7, 7)  # Kernel size for (F, T) dimensions
    stride = (2,1)  # Stride for convolution (F, T)
    dilation = (1,1)  # Dilation for convolution (F, T)

    causal_conv2d = CausalConv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)

    # Example input tensor with shape: B x C x F x T
    input_tensor = torch.randn(8, in_channels, 320, 100) 
    output = causal_conv2d(input_tensor)
    
    # Output shape
    print("Output shape:", output.shape)
