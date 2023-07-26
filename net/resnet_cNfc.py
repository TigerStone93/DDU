"""
Pytorch implementation of ResNet models.
Reference:
[1] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR, 2016.

ResNet
https://aistudy9314.tistory.com/58
https://pseudo-lab.github.io/pytorch-guide/docs/ch03-1.html

Spectral Normalization
https://pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html
https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/spectral_norm.py
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils

# from net.spectral_normalization.spectral_norm_conv_inplace import spectral_norm_conv # coeff
# from net.spectral_normalization.spectral_norm_fc import spectral_norm_fc # coeff

# ========================================================================================== #

class SpectralNormalizedConvolutionalBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(SpectralNormalizedConvolutionalBlock, self).__init__()
        self.convolutional_1 = utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        self.batch_normalization_1 = nn.BatchNorm2d(out_channels)
        self.convolutional_2 = utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        self.batch_normalization_2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                                          nn.BatchNorm2d(out_channels))

        self.leaky_relu = F.leaky_relu

    # ============================================================ #
    
    def forward(self, x):
        out = self.leaky_relu(self.batch_normalization_1(self.convolutional_1(x))) # relu
        out = self.batch_normalization_2(self.convolutional_2(out)
        out += self.shortcut(x)
        out = self.leaky_relu(out) # relu
        return out
        
# ========================================================================================== #

class SpectralNormalizedFullyConnectedBlock(nn.Module):
    expansion = 1
    def __init__(self, in_features, out_features):
        super(SpectralNormalizedFullyConnectedBlock, self).__init__()
        self.fully_connected_1 = utils.spectral_norm(nn.Linear(in_features, out_features))
        self.fully_connected_2 = utils.spectral_norm(nn.Linear(out_features, out_features))
        
        self.shortcut = utils.spectral_norm(nn.Linear(in_features, out_features)) # skip_connection
        
        self.leaky_relu = F.leaky_relu
        
    # ============================================================ #
    
    def forward(self, x):
        out = self.leaky_relu(self.fully_connected_1(x)) # relu
        out = self.fully_connected_2(out)

        residual = x
        if x.shape != out.shape:
            residual = self.shortcut(x)
        out += residual
        
        out = self.leaky_relu(out) # relu
        return out

# ========================================================================================== #

class ResNet(nn.Module):
    def __init__(
        self,
        num_outputs = 10,):
        super(ResNet, self).__init__()

        self.convolutional_block_1 = SpectralNormalizedConvolutionalBlock(3, 16) # [16, 300, 300]
        self.convolutional_block_2 = SpectralNormalizedConvolutionalBlock(16, 32) # [32, 150, 150]
        self.convolutional_block_3 = SpectralNormalizedConvolutionalBlock(32, 64) # [64, 75, 75]
        self.convolutional_block_4 = SpectralNormalizedConvolutionalBlock(64, 128) # [128, 38, 38]
        
        self.fully_connected_block_1 = SpectralNormalizedFullyConnectedBlock(128*38*38 + 5, 512)
        self.fully_connected_block_2 = SpectralNormalizedFullyConnectedBlock(512, 256)
        self.fully_connected_block_3 = SpectralNormalizedFullyConnectedBlock(256, 128)
        self.fully_connected_block_4 = SpectralNormalizedFullyConnectedBlock(128, num_outputs)
        
    # ============================================================ #
    
    def forward(self, map_input_tensor, record_input_tensor):
        out = self.convolutional_block_1(map_input_tensor)
        """
        out = self.convolutional_block_1(map_input_tensor.unsqueeze(0)) # Adding batch dimension ex) (100, 100) → (1, 100, 100)
        """
        out = self.convolutional_block_2(out)
        out = self.convolutional_block_3(out)
        out = self.convolutional_block_4(out)
        
        out = out.view(out.size(0), -1) # Flattening
        # record_input_tensor = record_input_tensor.view(record_input_tensor.size(0), -1)        
        out = torch.cat((out, record_input_tensor), dim=1) # Concatenating

        """
        out = out.view(-1) # Flattening
        # record_input_tensor = record_input_tensor.view(-1)
        out = torch.cat((out, record_input_tensor), dim=0) # Concatenating
        """
        
        out = self.fully_connected_block_1(out)
        """
        out = self.fully_connected_block_1(out.unsqueeze(0)) # Adding batch dimension ex) (100, 100) → (1, 100, 100)
        """
        out = self.fully_connected_block_2(out)
        out = self.fully_connected_block_3(out)
        out = self.fully_connected_block_4(out)

        out = out.view(out.size(0), 91, 91)
        """
        out = out.view(91, 91)
        """
        return out

# ========================================================================================== #

def resnet18(**kwargs):
    # num_blocks = [2, 2, 2, 2] # Convolution
    model = ResNet(
        **kwargs)
    return model

"""
def resnet50(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs)
    return model
"""
