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
    def __init__(self, in_channels, out_channels):
        super(SpectralNormalizedConvolutionalBlock, self).__init__()
        self.convolutional_1 = utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.batch_normalization_1 = nn.BatchNorm2d(out_channels)
        self.convolutional_2 = utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.batch_normalization_2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)),
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

        self.convolutional_block_1 = SpectralNormalizedConvolutionalBlock(16, 16)
        self.convolutional_block_2 = SpectralNormalizedConvolutionalBlock(16, 32)
        self.convolutional_block_3 = SpectralNormalizedConvolutionalBlock(32, 64)
        self.convolutional_block_4 = SpectralNormalizedConvolutionalBlock(64, 128)
        
        self.fully_connected_block_1 = SpectralNormalizedFullyConnectedBlock(784, 784)
        self.fully_connected_block_2 = SpectralNormalizedFullyConnectedBlock(392, 392)
        self.fully_connected_block_3 = SpectralNormalizedFullyConnectedBlock(196, 196)
        self.fully_connected_block_4 = SpectralNormalizedFullyConnectedBlock(98, 98)
        self.output_layer = nn.Linear(98, num_outputs)
        
        self.leaky_relu = F.leaky_relu
        
    # ============================================================ #
    
    def forward(self, x):
        out = self.convolutional_block_1(x)
        out = self.convolutional_block_2(out)
        out = self.convolutional_block_3(out)
        out = self.convolutional_block_4(out)
        out = out.view(out.size(0), -1) # Flattening
        
        out = self.leaky_relu(self.fully_connected_block_1(out))
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
