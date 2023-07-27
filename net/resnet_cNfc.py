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
        out = self.batch_normalization_2(self.convolutional_2(out))
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
        num_outputs = (91, 91),):
        super(ResNet, self).__init__()

        self.convolutional_block_1 = SpectralNormalizedConvolutionalBlock(3, 16) # [16, 300, 300]
        self.convolutional_block_2 = SpectralNormalizedConvolutionalBlock(16, 32) # [32, 150, 150]
        self.convolutional_block_3 = SpectralNormalizedConvolutionalBlock(32, 64) # [64, 75, 75]
        self.convolutional_block_4 = SpectralNormalizedConvolutionalBlock(64, 128) # [128, 38, 38]
        
        self.fully_connected_block_1 = SpectralNormalizedFullyConnectedBlock(128*38*38 + 5, 512)
        self.fully_connected_block_2 = SpectralNormalizedFullyConnectedBlock(512, 256)
        self.fully_connected_block_3 = SpectralNormalizedFullyConnectedBlock(256, 128)
        self.fully_connected_block_4_1 = SpectralNormalizedFullyConnectedBlock(128, num_outputs[0]*num_outputs[1])
        self.fully_connected_block_4_2 = SpectralNormalizedFullyConnectedBlock(128, num_outputs[0]*num_outputs[1])
        self.fully_connected_block_4_3 = SpectralNormalizedFullyConnectedBlock(128, num_outputs[0]*num_outputs[1])
        
    # ============================================================ #
    
    def forward(self, map_input_tensor, record_input_tensor):
        out = self.convolutional_block_1(map_input_tensor)
        out = self.convolutional_block_2(out)
        out = self.convolutional_block_3(out)
        out = self.convolutional_block_4(out)
        
        out = out.view(out.size(0), -1) # Flattening
        """
        record_input_tensor = record_input_tensor.view(record_input_tensor.size(0), -1)
        """
        out = torch.cat((out, record_input_tensor), dim=1) # Concatenating
        
        out = self.fully_connected_block_1(out)
        out = self.fully_connected_block_2(out)
        out = self.fully_connected_block_3(out)
        out_1 = self.fully_connected_block_4_1(out)
        out_2 = self.fully_connected_block_4_2(out)
        out_3 = self.fully_connected_block_4_3(out)

        out_1 = out_1.view(out_1.size(0), 91, 91)
        out_2 = out_2.view(out_2.size(0), 91, 91)
        out_3 = out_3.view(out_3.size(0), 91, 91)
        return out_1, out_2, out_3

# ========================================================================================== #

def resnet18(**kwargs):
    # num_blocks = [2, 2, 2, 2] # Convolution
    model = ResNet(
        **kwargs)
    return model
