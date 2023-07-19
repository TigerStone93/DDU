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
from torch.nn.utils import spectral_norm

# from net.spectral_normalization.spectral_norm_conv_inplace import spectral_norm_conv # coeff
# from net.spectral_normalization.spectral_norm_fc import spectral_norm_fc # coeff

# ========================================================================================== #

class SpectralNormalizedFullyConnected(nn.Module):
    def __init__(self, in_features, out_features):
        super(SpectralNormalizedFullyConnected, self).__init__()
        self.fully_connected = spectral_norm(nn.Linear(in_features, out_features))

    def forward(self. x):
        return self.fully_connected(x)

# ========================================================================================== #

# Basic Residual Block
# for ResNet18
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_features, out_features):
        super(BasicBlock, self).__init__()
        self.fully_connected_1 = SpectralNormalizedFullyConnected(in_features, out_features)
        self.fully_connected_2 = SpectralNormalizedFullyConnected(out_features, out_features)
        self.downsampled = SpectralNormalizedFullyConnected(in_features, out_features) # Skip Connection
        self.leaky_relu = F.leaky_relu # Activation

    # ============================================================ #
    
    def forward(self, x):
        out = self.leaky_relu(self.fully_connected_1(x))
        out = self.fully_connected_2(out)
        
        residual = x
        if residual.shape != out.shape:
            residual = self.downsampled(residual)
        out += residual
        
        out = self.leaky_relu(out)
        return out

# ========================================================================================== #

"""
# Bottleneck Block
# for ResNet50 ~ ResNet152
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, input_size, wrapped_conv, in_planes, planes, stride=1, mod=True):
        super(Bottleneck, self).__init__()
        self.conv1 = wrapped_conv(input_size, in_planes, planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = wrapped_conv(input_size, planes, planes, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = wrapped_conv(math.ceil(input_size / stride), planes, self.expansion * planes, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.mod = mod
        self.activation = F.leaky_relu if self.mod else F.relu

        self.shortcut = nn.Sequential() # Skip Connection
        if stride != 1 or in_planes != self.expansion * planes:
            if mod:
                self.shortcut = nn.Sequential(AvgPoolShortCut(stride, self.expansion * planes, in_planes))
            else:
                self.shortcut = nn.Sequential(wrapped_conv(input_size, in_planes, self.expansion * planes, kernel_size=1, stride=stride,), nn.BatchNorm2d(self.expansion * planes),)

    # ============================================================ #
    
    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out
"""

# ========================================================================================== #

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_outputs = 10,):
        super(ResNet, self).__init__()

        self.layer_1 = self._make_layer(block, 784, 784)
        self.layer_2 = self._make_layer(block, 392, 392)
        self.layer_3 = self._make_layer(block, 196, 196)
        self.layer_4 = self._make_layer(block, 98, 98)
        self.fully_connected_layer = nn.Linear(98 * block.expansion, num_outputs)

    # ============================================================ #

    # Fully Connected Layer
    def _make_layer(self, block, in_features, out_features):
        layers = []
        layers.append(block(in_features, out_features))
        layers.append(block(out_features, out_features))
        return nn.Sequential(*layers)
        
    # ============================================================ #
    
    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.fully_connected_layer(out)
        return out

# ========================================================================================== #

def resnet18(**kwargs):
    # num_blocks = [2, 2, 2, 2] # Convolution
    model = ResNet(
        BasicBlock,
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

def resnet101(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs)
    return model

def resnet110(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 26, 3],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs)
    return model

def resnet152(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 8, 36, 3],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs)
    return model
"""
