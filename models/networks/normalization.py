"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm
from torch.autograd import Variable

# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        subnorm_type = 'none'
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared_2 = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=2, dilation = 2),
            nn.LeakyReLU(0.2)
        )
        self.mlp_gamma_2 = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=2, dilation = 2)
        self.mlp_beta_2 = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=2, dilation = 2)
        


        self.mlp_shared_4 = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=4, dilation = 4),
            nn.LeakyReLU(0.2)
        )
        self.mlp_gamma_4 = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=4, dilation = 4)
        self.mlp_beta_4 = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=4, dilation = 4)
        


        self.mlp_shared_8 = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=8, dilation = 8),
            nn.LeakyReLU(0.2)
        )
        self.mlp_gamma_8 = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=8, dilation = 8)
        self.mlp_beta_8 = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=8, dilation = 8)
        


        self.mlp_shared_16 = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=16, dilation = 16),
            nn.LeakyReLU(0.2)
        )
        self.mlp_gamma_16 = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=16, dilation = 16)
        self.mlp_beta_16 = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=16, dilation = 16)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw, dilation = 1),
            nn.LeakyReLU(0.2)
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw, dilation = 1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw, dilation = 1)




    def forward(self, x, segmap, other_channels=[]):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        segmap_shape = segmap.shape[2]
        #print(other_channels)
        
        if len(other_channels)==0:
            all_layers = segmap
        else:
            all_layers = torch.cat((other_channels, segmap), dim=1)

        gamma = 0
        beta = 0
        '''
        if segmap_shape == 16:
            actv = self.mlp_shared_2(all_layers)
            gamma = self.mlp_gamma_2(actv)
            beta = self.mlp_beta_2(actv)
        if segmap_shape == 32 or segmap_shape == 64:
            actv = self.mlp_shared_4(all_layers)
            gamma = self.mlp_gamma_4(actv)
            beta = self.mlp_beta_4(actv)

        if segmap_shape == 128 or segmap_shape == 256:
            actv = self.mlp_shared_8(all_layers)
            gamma = self.mlp_gamma_8(actv)
            beta = self.mlp_beta_8(actv)

        if segmap_shape == 512 or segmap_shape == 1024:
            actv = self.mlp_shared_16(all_layers)
            gamma = self.mlp_gamma_16(actv)
            beta = self.mlp_beta_16(actv)

        '''
        actv = self.mlp_shared(all_layers)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)


        #print(normalized.shape, gamma.shape, beta.shape)

        # apply scale and bias
        out = normalized * (1+ gamma) + beta

        return out
