"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.architecture import ModifiedSPADEResnetBlock as ModifiedSPADEResnetBlock
from models.networks.architecture import NormalResnetBlock as NormalResnetBlock


class DualGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers', choices=('normal', 'more', 'most'), default='normal', help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)
        '''
        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc_surface = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)                   
        else:
            #Otherwise, we make the network deterministic by starting with 
            #downsampled segmentation map instead of random z
            self.fc_surface = nn.Conv2d(3, 16 * nf, 3, padding=1)
        '''

        self.surface_down_0 = SPADEResnetBlock(1, 1 * nf, opt)
        self.surface_down_1 = SPADEResnetBlock(1 * nf, 2 * nf, opt)
        self.surface_down_2 = SPADEResnetBlock(2 * nf, 4 * nf, opt)
        self.surface_down_3 = SPADEResnetBlock(4 * nf, 8 * nf, opt)
        self.surface_down_4 = SPADEResnetBlock(8 * nf, 16 * nf, opt)


        self.head_0_surface = SPADEResnetBlock(16 * nf, 16 * nf, opt)                          

        self.G_middle_0_surface = SPADEResnetBlock(16 * nf, 16 * nf, opt)                       
        self.G_middle_1_surface = SPADEResnetBlock(16 * nf, 16 * nf, opt)                  
        
        
        #Surface geneator layers
        self.surface_up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.surface_up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.surface_up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.surface_up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.surface_up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.surface_conv_img = nn.Conv2d(final_nc, 1, 3, padding=1)



        #Color geneator layers
        extra_channels = True
        self.head_0_color = ModifiedSPADEResnetBlock(16 * nf, 16 * nf, opt, extra_channels)                          

        self.G_middle_0_color = ModifiedSPADEResnetBlock(16 * nf, 16 * nf, opt, extra_channels)                       
        self.G_middle_1_color = ModifiedSPADEResnetBlock(16 * nf, 16 * nf, opt, extra_channels)              



        self.color_up_0 = ModifiedSPADEResnetBlock(16 * nf, 8 * nf, opt, extra_channels)
        self.color_up_1 = ModifiedSPADEResnetBlock(8 * nf, 4 * nf, opt, extra_channels)
        self.color_up_2 = ModifiedSPADEResnetBlock(4 * nf, 2 * nf, opt, extra_channels)
        self.color_up_3 = ModifiedSPADEResnetBlock(2 * nf, 1 * nf, opt, extra_channels)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.color_up_4 = ModifiedSPADEResnetBlock(1 * nf, nf // 2, opt, extra_channels)
            final_nc = nf // 2

        self.color_conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.b1x1_conv = nn.Conv2d(8 * nf, 16, 3, padding=1)
        self.b1x2_conv = nn.Conv2d(4 * nf, 16, 3, padding=1)
        self.b1x3_conv = nn.Conv2d(2 * nf, 16, 3, padding=1)
        self.b1x4_conv = nn.Conv2d(1 * nf, 16, 3, padding=1)
        self.b1x5_conv = nn.Conv2d(nf//2, 16, 3, padding=1)


        self.up = nn.Upsample(scale_factor=2)
        self.down = nn.Upsample(scale_factor = 0.5)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, seg, input_image, z=None):
        '''
        layers = []
       
        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x1 = self.fc_surface(z)
        #z = F.interpolate(input_image, size = (self.sh, self.sw))
        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(input_image, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0_surface(x)

        x = self.up(x)
        x = self.G_middle_0_surface(x)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1_surface(x)                   

        x = self.up(x)                                   #Common layers upto this
        '''
        #Branch1 (surface generator)

        x = self.surface_down_0(input_image, seg)
        x = self.down(x)

        x = self.surface_down_1(x,seg)
        x = self.down(x)

        x = self.surface_down_2(x,seg)
        x = self.down(x)

        x = self.surface_down_3(x,seg)
        x = self.down(x)

        x = self.surface_down_4(x,seg)
        x = self.down(x)

        x = self.head_0_surface(x,seg)
        x = self.up(x)
        x = self.G_middle_0_surface(x,seg)

        if self.opt.num_upsampling_layers == 'more' or self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1_surface(x,seg)
        x = self.up(x)

        b1x1 = self.surface_up_0(x,seg)
        b1x1_conv = self.b1x1_conv(b1x1)

        b1x2 = self.up(b1x1)
        b1x2 = self.surface_up_1(b1x2,seg)
        b1x2_conv = self.b1x2_conv(b1x2)

        b1x3 = self.up(b1x2)
        b1x3 = self.surface_up_2(b1x3,seg)
        b1x3_conv = self.b1x3_conv(b1x3)

        b1x4 = self.up(b1x3)
        b1x4 = self.surface_up_3(b1x4,seg)
        b1x4_conv = self.b1x4_conv(b1x4)
        
        surface = b1x4
        if self.opt.num_upsampling_layers == 'most':
            b1x5 = self.up(b1x4)
            b1x5 = self.surface_up_4(b1x5,seg)
            b1x5_conv = self.b1x5_conv(b1x5)

            surface = self.surface_conv_img(F.leaky_relu(b1x5, 2e-1))
        else:
            surface = self.surface_conv_img(F.leaky_relu(b1x4, 2e-1))

        surface = F.tanh(surface)

        #Branch2 (color generator)

        b2x1 = self.color_up_0(x, seg, b1x1_conv)
        b2x2 = self.up(b2x1)
        b2x2 = self.color_up_1(b2x2, seg, b1x2_conv)
        b2x3 = self.up(b2x2)
        b2x3 = self.color_up_2(b2x3, seg, b1x3_conv)
        b2x4 = self.up(b2x3)
        b2x4 = self.color_up_3(b2x4, seg, b1x4_conv)
        
        color = b2x4
        if self.opt.num_upsampling_layers == 'most':
            b2x5 = self.up(b2x4)
            b2x5 = self.color_up_4(b2x5, seg, b1x5_conv)
            color = self.color_conv_img(F.leaky_relu(b2x5, 2e-1))
        else: 
            color = self.color_conv_img(F.leaky_relu(b2x4, 2e-1))
        color = F.tanh(color)



        return surface, color





class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)
                
