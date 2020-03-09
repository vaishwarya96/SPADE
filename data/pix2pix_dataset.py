"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
import cv2
from torch.autograd import Variable
import torch

class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, surface_paths, color_paths, input_paths, instance_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(surface_paths)
        util.natural_sort(color_paths)
        util.natural_sort(input_paths)

        if not opt.no_instance:
            util.natural_sort(instance_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        surface_paths = surface_paths[:opt.max_dataset_size]
        color_paths = color_paths[:opt.max_dataset_size]
        input_paths = input_paths[:opt.max_dataset_size]

        instance_paths = instance_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2, path3, path4 in zip(label_paths, surface_paths, color_paths, input_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)
                assert self.paths_match(path1, path3), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path3)

                assert self.paths_match(path1, path4), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path4)


        self.label_paths = label_paths
        self.surface_paths = surface_paths
        self.color_paths = color_paths
        self.input_paths = input_paths
        self.instance_paths = instance_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        surface_paths = []
        color_paths = []
        input_paths =[]
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, surface_paths, color_paths, input_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):

        self.opt.no_flip = True
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        
        # target image surface (real images)
        surface_path = self.surface_paths[index]
        assert self.paths_match(label_path, surface_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, surface_path)
        '''
        surface = Image.open(surface_path)
        surface = surface.convert('RGB')

        transform_surface = get_transform(self.opt, params)
        surface_tensor = transform_surface(surface)
        '''
        surface = cv2.imread(surface_path, -1)
        surface = surface[:,:,0]
        surface = 2 * (surface - surface.min())/(surface.max() - surface.min()) - 1
        surface_tensor = torch.from_numpy(surface).float().unsqueeze(0)

        # target image color (real images)
        color_path = self.color_paths[index]
        assert self.paths_match(label_path, color_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, color_path)
        '''
        color = Image.open(color_path)
        color = color.convert('RGB')

        transform_color = get_transform(self.opt, params)
        color_tensor = transform_color(color)
        '''

        color = cv2.imread(color_path)
        color = color / 255.0
        color = 2 * color - 1
        color_tensor = torch.from_numpy(color.transpose((2,0,1))).float()
        # input image (smoothened radius map)
        input_path = self.input_paths[index]
        assert self.paths_match(label_path, input_path), \
                "The label_path %s and input path %s don't match." %\
                (label_path, input_path)

        '''
        input_img = Image.open(input_path)

        transform_input = get_transform(self.opt, params)
        input_tensor = transform_input(input_img)
        '''
        input_img = cv2.imread(input_path, -1)
        input_img = input_img[:,:,0]
        min_value = input_img.min()
        max_value = input_img.max()
        #min_value = torch.from_numpy(input_img.min()).float()
        #max_value = torch.from_numpy(input_img.max()).float()

        input_img = 2 * (input_img - input_img.min())/(input_img.max() - input_img.min()) - 1
        input_tensor = torch.from_numpy(input_img).float().unsqueeze(0)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        #print(image_tensor - input_tensor)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'surface': surface_tensor,
                      'color' : color_tensor,
                      'path': input_path,
                      'input': input_tensor,
                      'input_max': str(max_value),
                      'input_min': str(min_value)
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
