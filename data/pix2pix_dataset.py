"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
import cv2
import torch
from torch.autograd import Variable

class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, input_paths, instance_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        util.natural_sort(input_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        input_paths = input_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2, path3 in zip(label_paths, image_paths, input_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)
                assert self.paths_match(path1, path3), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path3)


        self.label_paths = label_paths
        self.image_paths = image_paths
        self.input_paths = input_paths
        self.instance_paths = instance_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        input_paths =[]
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, input_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image

        self.opt.no_flip = True
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        '''
        mean = 0.0; stddev = 0.001;
        noise = Variable(label_tensor.data.new(label_tensor.size()).normal_(mean, stddev))
        label_tensor = label_tensor + noise
        '''
        # target image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        '''
        image = Image.open(image_path)
        image = image.convert('RGB')
         
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)
        '''
        image = cv2.imread(image_path, -1)
        image = image[:,:,0]
        #image = image/65535.0
        #image = 2 * image - 1
        image = 2 * (image - image.min())/(image.max() - image.min()) - 1
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)

        # input image 
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
        #input_img = input_img/65535.0
        #input_img = 2 * input_img - 1
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
                      'image': image_tensor,
                      'path': image_path,
                      'input': input_tensor
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
