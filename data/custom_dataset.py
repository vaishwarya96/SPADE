"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class CustomDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--label_dir', type=str, required=True,
                            help='path to the directory that contains label images')
        parser.add_argument('--surface_dir', type=str, required=True,
                            help='path to the directory that contains surface images')
        parser.add_argument('--color_dir', type=str, required=True,
                            help='path to the directory that contains bark color images')
        parser.add_argument('--input_dir', type =str, required=True, 
                            help='path to the input image (smoothened surface) directory')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        surface_dir = opt.surface_dir
        surface_paths = make_dataset(surface_dir, recursive=False, read_cache=True)

        color_dir = opt.color_dir
        color_paths = make_dataset(color_dir, recursive=False, read_cache=True)


        input_dir = opt.input_dir
        input_paths = make_dataset(input_dir, recursive=False, read_cache=True)
        #print(label_paths[0:5],image_paths[0:5], input_paths[0:5])

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []
            
        print(len(label_paths), len(surface_paths), len(color_paths), len(input_paths), len(instance_paths))     
        assert len(label_paths) == len(surface_paths) == len(color_paths) == len(input_paths), "The #images in %s and %s do not match. Is there something wrong?"
        
        return label_paths, surface_paths, color_paths, input_paths, instance_paths
