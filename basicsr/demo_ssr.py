# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import os
from glob import glob
# from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite, set_random_seed

import argparse
from basicsr.utils.options import dict2str, parse
from basicsr.utils.dist_util import get_dist_info, init_dist
import random
from tqdm import tqdm

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=False, default='options/test/NAFSSR/NAFSSR-T_x4.yml', help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--input_path', type=str, default='../datasets/Track1/Val/LR_x4', help='The path to the input images. For stereo image inference only.')
    parser.add_argument('--output_path', type=str, default='results/NAFSSR/Track1/', help='The path to the output images. For stereo image inference only.')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    opt['input_path'] = args.input_path
    opt['output_path'] = args.output_path

    return opt

def imread(img_path):
    file_client = FileClient('disk')
    img_bytes = file_client.get(img_path, None)
    try:
        img = imfrombytes(img_bytes, float32=True)
    except:
        raise Exception("path {} not working".format(img_path))

    img = img2tensor(img, bgr2rgb=True, float32=True)
    return img

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    ## 1. create model
    opt['dist'] = False
    model = create_model(opt)

    img_paths = glob(os.path.join(opt['input_path'], '*'))
    output_path = opt['output_path']
    os.makedirs(output_path, exist_ok=True)

    ## 2. read image
    for img_path in tqdm(img_paths):
        if '_R' in img_path:
            continue
        img_l = imread(img_path)
        img_r = imread(img_path.replace('_L', '_R'))

        img = torch.cat([img_l, img_r], dim=0)


        model.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if model.opt['val'].get('grids', False):
            model.grids()

        model.test()

        if model.opt['val'].get('grids', False):
            model.grids_inverse()

        visuals = model.get_current_visuals()
        sr_img_l = visuals['result'][:,:3]
        sr_img_r = visuals['result'][:,3:]
        sr_img_l, sr_img_r = tensor2img([sr_img_l, sr_img_r])
        imwrite(sr_img_l, os.path.join(output_path, img_path.split('/')[-1]))
        imwrite(sr_img_r, os.path.join(output_path, img_path.split('/')[-1].replace('_L', '_R')))

        # print(f'inference {img_l_path} .. finished. saved to {output_l_path}')
        # print(f'inference {img_r_path} .. finished. saved to {output_r_path}')

if __name__ == '__main__':
    main()

