from PIL import Image
import cv2
import random
import numpy as np
import math
import os
from glob import glob
from tqdm import tqdm
import imageio.v2 as imageio

import pyiqa
import lpips as lpips_test
from pytorch_fid.fid_score import *
from argparse import ArgumentParser

def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

parser=ArgumentParser()
parser.add_argument('--result_dir', type=str, dest='result_dir', default='./datasets/FFHQ_Multi_iid/Res')
parser.add_argument('--gt_dir', type=str, dest='gt_dir', default='./datasets/FFHQ_Multi_iid/GT')
parser.add_argument('--fid_ref_dir', type=str, dest='fid_ref_dir', default='./datasets/FFHQ_Multi_iid/GT')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():

    iqa_nqie = pyiqa.create_metric('niqe', device=torch.device('cuda'))
    loss_fn_vgg = lpips_test.LPIPS(net='alex')
    loss_fn_vgg.to(device)

    total_psnr = 0
    total_lpips = 0
    total_niqe = 0

    nums = len(glob(args.result_dir+ '/' + '*'))
    for img_path in tqdm(sorted(glob(args.result_dir+ '/' + '*'))):
        
        img_name = img_path.split('/')[-1][:-4]

        img0p = imageio.imread(img_path) 
        img1p = imageio.imread(os.path.join(args.gt_dir, img_name + '.png'))

        img0l = lpips_test.im2tensor(lpips_test.load_image(img_path)) # RGB image from [-1,1]
        img1l = lpips_test.im2tensor(lpips_test.load_image(os.path.join(args.gt_dir, img_name + '.png')))

        img0l = img0l.to(device)
        img1l = img1l.to(device)

        #dist_psnr = psnr(img0p, img1p)
        dist_psnr = psnr(bgr2ycbcr(img0p, only_y=True), bgr2ycbcr(img1p, only_y=True))
        total_psnr = total_psnr + dist_psnr

        dist_lpips = loss_fn_vgg.forward(img0l, img1l)
        total_lpips = total_lpips + dist_lpips.data[0][0][0][0].item()

        dist_niqe = iqa_nqie(img_path)
        total_niqe = total_niqe + dist_niqe

    psnr_final = total_psnr / nums
    lpips_final = total_lpips / nums
    niqe_final = total_niqe / nums

    fid_final = calculate_fid_given_paths([args.result_dir, args.fid_ref_dir],
                                    1,
                                    device,
                                    2048,
                                    8)

    
    print(psnr_final)
    print(lpips_final)
    print(fid_final)
    print(niqe_final)

