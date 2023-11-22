import os
import __init_paths
import numpy as np
import math
import torch
import cv2

from torch.nn import functional as F
from tqdm import tqdm
from glob import glob

from face_model.face_gan import FaceGAN
from face_detect.retinaface_detection import RetinaFaceDetection
from align_faces import warp_and_crop_face, get_reference_facial_points
from argparse import ArgumentParser

parser=ArgumentParser()
parser.add_argument('--input_dir', type=str, dest='input_dir', default='./datasets/RealFaces200/LQ')
parser.add_argument('--output_dir', type=str, dest='output_dir', default='./datasets/RealFaces200')
args = parser.parse_args()

if os.path.exists(os.path.join(args.output_dir, 'Face_LQ')):
    pass
else:
    os.mkdir(os.path.join(args.output_dir, 'Face_LQ'))

if os.path.exists(os.path.join(args.output_dir, 'Face_HQ')):
    pass
else:
    os.mkdir(os.path.join(args.output_dir, 'Face_HQ'))  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    facedetector = RetinaFaceDetection(base_dir = './', device = device)
    facegan = FaceGAN(base_dir = './', 
                      in_size = 512,
                      out_size = None, 
                      model = 'GPEN-BFR-512', 
                      channel_multiplier = 2, 
                      narrow = 1, 
                      key = None, 
                      device = device)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    in_size = 512
    reference_5pts = get_reference_facial_points((in_size, in_size), inner_padding_factor, outer_padding, default_square)

indexes = []
for img_path in tqdm(sorted(glob(args.input_dir + '/' + '*'))):
    
    img_name = img_path.split('/')[-1][:-4]
    # print(img_name)
    input_b = cv2.imread(img_path, cv2.IMREAD_COLOR) # BGR

    with torch.no_grad():
        height, width = input_b.shape[:2]
        #print("image shape: ", input_b.shape)
        facebs, landms = facedetector.detect(input_b)
        if len(facebs) == 0:
            #raise ValueError("No face")
            print("No Face")
            indexes.append(img_name)
            continue

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
  
            if faceb[4] < 0.9: continue

            fh, fw = (faceb[3]-faceb[1]), (faceb[2]-faceb[0])
            x_min_land, y_min_land, x_max_land, y_max_land, _ = faceb
            x_min_land = int(x_min_land)
            x_max_land = int(x_max_land)
            y_min_land = int(y_min_land)
            y_max_land = int(y_max_land)

            if y_min_land < 0:
                y_min_land = 0
            if x_min_land < 0:
                x_min_land = 0

            facial5points = np.reshape(facial5points, (2, 5))

            of, tfm_inv = warp_and_crop_face(input_b, facial5points.astype(np.int32), reference_pts=reference_5pts, crop_size=(in_size, in_size)) #BGR
            ef = facegan.process(of, flip=True) #RGB

            img_hr = cv2.warpAffine(ef, tfm_inv*4, (width*4, height*4), flags=3)
            #print("image_hr shape: ", img_hr.shape)
            cv2.imwrite(os.path.join(args.output_dir, 'Face_LQ', img_name + f'_{i}' + '.png'), input_b[y_min_land:y_max_land+1, x_min_land:x_max_land+1,:])
            cv2.imwrite(os.path.join(args.output_dir, 'Face_HQ', img_name + f'_{i}' + '.png'), img_hr[y_min_land*4:y_max_land*4+4, x_min_land*4:x_max_land*4+4,:])


# print(indexes)