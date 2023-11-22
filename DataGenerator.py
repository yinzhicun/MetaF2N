from config import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

import random
import math
import cv2 as cv
import numpy as np
from PIL import Image
from scipy.io import loadmat

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import torch
from torch.nn import functional as F
from utils.diffjpeg import DiffJPEG
from face_model.face_gan import FaceGAN
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))

class DataGenerator(object):
    def __init__(self, output_shape, meta_batch_size, task_batch_size, tfrecord_path0, tfrecord_path1):
        self.buffer_size = 1000 # tf.data.TFRecordDataset buffer size

        self.TASK_BATCH_SIZE = task_batch_size
        self.HEIGHT, self.WIDTH, self.CHANNEL, self.HEIGHT1, self.WIDTH1 = output_shape

        self.back_size = 400
        self.face_size = 256
        self.patch_size = self.HEIGHT
        self.patch_size1 = self.HEIGHT1

        self.META_BATCH_SIZE = meta_batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            self.facegan = FaceGAN(base_dir = './', 
                            in_size = 256,
                            out_size = None, 
                            model = 'GPEN-BFR-256', 
                            channel_multiplier = 1, 
                            narrow = 0.5, 
                            key = None, 
                            device = self.device)

        self.tfrecord_path0 = tfrecord_path0
        self.tfrecord_path1 = tfrecord_path1
        self.label_train0, self.label_train1 = self.load_tfrecord()

        self.jpeger = DiffJPEG(differentiable=False).to(self.device) 
        self.usm_sharpener = USMSharp().to(self.device) 

        # Degradation settings which are same as Real-ESRGAN
        # settings for the first degradation
        self.blur_kernel_size = 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma = [0.2, 3]
        self.betag_range = [0.5, 4]  # betag used in generalized Gaussian blur kernels
        self.betap_range = [1, 2]  # betap used in plateau blur kernels
        self.sinc_prob = 0.1  # the probability for sinc filters

        self.resize_prob = [0.2, 0.7, 0.1]  # up, down, keep
        self.resize_range = [0.15, 1.5]
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.gray_noise_prob = 0.4
        self.jpeg_range = [30, 95]

        # settings for the second degradation
        self.blur_kernel_size2 = 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        self.sinc_prob2 = 0.1
        
        self.second_blur_prob = 0.8
        self.resize_prob2 = [0.3, 0.4, 0.3]  # up, down, keep
        self.resize_range2 = [0.3, 1.2]
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 25]
        self.poisson_scale_range2 = [0.05, 2.5]
        self.gray_noise_prob2 = 0.4
        self.jpeg_range2 = [30, 95]
        
        # a final sinc filter
        self.final_sinc_prob = 0.8

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
        

    def generate_data(self, sess):
        label_train_0=sess.run(self.label_train0)
        label_train_1=sess.run(self.label_train1)

        input_a = []
        label_a = []
        label_a_gt = []
        input_b = []
        label_b = []
        label_b_nousm = []
        for t in range(self.META_BATCH_SIZE):

            inputa_task = []
            labela_task = []
            labela_gt = []
            inputb_task = []
            labelb_task = []
            labelb_task_nousm = []

            #Blur
            randb2 = np.random.uniform()
            kernel_size = random.choice(self.kernel_range)

            if np.random.uniform() < self.sinc_prob:
                # this sinc filter setting is for kernels ranging from [7, 21]
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)

            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

            # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.sinc_prob2:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel2 = random_mixed_kernels(
                    self.kernel_list2,
                    self.kernel_prob2,
                    kernel_size,
                    self.blur_sigma2,
                    self.blur_sigma2, [-math.pi, math.pi],
                    self.betag_range2,
                    self.betap_range2,
                    noise_range=None)

            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

            # ------------------------------------- the final sinc kernel ------------------------------------- #
            if np.random.uniform() < self.final_sinc_prob:
                kernel_size = random.choice(self.kernel_range)
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                sinc_kernel = torch.FloatTensor(sinc_kernel)
            else:
                sinc_kernel = self.pulse_tensor
            
            kernel = torch.FloatTensor(kernel).to(self.device)
            kernel2 = torch.FloatTensor(kernel2).to(self.device)

            sinc_kernel = sinc_kernel.to(self.device)

            #Downsample
            updown_type1 = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
            if updown_type1 == 'up':
                scale1 = np.random.uniform(1, self.resize_range[1])
            elif updown_type1 == 'down':
                scale1 = np.random.uniform(self.resize_range[0], 1)
            else:
                scale1 = 1

            mode1 = random.choice(['area', 'bilinear', 'bicubic'])

            updown_type2 = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
            if updown_type2 == 'up':
                scale2 = np.random.uniform(1, self.resize_range2[1])
            elif updown_type2 == 'down':
                scale2 = np.random.uniform(self.resize_range2[0], 1)
            else:
                scale2 = 1

            mode2 = random.choice(['area', 'bilinear', 'bicubic'])

            #Noise
            randn1 = np.random.uniform()
            randn2 = np.random.uniform()

            #JEPG
            randj2 = np.random.uniform()
            mode3 = random.choice(['area', 'bilinear', 'bicubic'])
            jpeg_p = torch.zeros(1).uniform_(*self.jpeg_range).to(self.device)
            jpeg_p2 = torch.zeros(1).uniform_(*self.jpeg_range2).to(self.device)


            for idx in range(self.TASK_BATCH_SIZE):

                img1_ = label_train_0[t*self.TASK_BATCH_SIZE + idx]
                img2_ = label_train_1[t*self.TASK_BATCH_SIZE + idx]
                img_face_gt = img1_
                #img_back = img2_ / 255.
                
                img1_gt = torch.from_numpy(img1_.transpose(2, 0, 1) / 255.).unsqueeze(0).type(torch.FloatTensor).to(self.device)
                img2_gt = torch.from_numpy(img2_.transpose(2, 0, 1) / 255.).unsqueeze(0).type(torch.FloatTensor).to(self.device)
                
         
                img1 = img1_gt
                img2 = self.usm_sharpener(img2_gt)

                # ----------------------- The first degradation process ----------------------- #
                # blur
                out1 = filter2D(img1, kernel)
                out2 = filter2D(img2, kernel)
                
                # random resize
                out1 = F.interpolate(out1, scale_factor=scale1, mode=mode1)
                out2 = F.interpolate(out2, scale_factor=scale1, mode=mode1)
                
                # add noise
                gray_noise_prob = self.gray_noise_prob
                if randn1 < self.gaussian_noise_prob:
                    out1 = random_add_gaussian_noise_pt(
                        out1, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
                    out2 = random_add_gaussian_noise_pt(
                        out2, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob) 
                else:
                    out1 = random_add_poisson_noise_pt(
                        out1,
                        scale_range=self.poisson_scale_range,
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                    out2 = random_add_poisson_noise_pt(
                        out2,
                        scale_range=self.poisson_scale_range,
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)

                # JPEG compression
               
                out1 = torch.clamp(out1, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
                out1 = self.jpeger(out1, quality=jpeg_p)

                out2 = torch.clamp(out2, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifactsW
                out2 = self.jpeger(out2, quality=jpeg_p)


                # ----------------------- The second degradation process ----------------------- #
                # blur
                if randb2 < self.second_blur_prob:
                    out1 = filter2D(out1, kernel2)
                    out2 = filter2D(out2, kernel2)

                # random resize
                out1 = F.interpolate(
                    out1, size=(int(self.face_size/4 * scale2), int(self.face_size/4 * scale2)), mode=mode2)

                out2 = F.interpolate(
                    out2, size=(int(self.back_size/4 * scale2), int(self.back_size/4 * scale2)), mode=mode2)

                # add noise
                gray_noise_prob2 = self.gray_noise_prob2
                if randn2 < self.gaussian_noise_prob2:
                   
                    out1 = random_add_gaussian_noise_pt(
                        out1, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob2)
                    out2 = random_add_gaussian_noise_pt(
                        out2, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob2) 
                else:
                    
                    out1 = random_add_poisson_noise_pt(
                        out1,
                        scale_range=self.poisson_scale_range2,
                        gray_prob=gray_noise_prob2,
                        clip=True,
                        rounds=False)
                    out2 = random_add_poisson_noise_pt(
                        out2,
                        scale_range=self.poisson_scale_range2,
                        gray_prob=gray_noise_prob2,
                        clip=True,
                        rounds=False)
    

                # JPEG compression + the final sinc filter
              
                if randj2 < 0.5:
                    # resize back + the final sinc filter
                    out1 = F.interpolate(out1, size=(self.face_size // 4, self.face_size // 4), mode=mode3)
                    out1 = filter2D(out1, sinc_kernel)
                    out2 = F.interpolate(out2, size=(self.back_size // 4, self.back_size // 4), mode=mode3)
                    out2 = filter2D(out2, sinc_kernel)
                    # JPEG compression
                    out1 = torch.clamp(out1, 0, 1)
                    out1 = self.jpeger(out1, quality=jpeg_p2)
                    out2 = torch.clamp(out2, 0, 1)
                    out2 = self.jpeger(out2, quality=jpeg_p2)
                else:
                    # JPEG compression
                    out1 = torch.clamp(out1, 0, 1)
                    out1 = self.jpeger(out1, quality=jpeg_p2)
                    out2 = torch.clamp(out2, 0, 1)
                    out2 = self.jpeger(out2, quality=jpeg_p2)
                    # resize back + the final sinc filter
                    out1 = F.interpolate(out1, size=(self.face_size // 4, self.face_size // 4), mode=mode3)
                    out1 = filter2D(out1, sinc_kernel)
                    out2 = F.interpolate(out2, size=(self.back_size // 4, self.back_size // 4), mode=mode3)
                    out2 = filter2D(out2, sinc_kernel)
                
                img1_lr = torch.clamp((out1 * 255.0).round(), 0, 255)
                img1_hr_lr = F.interpolate(img1_lr, size=(256, 256), mode='bicubic')
                img1_hr_lr = img1_hr_lr.squeeze(0).permute(1,2,0).detach().cpu().numpy()
                img1_lr = img1_lr.squeeze(0).permute(1,2,0).detach().cpu().numpy()
                
                img2_lr = torch.clamp((out2 * 255.0).round(), 0, 255)
                img2_lr = img2_lr.squeeze(0).permute(1,2,0).detach().cpu().numpy()

                img1 = img1.squeeze(0).permute(1,2,0).detach().cpu().numpy()
                img2 = img2.squeeze(0).permute(1,2,0).detach().cpu().numpy()
                img2_nousm = img2_gt.squeeze(0).permute(1,2,0).detach().cpu().numpy()
                
                with torch.no_grad():
                    img1_hr = self.facegan.process(img1_hr_lr)
                
                count = 0
                while (1):
                    face_crop_h = random.randrange(0, 64 - self.patch_size//4)
                    face_crop_w = random.randrange(0, 64 - self.patch_size//4)
                    face_patch_gt = img_face_gt[face_crop_h*4:face_crop_h*4 + self.patch_size, face_crop_w*4:face_crop_w*4 + self.patch_size,:]
                    face_patch_hr = img1_hr[face_crop_h*4:face_crop_h*4 + self.patch_size, face_crop_w*4:face_crop_w*4 + self.patch_size,:]
                    face_patch_lr = img1_lr[face_crop_h:face_crop_h + self.patch_size//4, face_crop_w:face_crop_w + self.patch_size//4,:]
                    
                    count = count + 1
                    if np.log(gradients(face_patch_hr.astype(np.float64)/255.)+1e-10) >= -6.0:
                        break
                    
                    if count > 5:
                        count = 0
                        break
                
                back_crop_h = random.randrange(0, 100 - self.patch_size1//4)
                back_crop_w = random.randrange(0, 100 - self.patch_size1//4)
                back_patch_gt = img2[back_crop_h*4:back_crop_h*4 + self.patch_size1, back_crop_w*4:back_crop_w*4 + self.patch_size1,:]
                back_patch_lr = img2_lr[back_crop_h:back_crop_h + self.patch_size1//4, back_crop_w:back_crop_w + self.patch_size1//4,:]
                back_patch_gt_nousm = img2_nousm[back_crop_h*4:back_crop_h*4 + self.patch_size1, back_crop_w*4:back_crop_w*4 + self.patch_size1,:]
  
                hflip = random.random() < 0.5
                if hflip:
                    inputa_task.append(cv.flip(face_patch_lr/255., 1))
                    labela_task.append(cv.flip(face_patch_hr/255., 1))
                    labela_gt.append(cv.flip(face_patch_gt/255., 1))
                    inputb_task.append(cv.flip(back_patch_lr/255., 1))
                    labelb_task.append(cv.flip(back_patch_gt, 1))
                    labelb_task_nousm.append(cv.flip(back_patch_gt_nousm, 1))
                else:
                    inputa_task.append(face_patch_lr/255.)
                    labela_task.append(face_patch_hr/255.)
                    labela_gt.append(face_patch_gt/255.)
                    inputb_task.append(back_patch_lr/255.)
                    labelb_task.append(back_patch_gt)
                    labelb_task_nousm.append(back_patch_gt_nousm)

            input_a.append(np.asarray(inputa_task))
            input_b.append(np.asarray(inputb_task))
            label_a.append(np.asarray(labela_task))
            label_a_gt.append(np.asarray(labela_gt))
            label_b.append(np.asarray(labelb_task))
            label_b_nousm.append(np.asarray(labelb_task_nousm))
       
        input_a = np.asarray(input_a)
        input_b = np.asarray(input_b)
        label_a = np.asarray(label_a)
        label_a_gt = np.asarray(label_a_gt)
        label_b = np.asarray(label_b)
        label_b_nousm = np.asarray(label_b_nousm)

        inputa=input_a
        labela=label_a
        inputb=input_b
        labelb=label_b
        labelbnousm = label_b_nousm
        labelagt = label_a_gt

        return inputa, labela, inputb, labelb, labelagt, labelbnousm

    '''Load TFRECORD'''
    def _parse_function1(self, example_proto):
        keys_to_features = {'label': tf.compat.v1.FixedLenFeature([], tf.string)}

        parsed_features = tf.compat.v1.parse_single_example(example_proto, keys_to_features)

        img = parsed_features['label']
        img = tf.compat.v1.decode_raw(img, tf.uint8)
        img = tf.reshape(img, [self.back_size, self.back_size, self.CHANNEL])

        return img

    def _parse_function0(self, example_proto):
        keys_to_features = {'label': tf.compat.v1.FixedLenFeature([], tf.string)}

        parsed_features = tf.compat.v1.parse_single_example(example_proto, keys_to_features)

        img = parsed_features['label']
        img = tf.compat.v1.decode_raw(img, tf.uint8)
        img = tf.reshape(img, [self.face_size, self.face_size, self.CHANNEL])

        return img


    def load_tfrecord(self):
        dataset0 = tf.data.TFRecordDataset(self.tfrecord_path0)
        dataset0 = dataset0.map(self._parse_function0)

        dataset0 = dataset0.shuffle(self.buffer_size)
        dataset0 = dataset0.repeat()
        dataset0 = dataset0.batch(self.TASK_BATCH_SIZE*self.META_BATCH_SIZE)
        iterator0 = tf.compat.v1.data.make_one_shot_iterator(dataset0)


        dataset1 = tf.data.TFRecordDataset(self.tfrecord_path1)
        dataset1 =dataset1.map(self._parse_function1)

        dataset1 =dataset1.shuffle(self.buffer_size)
        dataset1 =dataset1.repeat()
        dataset1 =dataset1.batch(self.TASK_BATCH_SIZE*self.META_BATCH_SIZE)
        iterator1 = tf.compat.v1.data.make_one_shot_iterator(dataset1)

        label_train0 = iterator0.get_next()
        label_train1 = iterator1.get_next()

        return label_train0, label_train1
