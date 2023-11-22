import imageio
import os
import glob
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

def augmentation(x,mode):
    if mode ==0:
        y=x

    elif mode ==1:
        y=np.flipud(x)

    elif mode == 2:
        y = np.rot90(x,1)

    elif mode == 3:
        y = np.rot90(x, 1)
        y = np.flipud(y)

    elif mode == 4:
        y = np.rot90(x, 2)

    elif mode == 5:
        y = np.rot90(x, 2)
        y = np.flipud(y)

    elif mode == 6:
        y = np.rot90(x, 3)

    elif mode == 7:
        y = np.rot90(x, 3)
        y = np.flipud(y)

    return y

def imread(path):
    img = imageio.imread(path)
    return img

def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))

def write_to_tfrecord(writer, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
    }))
    writer.write(example.SerializeToString())
    return

def generate_TFRecord(label_path0, label_path1, tfrecord_file0, tfrecord_file1, patch_h, patch_w, stride):
    
    label_list0=np.sort(np.asarray(glob.glob(label_path0)))
    label_list1=np.sort(np.asarray(glob.glob(label_path1)))
    print(label_list1)
    offset=0

    fileNum0=len(label_list0)
    fileNum1=len(label_list1)
   
    labelsb=[]
    labelsa=[]

    writer = tf.io.TFRecordWriter("../datasets/" + tfrecord_file0)
    for n in range(fileNum0):
        print('[*] Image number: %d/%d' % ((n+1), fileNum0))
        label=imread(label_list0[n])

        x, y, ch = label.shape
        #for m in range(8):
        for i in range(0+offset,x-patch_h+1,stride):
            for j in range(0+offset,y-patch_w+1,stride):
                patch_l = label[i:i + patch_h, j:j + patch_w]
                write_to_tfrecord(writer, patch_l.tobytes())
    
                if np.log(gradients(patch_l.astype(np.float64)/255.)+1e-10) >= -6.0:
                    labelsb.append(patch_l.tobytes())

    #np.random.shuffle(labelsb)
    print('Num of patches:', len(labelsb))
    print('Shape: [%d, %d, %d]' % (patch_h, patch_w, ch))

    for i in range(len(labelsb)):
        if i % 10000 == 0:
            print('[%d/%d] Processed' % ((i+1), len(labelsb)))
       
    writer.close()

    writer = tf.io.TFRecordWriter("../datasets/" + tfrecord_file1)
    for n in range(30000):
        print('[*] Image number: %d/%d' % ((n+1), fileNum1))
        label = imread(label_list1[n])

        x, y, ch = label.shape
        #for m in range(8):
        for i in range(0+offset,x-patch_h+1,stride):
            for j in range(0+offset,y-patch_w+1,stride):
                patch_l = label[i:i + patch_h, j:j + patch_w]
                labelsa.append(patch_l)

    # np.random.shuffle(labelsa)
    print('Num of patches:', len(labelsa))
    print('Shape: [%d, %d, %d]' % (patch_h, patch_w, ch))

    writer = tf.io.TFRecordWriter(tfrecord_file1)
    for i in range(len(labelsa)):
        if i % 10000 == 0:
            print('[%d/%d] Processed' % ((i+1), len(labelsa)))
        write_to_tfrecord(writer, labelsa[i])

    writer.close()

if __name__=='__main__':

    labelpath0 = "./datasets/DF2K/DF2K_mutiscale"
    labelpath1 = "./datasets/ffhq1024/images256x256"
    labelpath0 = os.path.join(labelpath0, '*.png')
    labelpath1 = os.path.join(labelpath1, '*', '*.png')
    tfrecord_file0 = "back400" + '.tfrecord'
    tfrecord_file1 = "whole_face" + '.tfrecord'

    generate_TFRecord(labelpath0, labelpath1, tfrecord_file0, tfrecord_file1, 400, 400, 200)
    print('Done')