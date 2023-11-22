import os
from config import *
from train import Train
from DataGenerator import DataGenerator

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=f"{args.gpu}"

conf=tf.compat.v1.ConfigProto()
conf.gpu_options.allow_growth = True

def main():
    data_generator = DataGenerator(output_shape = [HEIGHT, WIDTH, CHANNEL, HEIGHT1, WIDTH1], 
                                               meta_batch_size = META_BATCH_SIZE,
                                               task_batch_size = TASK_BATCH_SIZE,
                                               tfrecord_path0 = TFRECORD_PATH0,
                                               tfrecord_path1 = TFRECORD_PATH1,
                                            )
    Trainer = Train(trial = args.trial, 
                          step = args.step, 
                          size = [HEIGHT, WIDTH, CHANNEL, HEIGHT1, WIDTH1],
                          meta_batch_size = META_BATCH_SIZE, 
                          meta_lr = META_LR, 
                          meta_iter = META_ITER, 
                          task_batch_size = TASK_BATCH_SIZE,
                          task_lr = TASK_LR, 
                          task_iter = TASK_ITER, 
                          data_generator = data_generator, 
                          checkpoint_dir = CHECKPOINT_DIR, 
                          conf=conf
                        )
    Trainer()


if __name__=='__main__':
    main()