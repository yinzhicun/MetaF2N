from argparse import ArgumentParser

parser=ArgumentParser()

# Global
parser.add_argument('--gpu', type=str, dest='gpu', default='0')
# For Meta-Training
parser.add_argument('--trial', type=int, dest='trial', default=0)
parser.add_argument('--step', type=int, dest='step', default=0)

args= parser.parse_args()

# Dataset Options
HEIGHT = 128
WIDTH = 128
CHANNEL = 3

HEIGHT1 = 256 
WIDTH1 = 256

META_ITER = 400000
META_BATCH_SIZE = 4
META_LR = 3e-5

TASK_ITER = 1
TASK_BATCH_SIZE = 4
TASK_LR = 1e-2

# inner loop data
TFRECORD_PATH0 = '../MetaF2N/MainSR/whole_face.tfrecord'
# outer loop data
TFRECORD_PATH1 = '../MetaF2N/MainSR/back400.tfrecord'
CHECKPOINT_DIR='./weights'
