import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

path = osp.join(this_dir, 'face_detect')
add_path(path)

path = osp.join(this_dir, 'face_model')
add_path(path)
