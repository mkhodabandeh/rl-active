import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))

root_path = osp.abspath(root_dir)
add_path(root_path)
