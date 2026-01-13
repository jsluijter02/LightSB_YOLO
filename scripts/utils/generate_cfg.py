import argparse
import os,sys
from scripts.utils.get_dirs import get_base_dir

sys.path.append(os.path.join(get_base_dir(), "models", "YOLOPX"))

from lib.config import cfg, update_config


def generate_cfg(args=None):
    if args is not None:
        update_config(cfg, args)
        return cfg

    BASE_DIR = get_base_dir()

    # setup default args
    args = argparse.Namespace()
    args.iou_thres = 0.6
    args.conf_thres = 0.25
    args.weights = "../F../weights/epoch-195.pth"
    args.modelDir = ''
    args.logDir = 'runs/'

    # annotations directory setup
    args.da_seg_annotations = os.path.join(BASE_DIR, '', 'data/da_seg_annotations')
    args.det_annotations = os.path.join(BASE_DIR, '', 'data/det_annotations')
    args.images = os.path.join(BASE_DIR, '', 'data/images')
    args.ll_seg_annotations = os.path.join(BASE_DIR, '', 'data/ll_seg_annotations')

    update_config(cfg, args)
    return cfg

    
    