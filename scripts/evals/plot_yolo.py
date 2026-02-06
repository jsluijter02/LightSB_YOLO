import os, sys
import numpy as np
from tqdm import tqdm

from PIL import Image

from scripts.utils import dirs

def plot_bounding_boxes(image_path, outdir, is_train=True):
    split = ''
    if is_train:
        split = 'train'
    else:
        split = 'val'

    image_json = os.path.splitext(os.path.basename(image_path))[0] + '.json'
    annotation_path = os.path.join(dirs.get_data_dir(), 'bdd', 'det_annotations', split, image_json)



