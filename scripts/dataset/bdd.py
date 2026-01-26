# Generatesa pickle file of the BDD database, so it doesn't have to be rebuilt every time you run YOLOPX dataset.
# Make sure to pass skip = true to YOLOPX BddDataset class
import os
import pickle as pkl

import scripts.utils.dirs as dirs
from scripts.utils.dirs import get_base_dir

dirs.add_YOLOPX_to_PATH()
import models.YOLOPX.lib.dataset as dataset

def pickle_path(is_train=False):
    if is_train:
        pkl_name = "full_train_dataset.pkl"
    else:
        pkl_name = "full_val_dataset.pkl"
    
    pkl_path = os.path.join(get_base_dir(), "data", "pkl_files", pkl_name)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    return pkl_path


# generates the pickled dataset.db files of either train or val, depending on is_train
def generate_bdd_db_pickles(cfg, is_train=True):
    data = dataset.BddDataset(
        cfg=cfg,
        is_train=is_train, 
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=None, 
        skip=False
    )
    with open(pickle_path(is_train), 'wb') as f:
        pkl.dump(data.db, f)

# Fetches the dataset.db object of either the train or val data, depending on is_train
def load_db(is_train=True):
    with open(pickle_path(is_train), 'rb') as f:
        db = pkl.load(f)
    return db

# Fetches either the Bdd training or validation data, depending on is_train
def get_bdd_dataset(cfg, is_train=True, skip=True, transform=None):
    data = dataset.BddDataset(
        cfg=cfg,
        is_train=is_train, 
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transform, 
        skip=skip
    )

    if skip:
        data.db = load_db(is_train=is_train)

    return data

def get_db(cfg, is_train, timeofday):
    full_dataset = get_bdd_dataset(cfg, is_train=is_train,skip=True)
    return full_dataset.select_data(timeofday)

# use thiss to remap images from data/bdd/images to data/SBimages
def remap_imgpath_db(db, new_root):
    for img in db:
        img_path = img['image']
        base = os.path.basename(img_path)
        new_path = os.path.join(new_root, base)
        img['image'] = new_path

    print("New image path: ", db[0]['image'])
    return db