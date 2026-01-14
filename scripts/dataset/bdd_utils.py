import os, sys
import pickle as pkl
import torchvision.transforms as transforms

from scripts.utils.generate_cfg import generate_cfg
from scripts.utils.get_dirs import get_base_dir

sys.path.append(os.path.join(get_base_dir(), "models", "YOLOPX"))
import lib.dataset as dataset

# generates the pickled dataset.db files of either train or val, depending on is_train
def generate_bdd_pickles(is_train=True):
    cfg = generate_cfg()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    data = dataset.BddDataset(
        cfg=cfg,
        is_train=is_train, 
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), 
        skip=False 
    )

    if is_train:
        pkl_name = "full_train_dataset.pkl"
    else:
        pkl_name = "full_val_dataset.pkl"
    
    pkl_path = os.path.join(get_base_dir(), "data", "pkl_files", pkl_name)

    with open(pkl_path, 'wb') as f:
        pkl.dump(data.db, f)

# Fetches the dataset.db object of either the train or val data, depending on is_train
def get_bdd_db(is_train=True):
    if is_train:
        pkl_name = "full_train_dataset.pkl"
    else:
        pkl_name = "full_val_dataset.pkl"
    
    pkl_path = os.path.join(get_base_dir(), "data", "pkl_files", pkl_name)

    with open(pkl_path, 'rb') as f:
        db = pkl.load(f)
    
    return db

# Fetches either the Bdd training or validation data, depending on is_train
def get_bdd_dataset(is_train=True):
    cfg = generate_cfg()

    data = dataset.BddDataset(
        cfg=cfg,
        is_train=is_train, 
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=None, 
        skip=True
    )

    data.db = get_bdd_db(is_train=is_train)

    return data


if __name__ == "__main__":
    generate_bdd_pickles(is_train=True)
    generate_bdd_pickles(is_train=False)