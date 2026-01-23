# Wrappers for the YOLOPX repo and the ultralytics yolov12n
import os
import pprint
from yacs.config import CfgNode as CN
from abc import ABC, abstractmethod
import torch
import torchvision.transforms as transforms
import numpy as np

# YOLOPX imports
from tensorboardX import SummaryWriter
from models.YOLOPX.lib.config import cfg, update_config
from models.YOLOPX.lib.models import get_net
from models.YOLOPX.lib.core.loss import get_loss
import models.YOLOPX.lib.dataset as dataset
from models.YOLOPX.lib.utils.utils import DataLoaderX
from models.YOLOPX.lib.core.function import validate
from models.YOLOPX.lib.utils.utils import create_logger
from models.YOLOPX.lib.core.general import fitness


# own repo imports
from scripts.utils import dirs
from scripts.utils.device import get_device
from scripts.dataset import bdd

class YOLO_BDD(ABC):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg

    def set_image_path(self, path):
        self.config.DATASET.DATAROOT = os.path.join(dirs.get_data_dir(), path)

    @abstractmethod
    def validate(self, SB:bool= False):
        pass

    @abstractmethod
    def setup_model(self):
        pass
    
    @abstractmethod
    def update_config(self):
        pass

# mostly follows yolopx test code, but this wrapper makes it easier to write notebooks with
class YOLOPX_BDD(YOLO_BDD):
    def __init__(self, args:CN=None):
        super().__init__(cfg)
        self.weights = dirs.get_YOLOPX_weights()
        self.device = get_device()

        if args:
            self.update_config(args)

        self.logger, self.final_output_dir, self.tb_log_dir = create_logger(
        self.config, self.config.LOG_DIR, 'test')

        self.logger.info(cfg)

        self.writer_dict = {
            'writer': SummaryWriter(log_dir=self.tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }

        self.setup_model()

    def setup_model(self):
        self.model = get_net(self.config)
        self.criterion = get_loss(self.config, self.device, self.model)

        # get the last weights from the checkpoint
        checkpoint = torch.load(self.weights, map_location=self.device)
        checkpoint_dict = checkpoint['state_dict']

        # update the model weights to the checkpoint weights
        new_dict = self.model.state_dict()
        new_dict.update(checkpoint_dict)
        self.model.load_state_dict(new_dict)

        self.model = self.model.to(self.device)
    
        self.model.gr = 1.0
        self.model.nc = 1

    def validate(self, timeofday='night'):
        # get the dataset
        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        validation_set = bdd.get_bdd_dataset(self.config, is_train=False, skip=True, transform=transf)
        validation_set.db = bdd.get_db(self.config, False, timeofday)

        msg = f'Num validation "{timeofday}" images: {len(validation_set.db)}'
        self.logger.info(msg)

        # dataset loader
        valid_loader = DataLoaderX(
            validation_set,
            batch_size=self.config.TEST.BATCH_SIZE_PER_GPU * len(self.config.GPUS),
            shuffle=False,
            num_workers=self.config.WORKERS,
            pin_memory=False,
            collate_fn=dataset.AutoDriveDataset.collate_fn
        )

        epoch = 0
        with torch.inference_mode():
            da_segment_results,ll_segment_results,detect_results, total_loss, maps, times = validate(
                epoch, self.config, valid_loader, validation_set, self.model, self.criterion,
                self.final_output_dir, self.tb_log_dir, self.writer_dict,
                self.logger, self.device, save_error_plots = True ## ADDED
            )

        fi = fitness(np.array(detect_results).reshape(1, -1))
        msg =   'Test:    Loss({loss:.3f})\n' \
                'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                        'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                        'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                        'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                            loss=total_loss, da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                            ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2],
                            p=detect_results[0],r=detect_results[1],map50=detect_results[2],map=detect_results[3],
                            t_inf=times[0], t_nms=times[1])
        self.logger.info(msg)
        print(msg)

    def update_config(self, args: CN):
        if hasattr(args, "WEIGHTS"):
            self.weights = args.WEIGHTS
        if hasattr(args, "IMAGES"):
            self.config.DATASET.DATAROOT = args.IMAGES

class YOLOv12n_BDD(YOLO_BDD):
    def __init__(self, db, img_path, annt_path, model):
        super().__init__(db, img_path, annt_path, model)

    def train(self):
        pass

    def finetune(self):
        pass

    def validate(self):
        pass

    def generate_backbone_encodings(db):
        pass

    def standard_config(self):
        pass

    def update_config(self, args, cfg: CN = None):
        if cfg == None:
            cfg = self.standard_config()

        ## hasattr