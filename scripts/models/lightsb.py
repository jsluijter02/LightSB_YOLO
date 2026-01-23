import torch
from yacs.config import CfgNode as CN
import argparse
from tqdm import tqdm

from scripts.utils.device import get_device
from scripts.utils import dirs

dirs.add_LIGHTSB_to_PATH()
from models.LightSB.src.light_sb import LightSB # make sure to add models/LightSB to path!!
from models.LightSB.src.distributions import TensorSampler

## model 
class LightSB_BDD:
    def __init__(self, config: CN, np_data):
        self.config = config

        # only night -> day
        self.X_train = torch.tensor(np_data["train_night"], device=get_device())
        self.Y_train = torch.tensor(np_data["train_day"], device=get_device())
        self.X_test = torch.tensor(np_data["val_night"], device=get_device())
        self.Y_test = torch.tensor(np_data["val_day"], device=get_device())

        self.X_sampler = TensorSampler(self.X_train, device=get_device())
        self.Y_sampler = TensorSampler(self.Y_train, device=get_device())

        self.model = LightSB(dim=self.config.DIM,
                             n_potentials=self.config.MODEL.N_POTENTIALS,
                             epsilon=self.config.MODEL.EPSILON,
                             sampling_batch_size = self.config.MODEL.SAMPLING_BATCH_SIZE,
                             S_diagonal_init=0.1,
                             is_diagonal=self.config.MODEL.IS_DIAGONAL).to(get_device())
        
        if self.config.MODEL.INIT_BY_SAMPLES:
            self.model.init_r_by_samples(self.Y_sampler.sample(self.config.MODEL.N_POTENTIALS))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.MODEL.D_LR)
    
    def train(self):
        for step in tqdm(range(self.config.CONTINUE+1, self.config.MAX_STEPS)):
            self.optimizer.zero_grad()

            X0, X1 = self.X_sampler.sample(self.config.MODEL.BATCH_SIZE), self.Y_sampler.sample(self.config.MODEL.BATCH_SIZE)

            log_potential = self.model.get_log_potential(X1)
            log_C = self.model.get_log_C(X0)

            loss = (-log_potential + log_C).mean()
            loss.backward()

            self.optimizer.step() 

    def transform(self, samples):
        with torch.inference_mode():
            transformed = self.model(samples)
            return transformed
    
    def update_config(self, args: argparse.Namespace):
        self.config.defrost()
        if hasattr(args, "DIM"):
            self.config.DIM = args.DIM
        
        # model param changes
        if hasattr(args, "BATCH_SIZE"):
            self.config.MODEL.BATCH_SIZE = args.BATCH_SIZE
        if hasattr(args,"EPSILON"):
            self.config.MODEL.EPSILON = args.EPSILON
        if hasattr(args,"D_LR"):
            self.config.MODEL.D_LR = args.D_LR
        if hasattr(args, "N_POTENTIALS"):
            self.config.MODEL.N_POTENTIALS = args.N_POTENTIALS
        if hasattr(args, "IS_DIAGONAL"):
            self.config.MODEL.IS_DIAGONAL = args.IS_DIAGONAL
        
        # max steps
        if hasattr(args, "MAX_STEPS"):
            self.config.MAX_STEPS = args.MAX_STEPS

        self.config.freeze()

    # after updating the config, you can reload the model with this function
    def reload_model(self):
        self.model = LightSB(dim=self.config.DIM,
                             n_potentials=self.config.MODEL.N_POTENTIALS,
                             epsilon=self.config.MODEL.EPSILON,
                             sampling_batch_size = self.config.MODEL.SAMPLING_BATCH_SIZE,
                             S_diagonal_init=0.1,
                             is_diagonal=self.config.MODEL.IS_DIAGONAL).to(get_device())
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    ### Static get config methsods
    @staticmethod
    def standard_config():
        _C = CN()
        _C.DIM = 4096
        _C.INPUT_DATA = "night"
        _C.TARGET_DATA = "daytime"

        # CONFIGS FROM LightSB_alae.ipynb
        _C.OUTPUT_SEED = 0xBADBEEF

        _C.MODEL = CN()
        _C.MODEL.BATCH_SIZE = 128
        _C.MODEL.EPSILON = 0.1
        _C.MODEL.D_LR = 1e-3 # 1e-3 for eps 0.1
        _C.MODEL.D_GRADIENT_MAX_NORM = float("inf")
        _C.MODEL.N_POTENTIALS = 10
        _C.MODEL.SAMPLING_BATCH_SIZE = 128
        _C.MODEL.INIT_BY_SAMPLES = True
        _C.MODEL.IS_DIAGONAL = True

        _C.MAX_STEPS = 10000
        _C.CONTINUE = -1

        _C.EXP_NAME = f'LightSB_{_C.INPUT_DATA}_TO_{_C.TARGET_DATA}_EPSILON_{_C.MODEL.EPSILON}'
        _C.OUTPUT_PATH = '../checkpoints/{}'.format(_C.EXP_NAME)

        return _C
      
    
## eval