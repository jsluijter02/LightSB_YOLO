import os, sys
import numpy
from numpy import cov
from numpy import iscomplexobj
from numpy import trace
from scipy.linalg import sqrtm

import torch
from torch import Tensor

# source: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
def latent_FID_score(list1: Tensor, list2: Tensor):
    list1_np = list1.detach().cpu().numpy()
    list2_np = list2.detach().cpu().numpy()

    mu1, sigma1 = list1_np.mean(axis=0), cov(list1_np, rowvar=False)
    mu2, sigma2 = list2_np.mean(axis=0), cov(list2_np, rowvar=False)

    ssdiff = numpy.sum((mu1-mu2)**2.0)

    covmean = sqrtm(sigma1.dot(sigma2))

    if iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid