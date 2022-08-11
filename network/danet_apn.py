from __future__ import division
import numpy as np
from .da_att import PAM_Module, CAM_Module
from torch.nn.functional import interpolate as interpolate
from .backbone import *
import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.scatter_gather import scatter


