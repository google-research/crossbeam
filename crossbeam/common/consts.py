from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import argparse
import logging
import numpy as np
import torch
import torch.nn as nn

t_float = torch.float32
np_float = np.float32
str_float = "float32"
N_INF = -1e9
EPS = 1e-18
