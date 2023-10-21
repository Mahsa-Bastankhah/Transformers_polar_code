
import argparse
import parser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import pandas as pd

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from IPython import display

import pickle
import os
import time
from datetime import datetime
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from utils import snr_db2sigma, errors_ber, errors_bitwise_ber, errors_bler, min_sum_log_sum_exp, moving_average, extract_block_errors, extract_block_nonerrors
from models import convNet,XFormerEndToEndGPT,XFormerEndToEndDecoder,XFormerEndToEndEncoder,rnnAttn
from polar import *
from pac_code import *

import math
import random
import numpy as np
from tqdm import tqdm
from collections import namedtuple
import sys
import csv

K = 8
N = 8
rate_profile = "Polar"
model = "encoder"
n_head = 8
n_layers = 6
run = 1

parser.add_argument('--batch_size', type=int, default=512, help='validation batch size for predefined validation data')

args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

randperm = torch.randperm(args.batch_size)

msg_bits = 1 - 2 * (torch.rand(args.batch_size, K, device=device) < 0.5).float()

results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                    .format(K, N, rate_profile,  model, n_head, n_layers)

results_save_path = results_save_path + '/' + '{0}'.format(run) 

torch.save(msg_bits, results_save_path + '/validation_msg_bits.pth')