
import argparse
import itertools
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
from collections import namedtuple
import sys
import csv
 # Generate all possible K-bit messages for the alphabet {-1, 1}
N = 16
K = 8
model = "encoder"
rate_profile = "polar"


data_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/data/'\
                                    .format(K, N, rate_profile,  model)
       
alphabet = np.array([-1, 1])
all_messages = np.array(list(itertools.product(alphabet, repeat=K)))

# Randomly select 20% of the messages as test data
test_size = int(0.2 * len(all_messages))
test_indices = np.random.choice(len(all_messages), test_size, replace=False)
test_data = all_messages[test_indices]

# Use the remaining messages as training data
training_data = np.delete(all_messages, test_indices, axis=0)

# Save test data
np.save(data_save_path + "/testData.npy" ,  test_data)
np.save(data_save_path + "/trainData.npy", training_data)

