import argparse
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




N = 4 
K = 4 
snr = -2
run = 4
print_freq = 300


head = 1
model = "encoder"
n_head = 8
n_layers = 6
head = 1
rate_profile = "polar"




results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                    .format(K, N, rate_profile,  model, n_head,n_layers)

results_save_path = results_save_path + '/' + '{0}'.format(run)


slf_attn_list_no_noise  = torch.load(results_save_path + "/attention_validation_no_noise.pth")
slf_attn_list  = torch.load(results_save_path + "/attention_validation.pth")

#print(slf_attn_list_no_noise.shape)



global_max = -float('inf')
global_min = float('inf')

# Iterate over slf_attn_list_no_noise and slf_attn_list
for tensor in slf_attn_list_no_noise + slf_attn_list:
    # Extract the head-1 tensor (assuming tensor_on_cpu is already defined)
    tensor = tensor.cpu()[head-1]
    
    # Find the maximum and minimum within the current tensor
    local_max = tensor.max().item()
    local_min = tensor.min().item()
    
    # Update global maximum and minimum
    global_max = max(global_max, local_max)
    global_min = min(global_min, local_min)

for i , tensor in enumerate(slf_attn_list_no_noise):
    tensor_on_cpu = tensor.cpu()
    print(i)
    plt.figure(figsize=(N, N))
    plt.imshow(tensor_on_cpu[head-1], cmap='viridis', interpolation='nearest', vmin=global_min, vmax = global_max)
    plt.title('Noiseless, Step : ' + str(i * print_freq))
    plt.colorbar()
    folder_path = results_save_path + '/figures/Attention/Noiseless'
    if not os.path.exists(folder_path):
        # If it doesn't exist, create the folder
        os.makedirs(folder_path)
   
    plt.savefig(folder_path + '/Head_{0}-Step_{1}.png'.format(head, i * print_freq), format='png')



for i , tensor in enumerate(slf_attn_list):
    tensor_on_cpu = tensor.cpu()
    plt.figure(figsize=(N, N))
    plt.imshow(tensor_on_cpu[head-1], cmap='viridis', interpolation='nearest', vmin=global_min, vmax = global_max)
    plt.title('Noisy, SNR = ' + str(snr) + ', Step : ' + str(i * print_freq))
    plt.colorbar()
    folder_path = results_save_path + '/figures/Attention/Noisy_SNR' + str(snr)
    if not os.path.exists(folder_path):
        # If it doesn't exist, create the folder
        os.makedirs(folder_path)
   
    plt.savefig(folder_path + '/Head_{0}-Step_{1}.png'.format(head, i * print_freq), format='png')