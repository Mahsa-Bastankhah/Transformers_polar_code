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
import argparse
import pickle
import os
import time
from datetime import datetime
import matplotlib
#matplotlib.use('AGG')
import matplotlib.pyplot as plt


import math
import random
import numpy as np
#from tqdm import tqdm
from collections import namedtuple
import sys
import csv


parser = argparse.ArgumentParser(description='Your script description here')


#  attention tensor list = [n_layer * (batchsize * heads * N * N)] each () is a tensor
# Define mandatory arguments
parser.add_argument('N', type=int, help='Value for N')
parser.add_argument('K', type=int, help='Value for K')
parser.add_argument('snr', type=float, help='Value for snr')
parser.add_argument('run', type=int, help='Value for run')
parser.add_argument('print_freq', type=int, help='Value for print_freq')


# Define optional arguments
parser.add_argument('--n_head', type=int, default=1,help='Value for n_head')
parser.add_argument('--n_layers', type=int, default=6, help='Value for n_layers')
parser.add_argument('--model', type=str, default='encoder', help='Value for model')
parser.add_argument('--rate_profile', type=str, default='polar', help='Value for rate_profile')

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument values
N = args.N
K = args.K
snr = args.snr
run = args.run
print_freq = args.print_freq
n_head = args.n_head
n_layers = args.n_layers
model = args.model
rate_profile = args.rate_profile


# N = 8 
# K = 8 
# snr = 6
# run = 4
# print_freq = 300
# n_head = 1
# n_layers = 6


#head = 1
model = "encoder"


rate_profile = "polar"




results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                    .format(K, N, rate_profile,  model, n_head,n_layers)

results_save_path = results_save_path + '/' + '{0}'.format(run)


slf_attn_list_no_noise  = torch.load(results_save_path + "/attention_validation_no_noise.pth")
slf_attn_list  = torch.load(results_save_path + "/attention_validation.pth")
# there is one redundant dimension

#print(len(slf_attn_list_no_noise)) # number of iterations
#n_layers = len(slf_attn_list[0]) # num of layers
#n_head = len(slf_attn_list[0][0]) # num of heads
#print(len(slf_attn_list[0][0][0])) # N
#print(len(slf_attn_list[0][0][0][0])) # N



global_max = -float('inf')
global_min = float('inf')

# Iterate over slf_attn_list_no_noise and slf_attn_list
for iter in range(len(slf_attn_list_no_noise)):
    for layer in range(n_layers):
        tensor1 = slf_attn_list_no_noise[iter][layer]
        tensor2 = slf_attn_list[iter][layer]
        
        #for tensor in slf_attn_list_no_noise + slf_attn_list:
            # Extract the head-1 tensor (assuming tensor_on_cpu is already defined)
            #tensor = tensor.cpu()
            
            # Find the maximum and minimum within the current tensor
        local_max1 = tensor1.max().item()
        local_min1 = tensor1.min().item()
        local_max2 = tensor2.max().item()
        local_min2 = tensor2.min().item()
            
            # Update global maximum and minimum
        global_max = max(global_max, local_max1, local_max2)
        global_min = min(global_min, local_min1, local_min2)


for iter in range(len(slf_attn_list_no_noise)):  
    print(iter)
    for layer in range(n_layers):
        for head in range(n_head):
            tensor_on_cpu = slf_attn_list_no_noise[iter][layer][head].cpu()
            #print(tensor_on_cpu.type)
            #print(tensor_on_cpu.shape)
            #print(tensor_on_cpu)
            plt.figure(figsize=(N, N))
            plt.imshow(tensor_on_cpu, cmap='viridis', interpolation='nearest', vmin=global_min, vmax = global_max)
            plt.title('Noiseless-S' + str(iter * print_freq) + 'L' + str(layer+1) + 'H' + str(head+1), fontname='sans-serif')
            plt.colorbar()
            folder_path = results_save_path + '/figures/Attention/Noiseless/L' + str(layer+1) + 'H' + str(head+1)
            if not os.path.exists(folder_path):
                # If it doesn't exist, create the folder
                os.makedirs(folder_path)
    
            plt.savefig(folder_path + '/Step_{0}.png'.format(iter * print_freq), format='png')




for iter in range(len(slf_attn_list)):  
    print(iter)
    for layer in range(n_layers):
        for head in range(n_head):
            tensor_on_cpu = slf_attn_list[iter][layer][head].cpu()
            plt.figure(figsize=(N, N))
            plt.imshow(tensor_on_cpu, cmap='viridis', interpolation='nearest', vmin=global_min, vmax = global_max)
            plt.title('Noisy-snr' + str(snr) + 'S' + str(iter * print_freq) + 'L' + str(layer+1) + 'H' + str(head+1), fontname='sans-serif')
            plt.colorbar()
            folder_path = results_save_path + '/figures/Attention/Noisy-SNR'+ str(snr)+'/L' + str(layer+1) + 'H' + str(head+1)
            if not os.path.exists(folder_path):
                # If it doesn't exist, create the folder
                os.makedirs(folder_path)
    
            plt.savefig(folder_path + '/Step_{0}.png'.format(iter * print_freq), format='png')

# for i , tensor in enumerate(slf_attn_list):
#     tensor_on_cpu = tensor.cpu()
#     plt.figure(figsize=(N, N))
#     plt.imshow(tensor_on_cpu[head-1], cmap='viridis', interpolation='nearest', vmin=global_min, vmax = global_max)
#     plt.title('Noisy, SNR = ' + str(snr) + ', Step : ' + str(i * print_freq))
#     plt.colorbar()
#     folder_path = results_save_path + '/figures/Attention/Noisy_SNR' + str(snr)
#     if not os.path.exists(folder_path):
#         # If it doesn't exist, create the folder
#         os.makedirs(folder_path)
   
#     plt.savefig(folder_path + '/Head_{0}-Step_{1}.png'.format(head, i * print_freq), format='png')