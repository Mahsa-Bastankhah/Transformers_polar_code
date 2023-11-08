import argparse
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
import functools

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

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






# Access the argument values




def generate_attention_maps(N, K, snr, run, print_freq, n_head, n_layers, model, rate_profile, compositional,oe):

    results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                        .format(K, N, rate_profile,  model, n_head,n_layers)

    results_save_path = results_save_path + '/' + '{0}'.format(run)
    
    if oe:
        slf_attn_no_noise  = torch.load(results_save_path + "/attention_validation_no_noise_oe.pth")
        slf_attn = torch.load(results_save_path + "/attention_validation_oe.pth")
    else:
        slf_attn_no_noise  = torch.load(results_save_path + "/attention_validation_no_noise.pth")
        slf_attn = torch.load(results_save_path + "/attention_validation.pth")
    


    
    slf_attn_list = [[[[[0.0] * N for _ in range(N)] for _ in range(n_head)] for _ in range(n_layers)] for _ in range(len(slf_attn))]
    slf_attn_list_no_noise = [[[[[0.0] * N for _ in range(N)] for _ in range(n_head)] for _ in range(n_layers)] for _ in range(len(slf_attn))]


    # convert the data to a list
    for iter in range(len(slf_attn)):  
        for layer in range(len(slf_attn[0])):
            for head in range(len(slf_attn[0][0])):
                slf_attn_list[iter][layer][head] = slf_attn[iter][layer][head].cpu().tolist()
                slf_attn_list_no_noise[iter][layer][head] = slf_attn_no_noise[iter][layer][head].cpu().tolist()
    
    # print(len(slf_attn_list_no_noise)) # number of iterations
    # print(len(slf_attn_list[0])) # num of layers
    # print(len(slf_attn_list[0][0])) # num of heads
    # print(len(slf_attn_list[0][0][0])) # N
    # print(len(slf_attn_list[0][0][0][0])) # N
    #print(slf_attn_list[0][0][0].type())
    
    if compositional:
       # multiply attention maps across layers to find the compositional map     

        slf_attn_list = np.prod(slf_attn_list, axis=1, keepdims=True)
        slf_attn_list = slf_attn_list.reshape(slf_attn_list.shape[0], 1, slf_attn_list.shape[2], slf_attn_list.shape[3], slf_attn_list.shape[4])
        slf_attn_list = slf_attn_list.tolist()
        
        slf_attn_list_no_noise = np.prod(slf_attn_list_no_noise, axis=1, keepdims=True)
        slf_attn_list_no_noise = slf_attn_list_no_noise.reshape(slf_attn_list_no_noise.shape[0], 1, slf_attn_list_no_noise.shape[2], slf_attn_list_no_noise.shape[3], slf_attn_list_no_noise.shape[4])
        slf_attn_list_no_noise = slf_attn_list_no_noise.tolist()
        # print(len(slf_attn_list_no_noise)) # number of iterations
        # print(len(slf_attn_list[0])) # num of layers
        # print(len(slf_attn_list[0][0])) # num of heads
        # print(len(slf_attn_list[0][0][0])) # N
        # print(len(slf_attn_list[0][0][0][0])) # N






    # Finding the appropriate range for drawing the maps
    global_max = -float('inf')
    global_min = float('inf')

    # Iterate over slf_attn_list_no_noise and slf_attn_list
    for iter in range(len(slf_attn_list)):
        for layer in range(len(slf_attn_list_no_noise[0])):
            tensor1 = slf_attn_list_no_noise[iter][layer]
            tensor2 = slf_attn_list[iter][layer]
                
                # Find the maximum and minimum within the current tensor
            local_max1 = np.max(tensor1)
            local_min1 = np.min(tensor1)
            local_max2 = np.max(tensor2)
            local_min2 = np.min(tensor2)
                
                # Update global maximum and minimum
            global_max = max(global_max, local_max1, local_max2)
            global_min = min(global_min, local_min1, local_min2)


    for iter in range(len(slf_attn_list_no_noise)):  
        #print(iter)
        for layer in range(len(slf_attn_list_no_noise[0])):
            for head in range(len(slf_attn_list_no_noise[0][0])):
                #tensor_on_cpu = slf_attn_list_no_noise[iter][layer][head].cpu()
                plt.figure(figsize=(N, N))
                plt.imshow(slf_attn_list_no_noise[iter][layer][head], cmap='viridis', interpolation='nearest', vmin=global_min, vmax = global_max)
                plt.title('Noiseless-S' + str(iter * print_freq) + 'L' + str(layer+1) + 'H' + str(head+1), fontname='sans-serif')
                plt.colorbar()
                if compositional == False:
                    folder_path = results_save_path + '/figures/Attention-' + str(oe)+ '/Noiseless/L' + str(layer+1) + 'H' + str(head+1)
                else:
                    folder_path = results_save_path + '/figures/AttentionCompos/Noiseless/H' + str(head+1)
                if not os.path.exists(folder_path):
                    # If it doesn't exist, create the folder
                    os.makedirs(folder_path)
        
                plt.savefig(folder_path + '/Step_{0}.png'.format(iter * print_freq), format='png', bbox_inches='tight', pad_inches=0)
                # tight margin
                # pdf_pages = PdfPages(folder_path + '/Step_{0}.pdf'.format(iter * print_freq))
                # pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=0)
                # pdf_pages.close()

                plt.close()  # Close the figure



    for iter in range(len(slf_attn_list)):  
        #print(iter)
        for layer in range(len(slf_attn_list[0])):
            for head in range(len(slf_attn_list[0][0])):
                #tensor_on_cpu = slf_attn_list[iter][layer][head].cpu()
                plt.figure(figsize=(N,N))
                plt.imshow(slf_attn_list[iter][layer][head], cmap='viridis', interpolation='nearest', vmin=global_min, vmax = global_max)
                plt.title('Noisy-snr' + str(snr) + 'S' + str(iter * print_freq) + 'L' + str(layer+1) + 'H' + str(head+1), fontname='sans-serif')
                plt.colorbar()
                if compositional==False:
                    folder_path = results_save_path + '/figures/Attention-' +  str(oe)+ '/Noisy-SNR'+ str(snr)+'/L' + str(layer+1) + 'H' + str(head+1)
                else:
                    folder_path = results_save_path + '/figures/AttentionCompos/Noisy-SNR'+ str(snr)+ '/H' + str(head+1)
                if not os.path.exists(folder_path):
                    # If it doesn't exist, create the folder
                    os.makedirs(folder_path)
        
                plt.savefig(folder_path + '/Step_{0}.png'.format(iter * print_freq), format='png', bbox_inches='tight', pad_inches=0)
                # pdf_pages = PdfPages(folder_path + '/Step_{0}.pdf'.format(iter * print_freq))
                # pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=0)
                # pdf_pages.close()

                plt.close()  # Close the figure





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your script description here')
    # Add your argument parsing code here if needed
    
    #  attention tensor list = [n_layer * (batchsize * heads * N * N)] each () is a tensor
    # Define mandatory arguments
    parser.add_argument('--compos', action="store_true", help='do you want compositional attention map or normal ones?')
    parser.add_argument('--oe', action="store_true", help='one example')
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
    N = args.N
    K = args.K
    snr = args.snr
    run = args.run
    print_freq = args.print_freq
    n_head = args.n_head
    n_layers = args.n_layers
    model = args.model
    rate_profile = args.rate_profile
    compos = args.compos
    oe = args.oe

    generate_attention_maps(N,K,snr,run,print_freq,n_head,n_layers,model,rate_profile,compos,oe)
    


