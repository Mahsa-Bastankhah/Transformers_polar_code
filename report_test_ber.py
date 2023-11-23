
import csv
import os
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

def calculate_mean_ber(start, end, K):
# Read the data from the CSV file

    
    with open(results_save_path + '/values_test.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        #print(data[0][start:end + 1])
    ber_noisy_str = data[1][start:end + 1]
    ber_noisy = []
    for x in ber_noisy_str:
        ber_noisy.append(float(x))
   

    # Extract relevant data based on start and end iterations
    ber_bitwise_noisy_str = [row[start:end + 1] for row in data[3 : 3 + K]] 
    ber_bitwise_noisy = []
    for x in ber_bitwise_noisy_str:
        row = []
        for y in x:
            row.append(float(y))
        ber_bitwise_noisy.append(row)
    # Calculate the mean for BER and bitwise BER
    mean_ber_noisy = sum(ber_noisy) / len(ber_noisy)
    mean_ber_bitwise_noisy = [sum(bits) / len(bits) for bits in ber_bitwise_noisy]

    # Print the mean values nicely
    print("Mean BER (Noisy):", round(mean_ber_noisy , 4))
    print("Mean Bitwise BER (Noisy):")
    for i, mean in enumerate(mean_ber_bitwise_noisy):
        print(f"  Info Bit {i}: {round(mean , 4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your script description here')
    parser.add_argument('start', type=int, help='start iter')
    parser.add_argument('end', type=int, help='end iter')
    parser.add_argument('N', type=int, help='Value for N')
    parser.add_argument('K', type=int, help='Value for K')
    parser.add_argument('run', type=int, help='Value for run')
    parser.add_argument('print_freq', type=int, help='Value for print_freq')
    parser.add_argument('--n_head', type=int, default=1,help='Value for n_head')
    parser.add_argument('--n_layers', type=int, default=3, help='Value for n_layers')
    args = parser.parse_args()

    results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}/{6}'\
                                    .format(args.K, args.N, "polar",  "encoder", args.n_head,args.n_layers, args.run)
       
    print("mean over iterations:" + str(args.start) + " and " + str(args.end))    

    calculate_mean_ber(int(args.start/args.print_freq) , int(args.end/args.print_freq), args.K)