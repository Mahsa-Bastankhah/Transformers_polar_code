from pyeda.inter import *
from itertools import product
import random
import pandas as pd
import csv
import matplotlib.pyplot as plt
import argparse
import numpy as np
import ast
from functools import reduce
import os
import math
import pickle
import json
from polar import *










parser = argparse.ArgumentParser(description='Your script description here')


#  attention tensor list = [n_layer * (batchsize * heads * N * N)] each () is a tensor
# Define mandatory arguments
parser.add_argument('N', type=int, help='Value for N')
parser.add_argument('K', type=int, help='Value for K')
parser.add_argument('run', type=int, help='Value for run')
# parser.add_argument('model_iters', type=int, help='model iteration')
# Define optional arguments
parser.add_argument('--snr', type=int, default=3,help='SNR')
parser.add_argument('--n_head', type=int, default=1,help='Value for n_head')
parser.add_argument('--n_layers', type=int, default=3, help='Value for n_layers')

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument values
N = args.N
K = args.K
run = args.run
args.hard_decision = False
# model_iters = args.model_iters
model = "encoder"
rate_profile = "polar"
n_head = args.n_head
n_layers = args.n_layers
mode = "combine"
results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}/{6}/code_tables/noisy/'\
                                        .format(K, N, rate_profile,  model, n_head,n_layers, run)


# List of model iterations
model_iters_list = [1000 * (i+1) for i in range(100)]

# Initialize an array to accumulate sum of bitwise differences
total_bitwise_difference_sum = np.zeros(16)
# Loop over model iterations
for model_iters in model_iters_list:
    # Load the training results DataFrame
    #train_results_path = results_save_path + '16bits_noiseless' + str(model_iters) + '.csv'  # Update with the actual path
    train_results_path = results_save_path + 'train_' + str(model_iters) + '.csv'  # Update with the actual path
    df_train = pd.read_csv(train_results_path)

    # # Load the test results DataFrame
    test_results_path = results_save_path + 'test_' + str(model_iters) + '.csv'
    df_test = pd.read_csv(test_results_path)
    

    # Convert string representations of lists to actual lists
    #df_train['Original_Message'] = df_train['Original_Message'].apply(ast.literal_eval)
    #df_train['Decoded_Bits'] = df_train['Decoded_Bits'].apply(ast.literal_eval)
    df_train['Binary_Corrupted_codeword'] = df_train['Binary_Corrupted_codeword'].apply(ast.literal_eval)
    df_train['Polar_code'] = df_train['Polar_code'].apply(ast.literal_eval)
    # df_test['Original_Message'] = df_test['Original_Message'].apply(ast.literal_eval)
    #df_test['Decoded_Bits'] = df_test['Decoded_Bits'].apply(ast.literal_eval)
    df_test['Binary_Corrupted_codeword'] = df_test['Binary_Corrupted_codeword'].apply(ast.literal_eval)
    df_test['Polar_code'] = df_test['Polar_code'].apply(ast.literal_eval)


    # Concatenate along rows (axis=0) to combine train and test DataFrames vertically
    combined_df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # If you want to concatenate along columns (axis=1), you can do the following:
    # combined_df = pd.concat([df_train, df_test], axis=1)

    # Reset index to ensure a continuous index for the new DataFrame
    combined_df = combined_df.reset_index(drop=True)


    # Convert the columns to NumPy arrays
    bin_codeword_array = combined_df['Binary_Corrupted_codeword'].apply(np.array).values
    polar_code_array = combined_df['Polar_code'].apply(np.array).values

    # Calculate the bitwise difference
    bitwise_difference = np.bitwise_xor(bin_codeword_array, polar_code_array)
    bitwise_difference_sum = np.sum(bitwise_difference, axis=0)

    total_bitwise_difference_sum += bitwise_difference_sum



print(total_bitwise_difference_sum)
#compare_with_sc()



   