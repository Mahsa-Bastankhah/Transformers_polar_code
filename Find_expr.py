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






def prepare_data_binary(results_save_path, model_iters, output_idx):
    # Load the training results DataFrame
    train_results_path = results_save_path + 'train_' + str(model_iters) + '.csv'  # Update with the actual path
    df_train = pd.read_csv(train_results_path)

    # Load the test results DataFrame
    test_results_path = results_save_path + 'test_' + str(model_iters) + '.csv'
    df_test = pd.read_csv(test_results_path)

    # Convert string representations of lists to actual lists
    df_train['Original_Message'] = df_train['Original_Message'].apply(ast.literal_eval)
    df_train['Decoded_Bits'] = df_train['Decoded_Bits'].apply(ast.literal_eval)
    df_train['Binary_Corrupted_codeword'] = df_train['Binary_Corrupted_codeword'].apply(ast.literal_eval)

    df_test['Original_Message'] = df_test['Original_Message'].apply(ast.literal_eval)
    df_test['Decoded_Bits'] = df_test['Decoded_Bits'].apply(ast.literal_eval)
    df_test['Binary_Corrupted_codeword'] = df_test['Binary_Corrupted_codeword'].apply(ast.literal_eval)


   # Extracting the labels and input bits for training data
    train_table = df_train.apply(lambda row: tuple(row['Binary_Corrupted_codeword'] + [row['Decoded_Bits'][output_idx]]), axis=1).tolist()
    # Print the dimensions
    

    
    # Extracting the labels and input bits for test data
    test_table = df_test.apply(lambda row: tuple(row['Binary_Corrupted_codeword'] + [row['Decoded_Bits'][output_idx]]), axis=1).tolist()
    # for i in range(len(test_table)):

    #     print(test_table[i][15], test_table[i][16] )
    # # Combine train and test truth tables
    return  train_table, test_table




def prepare_data_anal(results_save_path, model_iters):
    # Load the training results DataFrame
    train_results_path = results_save_path + 'train_' + str(model_iters) + '.csv'  # Update with the actual path
    df_train = pd.read_csv(train_results_path)

    # Load the test results DataFrame
    test_results_path = results_save_path + 'test_' + str(model_iters) + '.csv'
    df_test = pd.read_csv(test_results_path)

    # Convert string representations of lists to actual lists
    df_train['Original_Message'] = df_train['Original_Message'].apply(ast.literal_eval)
    df_train['Output'] = df_train['Output'].apply(ast.literal_eval)
    df_train['Corrupted_codeword'] = df_train['Corrupted_codeword'].apply(ast.literal_eval)
    df_train['Decoded_Bits'] = df_train['Decoded_Bits'].apply(ast.literal_eval)
    #print(df_train['Decoded_Bits'])
    # Create a new DataFrame with two columns
    new_df_train = pd.DataFrame({
        'Original_Message' :df_train['Original_Message'],
        'Decoded_Bits': df_train['Decoded_Bits'],
        'Corrupted_codeword': df_train['Corrupted_codeword']
    })

    df_test['Original_Message'] = df_test['Original_Message'].apply(ast.literal_eval)
    df_test['Output'] = df_test['Output'].apply(ast.literal_eval)
    df_test['Corrupted_codeword'] = df_test['Corrupted_codeword'].apply(ast.literal_eval)
    df_test['Decoded_Bits'] = df_test['Decoded_Bits'].apply(ast.literal_eval)

    # Create a new DataFrame with two columns
    new_df_test = pd.DataFrame({
        'Original_Message' :df_test['Original_Message'],
        'Decoded_Bits': df_test['Decoded_Bits'],
        'Corrupted_codeword': df_test['Corrupted_codeword']
    })
   
    return  new_df_train, new_df_test


def calculate_error(table, weights):
    # Calculate the error for a given set of weights
    errors = 0
    for *inputs, label in table :
        if sum(x * w for x, w in zip(inputs, weights)) % 2 != label:
            errors = errors + 1
    return errors/len(table)

def calculate_mse_error(table, weights):
    # Calculate the error for a given set of weights
    mse_error = 0
    predicted_output = 0
    for *inputs, label in table :
        print(inputs, weights)
        for x, w in zip(inputs, weights):
            predicted_output *= x ** w
        print(predicted_output)
        error = math.copysign(1 , predicted_output) != math.copysign(1 , label)
        mse_error += error
    mse_error = mse_error/ len(table) 
    return mse_error

def find_best_weights(table, m):
    num_inputs = len(table[0]) - 1  # Number of input bits
    weight_combinations = product([0, 1], repeat=num_inputs)  # All possible weight combinations

    # Initialize a list to store the top m weight-error pairs
    top_weights = [(None, float('inf'))] * m

    for weights in weight_combinations:
        error = calculate_error(table, weights)

        # Check if the current weights have a lower error than any in the top m
        for i, (_, top_error) in enumerate(top_weights):
            if error < top_error:
                top_weights.insert(i, (weights, round(error,4)))
                top_weights = top_weights[:m]  # Keep only the top m weights
                break
            
        

    return top_weights

def evaluate_all_weights(truth_table):
    num_inputs = len(truth_table[0]) - 1  # Number of input bits
    weight_combinations = product([0, 1], repeat=num_inputs)  # All possible weight combinations

    # Initialize a dictionary to store the best weights and associated errors for each count of non-zero elements
    best_weights_dict = {i: (None, float('inf')) for i in range(17)}

    for weights in weight_combinations:
        num_non_zero = sum(weights)
        error = calculate_error(truth_table, weights) / len(truth_table)

        # Check if the current weights have a lower error than any in the best weights for the current count of non-zero elements
        if error < best_weights_dict[num_non_zero][1]:
            best_weights_dict[num_non_zero] = (weights, error)

    return best_weights_dict




def compare_with_sc():
    n = int(np.log2(args.N))
    info_inds = [7,9,10,11,12,13,14,15]
    rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1
    # Multiple SNRs:
    
    polar = PolarCode(n, args.K, args, rs=rs)
    # Convert the train_table to a DataFrame
    # Extracting corrupted codewords and labels for test data
    iters = [1000 * (i+1) for i in range(1)]
    #iters = [10000]
    bitwise_error_arr = []
    for (i , iter) in enumerate(iters):
        print("=========== Iteration " + str(iter) + "===========")
        for mode in ["combined"]:
            print( mode + " data")

            df_train, df_test = prepare_data_anal(results_save_path, iter)
            df = ""
            if mode == "train":
                df = df_train
            elif mode == "test":
                df = df_test
            else:
                df = pd.concat([df_test, df_train], ignore_index=True)


            codewords = np.array([np.array(item) for item in df['Corrupted_codeword']])
            trnsf_decoded = np.array([np.array(item) for item in df['Decoded_Bits']])
            original_msg = np.array([np.array(item) for item in df['Original_Message']])



            # Convert the NumPy array to a PyTorch tensor and move it to the CPU
            corrupted_codewords_tensor = torch.tensor(codewords, dtype=torch.float32, device='cpu')

            SC_llrs, decoded_SC_msg_bits = polar.sc_decode_new(corrupted_codewords_tensor , args.snr )
            decoded_bits_np = decoded_SC_msg_bits.numpy()
            
    
            decoded_bits_np[decoded_bits_np == 1] = 0
            decoded_bits_np[decoded_bits_np == -1] = 1

       
            
            total_bits_per_bit = trnsf_decoded.size  # Assuming gt is a 2D tensor
            
            error = 0
            for (trs , orgi) in zip(decoded_bits_np, trnsf_decoded):
                for (i, j) in zip(trs , orgi):
                    if i != j:
                        error = error + 1
            
            
           
            print("Mean Difference beween SC and Transformer: ", error / total_bits_per_bit)

            


            error = 0
            for (trs , orgi) in zip(decoded_bits_np, original_msg):
               
                
                for (i, j) in zip(trs , orgi):
                    if i != j:
                        error = error + 1
            print("SC decoder bit error rate: ", error / total_bits_per_bit)



            # bitwise_errors = np.sum(trnsf_decoded != original_msg, axis=0)
            # total_bits_per_bit = trnsf_decoded.size  # Assuming gt is a 2D tensor
            # # # Calculate Bitwise Error Rate (BER) for each bit position
            # ber_per_bit = bitwise_errors / total_bits_per_bit
            error = 0
            for (trs , orgi) in zip(trnsf_decoded, original_msg):
               
                
                for (i, j) in zip(trs , orgi):
                    if i != j:
                        error = error + 1
            print("Transformer decoder bit error rate: ", error / (256 * 8))



def find_binary_expr():
    m = 5
    #iters = [1000 * (i+1) for i in range(10)]
    #iters = [70000, 90000, 100000, 110000, 120000, 150000, 170000, 200000, 210000, 220000, 230000, 250000, 260000, 270000, 280000, 290000, 300000]
    iters = [20000, 30000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 105000, 110000, 115000, 120000]
    #iters = [1000 * (i+1) for i in range(5)]
    best_error = [[0] * K for _ in iters]


    path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}/{6}/'\
                                        .format(K, N, rate_profile,  model, n_head,n_layers, run)
    figure_path=path + "figures/noisyLinFunc/"
    os.makedirs(figure_path, exist_ok=True)  # Create folder if it doesn't exist

    


    weights_arr = [[[0 for _ in range(m)] for _ in range(K)] for _ in range(len(iters))]

    # for iter_value, top_weights_for_iter in zip(iters, top_weights):
    #     top_weights_dict[iter_value] = {idx: weights for idx, weights in enumerate(top_weights_for_iter)}

    for (i , iter) in enumerate(iters):
        print("=========== Iteration " + str(iter) + "===========")
        for mode in ["combine"]:
            print( mode + " data")
            for output_idx in range(K):
                print("====== Bit " + str(output_idx) + "======")

                table_train, table_test = prepare_data_binary(results_save_path, iter, output_idx)
                table = ""
                if mode == "train":
                    table = table_train
                elif mode == "test":
                    table = table_test
                else:
                    table = table_train + table_test


                top_weight = find_best_weights(table , m)
                print(top_weight)
                weights_arr[i][output_idx][:] = [iter , output_idx] + top_weight
                best_error[i][output_idx] = top_weight[0][1]
              
                




    best_error = np.mean(np.array(best_error), axis=1)
    #print(best_error)
    plt.figure()
    plt.plot(iters, best_error)
    plt.xlabel('iteration')
    plt.ylabel('Err: Transformer - best Lin func')
    # Adding legend
    plt.title("L" + str(args.n_layers) + "-H" + str(args.n_head) + "-Run" + str(args.run))
    os.makedirs(figure_path, exist_ok=True)  # Create folder if it doesn't exist
    plt.savefig(figure_path  + "meanTrtoLinErr.png")
    plt.close()

    np.savez(results_save_path + 'meanTrtoLinErr.npz', iters=np.array(iters), best_error=best_error)

    #print(weights_arr)            
    json_data = json.dumps(weights_arr, default=lambda x: list(x) if isinstance(x, tuple) else x)

    # Save to a file
    with open(results_save_path + 'top_weights.json', 'w') as json_file:
        json_file.write(json_data)

    print("Data saved")








parser = argparse.ArgumentParser(description='Your script description here')


#  attention tensor list = [n_layer * (batchsize * heads * N * N)] each () is a tensor
# Define mandatory arguments
parser.add_argument('N', type=int, help='Value for N')
parser.add_argument('K', type=int, help='Value for K')
parser.add_argument('run', type=int, help='Value for run')
#parser.add_argument('model_iters', type=int, help='model iteration')
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
#model_iters = args.model_iters
model = "encoder"
rate_profile = "polar"
n_head = args.n_head
n_layers = args.n_layers
mode = "combine"
results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}/{6}/code_tables/noisy/'\
                                        .format(K, N, rate_profile,  model, n_head,n_layers, run)



find_binary_expr()
#compare_with_sc()



   