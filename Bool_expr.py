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


def generate_random_truth_table(num_bits):
    # Generate all possible 3-bit binary sequences
    binary_sequences = list(product([0, 1], repeat=num_bits))

    # Assign random labels (0 or 1) to each sequence
    truth_table = [(seq + (seq[0]*seq[1]*seq[10] or seq[5] or Xor(seq[12], seq[13], seq[14]) ,)) for seq in binary_sequences]

    return truth_table

def simplify_boolean_function(truth_table):
    num_inputs = len(truth_table[0]) - 1

    # Use the first num_inputs variables as input variables
    input_variables = [exprvar(f'x{i}') for i in range(num_inputs)]

   # Identify rows where the output is 1
    positive_rows = [row[:-1] for row in truth_table if row[-1]]

    # Create a conjunction for each positive row
    conjunctions = [And(*[var if bit else ~var for var, bit in zip(input_variables, positive_row)]) for positive_row in positive_rows]

    # Create a disjunction of the conjunctions
    boolean_expression = Or(*conjunctions)
    #print(boolean_expression)

    # Use Espresso to perform minimization
    simplified_boolean_function, = espresso_exprs(boolean_expression.to_dnf())

    return simplified_boolean_function

def prepare_data(results_save_path, model_iters, output_idx):
    # Load the training results DataFrame
    train_results_path = results_save_path + 'train_' + str(model_iters) + '.csv'  # Update with the actual path
    df_train = pd.read_csv(train_results_path)

    # Load the test results DataFrame
    test_results_path = results_save_path + 'test_' + str(model_iters) + '.csv'
    df_test = pd.read_csv(test_results_path)

    # Convert string representations of lists to actual lists
    df_train['Original_Message'] = df_train['Original_Message'].apply(ast.literal_eval)
    df_train['Decoded_Bits'] = df_train['Decoded_Bits'].apply(ast.literal_eval)
    df_train['Polar_code'] = df_train['Polar_code'].apply(ast.literal_eval)

    df_test['Original_Message'] = df_test['Original_Message'].apply(ast.literal_eval)
    df_test['Decoded_Bits'] = df_test['Decoded_Bits'].apply(ast.literal_eval)
    df_test['Polar_code'] = df_test['Polar_code'].apply(ast.literal_eval)

    #print(df_train)


    # Calculate bitwise error rate for each bit position in test data
    # test_original_bits = np.array(df_test['Original_Message'].tolist())
    # test_decoded_bits = np.array(df_test['Decoded_Bits'].tolist())
    # bitwise_errors_test = np.sum(test_original_bits != test_decoded_bits, axis=0)
    # bitwise_error_rate_test = bitwise_errors_test / len(df_test)

    # # Calculate overall bit error rate
    
    

    # print("\nBitwise Error Rates for Test Data:")
    # print(bitwise_error_rate_test)



    #print(df_train['Decoded_Bits'])
    # Extracting the labels and input bits for training data
   # Extracting the labels and input bits for training data
    train_truth_table = df_train.apply(lambda row: tuple(row['Polar_code'] + [row['Decoded_Bits'][output_idx]]), axis=1).tolist()
    #print(train_truth_table)
    # Extracting the labels and input bits for test data
    test_truth_table = df_test.apply(lambda row: tuple(row['Polar_code'] + [row['Decoded_Bits'][output_idx]]), axis=1).tolist()

    # # Combine train and test truth tables
    #combined_truth_table = train_truth_table + test_truth_table
    #print(combined_truth_table)
    return  train_truth_table, test_truth_table




def calculate_error(truth_table, weights):
    # Calculate the error for a given set of weights
    errors = 0
    for *inputs, label in truth_table :
        if sum(x * w for x, w in zip(inputs, weights)) % 2 != label:
            errors = errors + 1
    return errors

def find_best_weights(truth_table, m):
    num_inputs = len(truth_table[0]) - 1  # Number of input bits
    weight_combinations = product([0, 1], repeat=num_inputs)  # All possible weight combinations

    # Initialize a list to store the top m weight-error pairs
    top_weights = [(None, float('inf'))] * m

    for weights in weight_combinations:
        error = calculate_error(truth_table, weights) / len(truth_table)

        # Check if the current weights have a lower error than any in the top m
        for i, (top_weights_weights, top_error) in enumerate(top_weights):
            if error < top_error:
                top_weights.insert(i, (weights, error))
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



parser = argparse.ArgumentParser(description='Your script description here')


#  attention tensor list = [n_layer * (batchsize * heads * N * N)] each () is a tensor
# Define mandatory arguments
parser.add_argument('N', type=int, help='Value for N')
parser.add_argument('K', type=int, help='Value for K')
parser.add_argument('run', type=int, help='Value for run')
parser.add_argument('model_iters', type=int, help='model iteration')
# Define optional arguments
parser.add_argument('--n_head', type=int, default=1,help='Value for n_head')
parser.add_argument('--n_layers', type=int, default=3, help='Value for n_layers')

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument values
N = args.N
K = args.K
run = args.run
model_iters = args.model_iters

n_head = args.n_head
n_layers = args.n_layers
m = 5


model = "encoder"
rate_profile = "polar"




path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}/{6}/'\
                                    .format(K, N, rate_profile,  model, n_head,n_layers, run)
figure_path=path + "figures/Boolean/"
os.makedirs(figure_path, exist_ok=True)  # Create folder if it doesn't exist

results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}/{6}/code_tables/'\
                                    .format(K, N, rate_profile,  model, n_head,n_layers, run)

print("=========== Iteration " + str(model_iters) + "===========")
for mode in ["test"]:
    print( mode + " data")
    for output_idx in range(K):

        truth_table_train, truth_table_test = prepare_data(results_save_path, model_iters, output_idx)
        truth_table = ""
        if mode == "train":
            truth_table = truth_table_train
        elif mode == "test":
            truth_table = truth_table_test
        else:
            truth_table = truth_table_train + truth_table_test
        #print(simplify_boolean_function(truth_table))

        #print(find_best_weights(truth_table , m))
        best_weights_dict = evaluate_all_weights(truth_table)
        if output_idx == 2:
            print(best_weights_dict)

        # Extract data for plotting
        num_non_zero_elements = list(best_weights_dict.keys())
        errors = [error for _, error in best_weights_dict.values()]
        #print("num_non_zero_elements:", num_non_zero_elements)
        #print("errors:", errors)

        # Plotting
        plt.figure()
        plt.plot(num_non_zero_elements, errors, marker='o', color=plt.cm.get_cmap('tab10')(output_idx / K))

        plt.xlabel('Number of Non-Zero Elements in Weights')
        plt.ylabel('Err: Transformer - proposed function')
        plt.title("Iteration : " + str(model_iters) + "Bit : " + str(output_idx))
        this_fig_path = figure_path + "bit" + str(output_idx)
        os.makedirs(this_fig_path, exist_ok=True)  # Create folder if it doesn't exist
        plt.savefig(this_fig_path  + "/iter" + str(model_iters) + ".png")
        plt.close()
            # # Simplify Boolean function using Espresso
                #simplified_function = simplify_boolean_function(truth_table)
            #print(f"\nSimplified boolean function: {simplified_function}")
