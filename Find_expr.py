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



def simplify_boolean_function(model_iters , output_idx):
    # Load the training results DataFrame
    results_path = results_save_path + '16bits_noiseless' + str(model_iters) + '.csv'  # Update with the actual path
   
    df = pd.read_csv(results_path)


    # Convert string representations of lists to actual lists
    #df_train['Original_Message'] = df_train['Original_Message'].apply(ast.literal_eval)
    df['Decoded_Bits'] = df['Decoded_Bits'].apply(ast.literal_eval)
    df['Polar_code'] = df['Polar_code'].apply(ast.literal_eval)

    df = df.sample(n=256, random_state=20)  # Set random_state for reproducibility

    # df_test['Original_Message'] = df_test['Original_Message'].apply(ast.literal_eval)


   # Extracting the labels and input bits for training data
    truth_table = df.apply(lambda row: tuple(row['Polar_code'] + [row['Decoded_Bits'][output_idx]]), axis=1).tolist()
    # Print the dimension
   
   
   
   
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


def prepare_data_binary(results_save_path, model_iters, output_idx):
    # Load the training results DataFrame
    #train_results_path = results_save_path + '16bits_noiseless' + str(model_iters) + '.csv'  # Update with the actual path
    train_results_path = results_save_path + 'train_' + str(model_iters) + '.csv'  # Update with the actual path
    df_train = pd.read_csv(train_results_path)

    # # Load the test results DataFrame
    test_results_path = results_save_path + 'test_' + str(model_iters) + '.csv'
    df_test = pd.read_csv(test_results_path)

    # Convert string representations of lists to actual lists
    #df_train['Original_Message'] = df_train['Original_Message'].apply(ast.literal_eval)
    df_train['Decoded_Bits'] = df_train['Decoded_Bits'].apply(ast.literal_eval)
    df_train['Binary_Corrupted_codeword'] = df_train['Binary_Corrupted_codeword'].apply(ast.literal_eval)

    # df_test['Original_Message'] = df_test['Original_Message'].apply(ast.literal_eval)
    df_test['Decoded_Bits'] = df_test['Decoded_Bits'].apply(ast.literal_eval)
    df_test['Binary_Corrupted_codeword'] = df_test['Binary_Corrupted_codeword'].apply(ast.literal_eval)


   # Extracting the labels and input bits for training data
    train_table = df_train.apply(lambda row: tuple(row['Binary_Corrupted_codeword'] + [row['Decoded_Bits'][output_idx]]), axis=1).tolist()
    # Print the dimensions
    

    
    # # Extracting the labels and input bits for test data
    test_table = df_test.apply(lambda row: tuple(row['Binary_Corrupted_codeword'] + [row['Decoded_Bits'][output_idx]]), axis=1).tolist()
    # # for i in range(len(test_table)):

    #     print(test_table[i][15], test_table[i][16] )
    # # Combine train and test truth tables
    return  train_table, test_table


def prepare_data_binary_16(results_save_path, model_iters, output_idx):
    # Load the training results DataFrame
    results_path = results_save_path + '16bits_noiseless' + str(model_iters) + '.csv'  # Update with the actual path
   
    df = pd.read_csv(results_path)


    # Convert string representations of lists to actual lists
    #df_train['Original_Message'] = df_train['Original_Message'].apply(ast.literal_eval)
    df['Decoded_Bits'] = df['Decoded_Bits'].apply(ast.literal_eval)
    df['Polar_code'] = df['Polar_code'].apply(ast.literal_eval)

    df = df.sample(n=256, random_state=20)  # Set random_state for reproducibility

    # df_test['Original_Message'] = df_test['Original_Message'].apply(ast.literal_eval)


   # Extracting the labels and input bits for training data
    table = df.apply(lambda row: tuple(row['Polar_code'] + [row['Decoded_Bits'][output_idx]]), axis=1).tolist()
    # Print the dimensions
    

    return  table

def test(results_save_path, model_iters):
    # Load the training results DataFrame
    results_path = results_save_path + '16bits_noiseless' + str(model_iters) + '.csv'  # Update with the actual path
   
    df = pd.read_csv(results_path)


    # Convert string representations of lists to actual lists
    #df_train['Original_Message'] = df_train['Original_Message'].apply(ast.literal_eval)
    df['Decoded_Bits'] = df['Decoded_Bits'].apply(ast.literal_eval)
    df['Polar_code'] = df['Polar_code'].apply(ast.literal_eval)

    train_results_path = results_save_path + 'noisy/train_' + str(model_iters) + '.csv'  # Update with the actual path
    df_train = pd.read_csv(train_results_path)

    # # Load the test results DataFrame
    test_results_path = results_save_path + 'noisy/test_' + str(model_iters) + '.csv'
    df_test = pd.read_csv(test_results_path)

    # Convert string representations of lists to actual lists
    #df_train['Original_Message'] = df_train['Original_Message'].apply(ast.literal_eval)
    df_train['Decoded_Bits'] = df_train['Decoded_Bits'].apply(ast.literal_eval)
    df_train['Binary_Corrupted_codeword'] = df_train['Binary_Corrupted_codeword'].apply(ast.literal_eval)

    # df_test['Original_Message'] = df_test['Original_Message'].apply(ast.literal_eval)
    df_test['Decoded_Bits'] = df_test['Decoded_Bits'].apply(ast.literal_eval)
    df_test['Binary_Corrupted_codeword'] = df_test['Binary_Corrupted_codeword'].apply(ast.literal_eval)
    combined_df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # Iterate through rows and compare the corresponding columns
    different_decoded_bits_count = 0
    total_rows = 0
    diff = 0
    for index, row in df.iterrows():
        polar_code = row['Polar_code']
        decoded_bits_df = combined_df[combined_df['Binary_Corrupted_codeword'].isin([polar_code])]['Decoded_Bits']
        total_rows = total_rows + len(decoded_bits_df)
        if len(decoded_bits_df) > 0:
            diff = diff + 1
            
        if not decoded_bits_df.empty:
            for decoded in decoded_bits_df.values.flatten():
                if decoded != row['Decoded_Bits']:
                    #print(decoded)
                    different_decoded_bits_count += 1
                
        #print("========= =========")

    print(f"Number of rows with the same Polar_code and different Decoded_Bits: {different_decoded_bits_count/total_rows}")

    print(total_rows)
    print(diff)


def prepare_data_anal(results_save_path, model_iters):
    # Load the training results DataFrame
    train_results_path = results_save_path + 'Ltrain_' + str(model_iters) + '.csv'  # Update with the actual path
    df_train = pd.read_csv(train_results_path)

    # Load the test results DataFrame
    test_results_path = results_save_path + 'Ltest_' + str(model_iters) + '.csv'
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

    # print(new_df_test['Original_Message'])
    # print(new_df_test['Decoded_Bits'])
   
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
                if error <= 0.005:
                    return top_weights
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



#
# sixteenBits should be false for running thios function because here we use the actual noisy data
def compare_with_sc():
    n = int(np.log2(args.N))
    info_inds = [7,9,10,11,12,13,14,15]
    rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1
    # Multiple SNRs:
    print("<<<<<<<<<<<<<<<<<<<<<<< comparing the trnsfrm BER with SC >>>>>>>>>>>>>>>>>>>>>>>>")
    
    polar = PolarCode(n, args.K, args, rs=rs)
    # Convert the train_table to a DataFrame
    # Extracting corrupted codewords and labels for test data
    iters = [20000 * (i+1) for i in range(8)]
    #iters = [10000 * (i+1) for i in range(20)]
    #iters = [10000]
    print("==============Train and Test Data Combined  ============== ")
    for (i , iter) in enumerate(iters):
        print("============== Iteration " + str(iter) + "==============")
        

        df_train, df_test = prepare_data_anal(results_save_path, iter)
        df = pd.concat([df_test, df_train], ignore_index=True)
        #df = df_train
        


        codewords = np.array([np.array(item) for item in df['Corrupted_codeword']])
        trnsf_decoded = np.array([np.array(item) for item in df['Decoded_Bits']])
        original_msg = np.array([np.array(item) for item in df['Original_Message']])

        print(len(trnsf_decoded))
        # print(original_msg)

        # Convert the NumPy array to a PyTorch tensor and move it to the CPU
        corrupted_codewords_tensor = torch.tensor(codewords, dtype=torch.float32, device='cpu')

        SC_llrs, decoded_SC_msg_bits = polar.sc_decode_new(corrupted_codewords_tensor , args.snr )
        decoded_bits_np = decoded_SC_msg_bits.numpy()
        

        decoded_bits_np[decoded_bits_np == 1] = 0
        decoded_bits_np[decoded_bits_np == -1] = 1

    

        

        
        print("SC decoder error")
        bitwise_error_percentages = [0] * 8  # Initialize list to store error percentages for each bit position

        for (trs, orgi) in zip(decoded_bits_np, original_msg):
            for bit_position in range(K):
                if trs[bit_position] != orgi[bit_position]:
                    bitwise_error_percentages[bit_position] = bitwise_error_percentages[bit_position] + 1

        # Calculate the average error percentage for each bit position
        average_error_percentages = [percentage / len(decoded_bits_np) for percentage in bitwise_error_percentages]

        # Print or use the average_error_percentages as needed
        for bit_position, error_percentage in enumerate(average_error_percentages):
            print(f"Bit Position {bit_position}: {round(error_percentage,4)}")


        print("==== Transformers Error ====")
        bitwise_error_percentages = [0] * 8  # Initialize list to store error percentages for each bit position

        for (trs, orgi) in zip(trnsf_decoded, original_msg):
            for bit_position in range(K):
                if trs[bit_position] != orgi[bit_position]:
                    bitwise_error_percentages[bit_position] = bitwise_error_percentages[bit_position] + 1

        # Calculate the average error percentage for each bit position
        average_error_percentages = [percentage / len(trnsf_decoded) for percentage in bitwise_error_percentages]

        # Print or use the average_error_percentages as needed
        for bit_position, error_percentage in enumerate(average_error_percentages):
            print(f"Bit Position {bit_position}: {round(error_percentage,4)}")



def find_binary_expr():
    m = 3
    
    #iters = [100 * (i+1) for i in range(20)]
    #iters = iters + [1000 * (i+3) for i in range(8)]
    iters = [10000 * (i+1) for i in range(2,20)]

    best_error = [[0] * K for _ in iters]

    weights_arr = [[[0 for _ in range(m)] for _ in range(K)] for _ in range(len(iters))]

    if sixteenBits:
        print("<<<<<<<<<<<<<<<<<<<<<<< direct binary input - output >>>>>>>>>>>>>>>>>>>>>>>>")
    else:
        print("<<<<<<<<<<<<<<<<<<<<<<< find the function from binarized corrupted codeword >>>>>>>>>>>>>>>>>>>>>>>>")
    for (i , iter) in enumerate(iters):
        print("=========== Iteration " + str(iter) + "===========")
        
        for output_idx in range(K):
            print("====== Bit " + str(output_idx) + "======")

            #
            if sixteenBits:
                table = prepare_data_binary_16(results_save_path, iter, output_idx)
            else:
                table_train , table_test = prepare_data_binary(results_save_path, iter, output_idx)
                print("====only training data =====")
                table = table_train
            top_weight = find_best_weights(table , m)
            print(top_weight)
            weights_arr[i][output_idx][:] = [iter , output_idx] + top_weight
            best_error[i][output_idx] = top_weight[0][1]
              


    best_error = np.mean(np.array(best_error), axis=1)
    #print(best_error)
   

    if sixteenBits:

        try:
            loaded_data = np.load(results_save_path + 'meanTrtoLinErr16bitNoiseless.npz')
            loaded_iters = loaded_data['iters']
            loaded_best_error = loaded_data['best_error']
        except FileNotFoundError:
            # Initialize with empty arrays if the file is not found
            loaded_iters = np.array([])
            loaded_best_error = np.array([])

    else:
        loaded_data = np.load(results_save_path + 'meanTrtoLinErr.npz')
        loaded_iters = loaded_data['iters']
        loaded_best_error = loaded_data['best_error']
    

    # Access individual arrays

    combined_iters = np.concatenate([iters, loaded_iters])
    combined_best_error = np.concatenate([best_error, loaded_best_error])
    # Get unique pairs while maintaining the order
    unique_indices = np.unique(combined_iters, return_index=True)[1]
    unique_iters = combined_iters[unique_indices]
    unique_best_error = combined_best_error[unique_indices]
    #print(unique_iters, unique_best_error)

    plt.figure()
    plt.plot(unique_iters, unique_best_error)
    plt.xlabel('iteration')
    plt.ylabel('Err: Transformer - best Lin func')
    # Adding legend
    plt.title("L" + str(args.n_layers) + "-H" + str(args.n_head) + "-Run" + str(args.run))
    os.makedirs(figure_path, exist_ok=True)  # Create folder if it doesn't exist
    if sixteenBits:
        plt.savefig(figure_path  + "meanTrtoLinErr16bitNoiseless.png")
    else:
        plt.savefig(figure_path  + "meanTrtoLinErr.png")

    plt.close()

    
    if sixteenBits:
        np.savez(results_save_path + 'meanTrtoLinErr16bitNoiseless.npz', iters=np.array(unique_iters), best_error=unique_best_error)
    else:
        np.savez(results_save_path + 'meanTrtoLinErr.npz', iters=np.array(unique_iters), best_error=unique_best_error)
    
    #print(weights_arr)            
    json_data = json.dumps(weights_arr, default=lambda x: list(x) if isinstance(x, tuple) else x)

    # Save to a file
    

    if sixteenBits:
        with open(results_save_path + 'top_weights16bitNoiseless.json', 'w') as json_file:
            json_file.write(json_data)
    else:
        with open(results_save_path + 'top_weights.json', 'w') as json_file:
            json_file.write(json_data)
    

    print("Data saved")



def plotBitwiseLinErr():
    if sixteenBits:
        filename = 'top_weights16bitNoiseless.json'
    else:
        filename = 'top_weights.json'
    with open(results_save_path + filename, 'r') as json_file:
        json_data = json.load(json_file)
        BitwiseErrArr = np.zeros((len(json_data), 8))
        itersArr = np.zeros((len(json_data)))

        for iter,iterData in enumerate(json_data):
            for bit,bitData in enumerate(iterData):
                BitwiseErrArr[iter][bit] = bitData[2][1]
                itersArr[iter] = bitData[0]
        
        colors = plt.cm.RdBu(np.linspace(0, 1, K))
        # Plot each bit separately
        for bit in range(BitwiseErrArr.shape[1]):
            plt.plot(itersArr[:], BitwiseErrArr[:, bit], label=f'Bit {bit}', color = colors[bit])

        plt.xlabel('Iteration')
        plt.ylabel('Bitwise Err')
        plt.title('Error Between the Trnsfrmr and the Lin Approx')
        plt.legend()
        plt.savefig(figure_path + 'bitwise_error16Bit.png')






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
sixteenBits = False
if sixteenBits:
    results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}/{6}/code_tables/'\
                                        .format(K, N, rate_profile,  model, n_head,n_layers, run)
else:
    results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}/{6}/code_tables/noisy/'\
                                        .format(K, N, rate_profile,  model, n_head,n_layers, run)
path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}/{6}/'\
                                    .format(K, N, rate_profile,  model, n_head,n_layers, run)

if sixteenBits:
    figure_path=path + "figures/sixteenBitsLinFunc/"
else:
    figure_path=path + "figures/noisyLinFunc/"
os.makedirs(figure_path, exist_ok=True)  # Create folder if it doesn't exist



#print(simplify_boolean_function(120000, 6))
#plotBitwiseLinErr()
find_binary_expr()
#compare_with_sc()
# iters = [100,200,300,400,500,600,700,800,900]
# iters = [10000 * (i +1) for i in range(1,12)]
# #iters = [4000, 5000, 6000, 7000]
# for i in iters:
#     print("iter :", i)
#     test(results_save_path,i)
#compare_with_sc()


   