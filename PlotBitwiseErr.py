import csv
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Your script description here')


#  attention tensor list = [n_layer * (batchsize * heads * N * N)] each () is a tensor
# Define mandatory arguments
parser.add_argument('N', type=int, help='Value for N')
parser.add_argument('K', type=int, help='Value for K')
parser.add_argument('snr', type=float, help='Value for snr')
parser.add_argument('run', type=int, help='Value for run')
parser.add_argument('print_freq', type=int, help='Value for print_freq')
parser.add_argument('--oe', action="store_true", help='one example')




# Define optional arguments
parser.add_argument('--all', action="store_true", help='do you want to plot bitwise error rate for all bits?')
parser.add_argument('--n_head', type=int, default=1,help='Value for n_head')
parser.add_argument('--n_layers', type=int, default=6, help='Value for n_layers')
parser.add_argument('--model', type=str, default='encoder', help='Value for model')
parser.add_argument('--rate_profile', type=str, default='polar', help='Value for rate_profile')
parser.add_argument('--rng', type=int,default=20, help='number of steps maximum')

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument values
N = args.N
K = args.K
snr = args.snr
run = args.run

n_head = args.n_head
n_layers = args.n_layers
model = args.model
rate_profile = args.rate_profile
rng = args.rng

model = "encoder"
rate_profile = "polar"




results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                    .format(K, N, rate_profile,  model, n_head,n_layers)

results_save_path = results_save_path + '/' + '{0}'.format(run)
# Load data from the CSV file
with open(results_save_path +'/values_validation.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

    # Define the number of steps
    num_steps = N

    # Create a color gradient from red to blue
    colors = plt.cm.RdBu(np.linspace(0, 1, K))

    valid_steps = list(map(int, data[0]))
    bitwise_errors_noisy = [list(map(float, row)) for row in data[3:3+K]]  
    bitwise_errors_no_noise = [list(map(float, row)) for row in data[3+K:3+2*K]] 
    filename_noisy= '/bitwise_noisy_rng' + str(rng) + '.pdf'
    filename_noiseless= '/bitwise_noisless_rng' + str(rng) + '.pdf'
    if args.all:
        start = 3+2*K
        bitwise_errors_noisy = [list(map(float, row)) for row in data[start:start+N]]  
        bitwise_errors_no_noise = [list(map(float, row)) for row in data[start+N:start+2*N]] 
        filename_noisy= '/bitwise_error_plot_noisy_all.pdf'
        filename_noiseless= '/bitwise_error_plot_noisless_all.pdf'
        colors = plt.cm.RdBu(np.linspace(0, 1, N))

    plt.figure()
    # Plot the bitwise errors
    for i, errors in enumerate(bitwise_errors_noisy):
        plt.plot(valid_steps[0:rng], errors[0:rng], label=f'Bit {i}', color=colors[i])

    plt.xlabel('Number of Steps')
    plt.ylabel('Bitwise Error')
    plt.title('Validation data, Noisy,snr' + str(snr) + 'L' + str(n_layers) + 'H' + str(n_head), fontname='sans-serif')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(results_save_path + "/plots" + filename_noisy)
    plt.show()

    plt.figure()
    for i, errors in enumerate(bitwise_errors_no_noise):
        plt.plot(valid_steps[0:rng], errors[0:rng], label=f'Bit {i}', color=colors[i])

    plt.xlabel('Number of Steps')
    plt.ylabel('Bitwise Error')
    plt.title('Train data, Noiseless' + 'L' + str(n_layers) + 'H' + str(n_head), fontname='sans-serif')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(results_save_path + "/plots" + filename_noiseless)
    plt.show()


# converge_step_no_noise = []
# for errors in bitwise_errors_no_noise:
#     try:
#         index = next(i for i, value in enumerate(errors) if abs(value-errors[i+1])<= 0.01 and value <= min(errors) + 0.01 )
#         converge_step_no_noise.append(args.print_freq * index)
#         #print(f"The first index with a value of zero is {index}")
#     except StopIteration:
#         print("No zero values found in the list")
# print("the convergence index for noiseless validation" , converge_step_no_noise)


# converge_step = []
# for errors in bitwise_errors_noisy:
#     try:
#         index = next(i for i, value in enumerate(errors) if abs(value-errors[i+1])<= 0.01 and value <= min(errors) + 0.01 )
#         converge_step.append(args.print_freq * index)
#         #print(f"The first index with a value of zero is {index}")
#     except StopIteration:
#         print("No zero values found in the list")
# print("the convergence index for noisy validation    " , converge_step)


########################### Test data, noisy ###########################
# Load data from the CSV file
with open(results_save_path +'/values_test.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

    # Define the number of steps
    num_steps = N

    # Create a color gradient from red to blue
    #colors = plt.cm.RdBu(np.linspace(0, 1, K))

    test_ber = list(map(float, data[1]))
    # start = 2+K
    # test_bitwise_errors_noisy = [list(map(float, row)) for row in data[start:start+N]]  
    start = 3
    test_bitwise_errors_noisy = [list(map(float, row)) for row in data[start:start+K]]  
    filename_noisy = '/test_bitwise_noisy_rng' + str(rng) +'.pdf'
    if args.all:
        start = 3+2*K
        test_bitwise_errors_noisy  = [list(map(float, row)) for row in data[start:start+N]]  
        filename_noisy = '/test_bitwise_error_plot_noisy_all.pdf'
        colors = plt.cm.RdBu(np.linspace(0, 1, N))

    plt.figure()
    

    plt.plot(valid_steps[0:rng], test_ber[0:rng])

    plt.xlabel('Number of Steps')
    plt.ylabel('Bit Error Rate')
    plt.title('Test data, Noisy,snr' + str(snr) + 'L' + str(n_layers) + 'H' + str(n_head), fontname='sans-serif')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(results_save_path + '/plots/test_ber_plot_noisy.pdf')
    plt.show()



    plt.figure()
    # Plot the bitwise errors
    for i, errors in enumerate(test_bitwise_errors_noisy):
        plt.plot(valid_steps[0:rng], errors[0:rng], label=f'Bit {i}', color=colors[i])

    plt.xlabel('Number of Steps')
    plt.ylabel('Bitwise Error')
    plt.title('Test data, Noisy,snr' + str(snr) + 'L' + str(n_layers) + 'H' + str(n_head), fontname='sans-serif')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(results_save_path + "/plots"+ filename_noisy)
    plt.show()

    ########################### Test data, noiseless ###########################
    test_ber_noiseless = list(map(float, data[2]))
    # start = 2+K
    # test_bitwise_errors_noisy = [list(map(float, row)) for row in data[start:start+N]]  

    if args.all== False:
        start = start+K
        test_bitwise_errors_noiseless = [list(map(float, row)) for row in data[start:start+K]]  
        filename_noiseless = '/test_bitwise_no_noise_rng' + str(rng) + '.pdf'
    elif args.all:
        start = start+N
        test_bitwise_errors_noiseless  = [list(map(float, row)) for row in data[start:start+N]]  
        filename_noiseless = '/test_bitwise_error_plot_no_noise_all.pdf'
        colors = plt.cm.RdBu(np.linspace(0, 1, N))
    plt.figure()
    # Plot the bitwise errors

    plt.plot(valid_steps[0:rng], test_ber_noiseless[0:rng])

    plt.xlabel('Number of Steps')
    plt.ylabel('Bit Error Rate')
    plt.title('Test data, Noiseless  L ' + str(n_layers) + 'H' + str(n_head), fontname='sans-serif')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(results_save_path + '/plots/test_ber_plot_no_noise.pdf')
    plt.show()



    plt.figure()
    # Plot the bitwise errors
    for i, errors in enumerate(test_bitwise_errors_noiseless):
        plt.plot(valid_steps[0:rng], errors[0:rng], label=f'Bit {i}', color=colors[i])

    plt.xlabel('Number of Steps')
    plt.ylabel('Bitwise Error')
    plt.title('Test data, Noiseless , L' + str(n_layers) + 'H' + str(n_head), fontname='sans-serif')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(results_save_path + "/plots" +  filename_noiseless)
    plt.show()





################################### ONE EXAMPLE ###################################

    if args.oe:

        with open(results_save_path +'/values_validation_oe.csv', 'r') as f:
            reader = csv.reader(f)
            data = list(reader)

        # Define the number of steps


        ber = list(map(float, data[1]))
        # start = 2+K
        # test_bitwise_errors_noisy = [list(map(float, row)) for row in data[start:start+N]]  

        

        plt.figure()
        plt.plot(valid_steps[0:rng], ber[0:rng])
        plt.xlabel('Number of Steps')
        plt.ylabel('Bit Error Rate')
        plt.title('one example valid data, Noisy,snr' + str(snr) + 'L' + str(n_layers) + 'H' + str(n_head), fontname='sans-serif')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(results_save_path + '/plots/ber_plot_noisy_oe.pdf')
        plt.show()




        # Create a color gradient from red to blue
        colors = plt.cm.RdBu(np.linspace(0, 1, K))

        valid_steps = list(map(int, data[0]))
        bitwise_errors_noisy = [list(map(float, row)) for row in data[3:3+K]]  
        bitwise_errors_no_noise = [list(map(float, row)) for row in data[3+K:3+2*K]] 
        filename_noisy= '/bitwise_noisy_oe_rng' + str(rng) + '.pdf'
        filename_noiseless= '/bitwise_noisless_oe_rng' + str(rng) + '.pdf'
        
        plt.figure()
        # Plot the bitwise errors
        for i, errors in enumerate(bitwise_errors_noisy):
            plt.plot(valid_steps[0:rng], errors[0:rng], label=f'Bit {i}', color=colors[i])

        plt.xlabel('Number of Steps')
        plt.ylabel('Bitwise Error')
        plt.title('one example valid data, Noisy,snr' + str(snr) + 'L' + str(n_layers) + 'H' + str(n_head), fontname='sans-serif')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(results_save_path + "/plots/" + filename_noisy)
        plt.show()

        plt.figure()
        for i, errors in enumerate(bitwise_errors_no_noise):
            plt.plot(valid_steps[0:rng], errors[0:rng], label=f'Bit {i}', color=colors[i])

        plt.xlabel('Number of Steps')
        plt.ylabel('Bitwise Error')
        plt.title('one example valid data, Noiseless' + 'L' + str(n_layers) + 'H' + str(n_head), fontname='sans-serif')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(results_save_path + "/plots/" + filename_noiseless)
        plt.show()







