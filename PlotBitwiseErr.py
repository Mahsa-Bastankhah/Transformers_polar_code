import csv
import matplotlib.pyplot as plt
import argparse
import numpy as np
rng = 20
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

n_head = args.n_head
n_layers = args.n_layers
model = args.model
rate_profile = args.rate_profile

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
colors = plt.cm.RdBu(np.linspace(0, 1, num_steps))

valid_steps = list(map(int, data[0]))
bitwise_errors_noisy = [list(map(float, row)) for row in data[3:3+N]]  
bitwise_errors_no_noise = [list(map(float, row)) for row in data[3+N:3+2*N]] 

plt.figure()
# Plot the bitwise errors
for i, errors in enumerate(bitwise_errors_noisy):
    plt.plot(valid_steps[0:rng], errors[0:rng], label=f'Bit {i}', color=colors[i])

plt.xlabel('Number of Steps')
plt.ylabel('Bitwise Error')
plt.title('Noisy,snr' + str(snr) + 'L' + str(n_layers) + 'H' + str(n_head), fontname='sans-serif')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(results_save_path + '/bitwise_error_plot_noisy.pdf')
plt.show()

plt.figure()
for i, errors in enumerate(bitwise_errors_no_noise):
    plt.plot(valid_steps[0:rng], errors[0:rng], label=f'Bit {i}', color=colors[i])

plt.xlabel('Number of Steps')
plt.ylabel('Bitwise Error')
plt.title('Noiseless' + 'L' + str(n_layers) + 'H' + str(n_head), fontname='sans-serif')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(results_save_path + '/bitwise_error_plot_noisless.pdf')
plt.show()


converge_step_no_noise = []
for errors in bitwise_errors_no_noise:
    try:
        index = next(i for i, value in enumerate(errors) if abs(value-errors[i+1])<= 0.01 and value <= min(errors) + 0.01 )
        converge_step_no_noise.append(args.print_freq * index)
        #print(f"The first index with a value of zero is {index}")
    except StopIteration:
        print("No zero values found in the list")
print("the convergence index for noiseless validation" , converge_step_no_noise)


converge_step = []
for errors in bitwise_errors_noisy:
    try:
        index = next(i for i, value in enumerate(errors) if abs(value-errors[i+1])<= 0.01 and value <= min(errors) + 0.01 )
        converge_step.append(args.print_freq * index)
        #print(f"The first index with a value of zero is {index}")
    except StopIteration:
        print("No zero values found in the list")
print("the convergence index for noisy validation    " , converge_step)





