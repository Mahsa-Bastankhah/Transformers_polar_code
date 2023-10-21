import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt


K = 8
N = 8
rate_profile = "Polar"
model = "encoder"
n_head = 8
n_layers = 6
run = 1
print_freq = 500
snr = 5

results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                    .format(K, N, rate_profile,  model, n_head, n_layers)

results_save_path = results_save_path + '/' + '{0}'.format(run)

attn_file_path_no_noise = results_save_path + '/attention_validation_no_noise.pth'
attn_file_path = results_save_path + '/attention_validation.pth'



slf_attn_list_no_noise = torch.load(attn_file_path_no_noise)
slf_attn_list = torch.load(attn_file_path)

# Now you can access the tensors from the loaded list
for i , tensor in enumerate(slf_attn_list_no_noise):
    print(i)
    plt.figure(figsize=(N, N))
    plt.imshow(tensor[0], cmap='viridis', interpolation='nearest')
    plt.title('Noiseless, Step : ' + str(i * print_freq))
    plt.colorbar()
    image_path = results_save_path + '/figures/Noiseless-Step' + str(i * print_freq) + '.png'
    plt.savefig(image_path, format='png')


# Now you can access the tensors from the loaded list
for i , tensor in enumerate(slf_attn_list):
    plt.figure(figsize=(N, N))
    plt.imshow(tensor[0], cmap='viridis', interpolation='nearest')
    plt.title('Noisy, SNR =  ' + str(snr) + ', Step : ' + str(i * print_freq))
    plt.colorbar()
    image_path = results_save_path + '/figures/Noisy-Step' + str(i * print_freq) + '.png'
    plt.savefig(image_path, format='png')


# attn_numpy_array = np.load(results_save_path +  '/attention_validation.npy')
# attn_tensor = torch.from_numpy(attn_numpy_array)



# attn_tensor_avg  = torch.mean(attn_tensor, dim=0)

# attn_tensor_avg_head1 = attn_tensor_avg[0]

# print("Type of attn:", type(attn_tensor_avg_head1))
# print("Shape of attn:", attn_tensor_avg_head1.shape)



