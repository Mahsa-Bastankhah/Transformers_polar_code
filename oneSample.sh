#!/bin/bash
#SBATCH --job-name=<name>         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=mig
#SBATCH --mail-type=fail        # send email when job fails
#SBATCH --mail-user=mb6458@princeton.edu

# module purge
# module load anaconda3/2023.3
# source activate mlenv

run=1
n_head=1
n_layers=1
num_steps=100
print_freq=10
N=4
K=4
snr=5.5

mkdir ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}
touch ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt

#python -u run_models.py --val_one_sample --model encoder --N ${N} --K ${K} --max_len ${N} --dec_train_snr ${snr} --validation_snr ${snr} --n_head ${n_head} --n_layers ${n_layers} --num_steps ${num_steps} --batch_size 8192  --print_freq ${print_freq} --code polar --rate_profile polar  --target_K ${K} --run ${run} > ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
python -u run_models.py --val_one_sample --model encoder --N ${N} --K ${K} --max_len ${N} --dec_train_snr ${snr} --validation_snr ${snr} --n_head ${n_head} --n_layers ${n_layers} --num_steps ${num_steps} --batch_size 8192  --print_freq ${print_freq} --code polar --rate_profile polar  --target_K ${K} --run ${run} 

# (16,8), (8,4) >> SNR = 1.2, 15000 steps needed
# (8,8) , (4,4) >> SNR = 5.5, 5000 steps is enough