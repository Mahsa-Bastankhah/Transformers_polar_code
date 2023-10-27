#!/bin/bash
#SBATCH --job-name=<name>         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=mig

module purge
module load anaconda3/2023.3
source activate mlenv

run=6
n_head=1
n_layers=6

mkdir ./Supervised_Xformer_decoder_Polar_Results/Polar_8_8/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}
touch ./Supervised_Xformer_decoder_Polar_Results/Polar_8_8/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
mkdir ./Supervised_Xformer_decoder_Polar_Results/Polar_4_4/Scheme_polar/encoder/1_depth_${n_layers}/${run}
touch ./Supervised_Xformer_decoder_Polar_Results/Polar_4_4/Scheme_polar/encoder/1_depth_${n_layers}/${run}/output.txt
 
python -u run_models.py --model encoder --N 8 --max_len 8 --K 8 --dec_train_snr 5.5 --validation_snr 5.5 --n_head ${n_head} --n_layers ${n_layers} --embed_dim 32 --num_steps 2000 --batch_size 8192  --print_freq 50 --code polar --rate_profile polar  --target_K 8 --run ${run} > ./Supervised_Xformer_decoder_Polar_Results/Polar_8_8/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
python -u run_models.py --model encoder --N 4 --max_len 4 --K 4 --dec_train_snr 5.5 --validation_snr 5.5 --n_head ${n_head} --n_layers ${n_layers} --embed_dim 32 --num_steps 2000 --batch_size 8192  --print_freq 50  --code polar --rate_profile polar  --target_K 4 --run ${run} > ./Supervised_Xformer_decoder_Polar_Results/Polar_4_4/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
