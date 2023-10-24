#!/bin/bash
#SBATCH --job-name=<name>         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=mb6458@princeton.edu
#SBATCH --partition=mig

module purge
module load anaconda3/2023.3
source activate mlenv

mkdir ./Supervised_Xformer_decoder_Polar_Results/Polar_8_8/Scheme_polar/encoder/8_depth_6/4
touch ./Supervised_Xformer_decoder_Polar_Results/Polar_8_8/Scheme_polar/encoder/8_depth_6/4/output.txt
mkdir ./Supervised_Xformer_decoder_Polar_Results/Polar_4_4/Scheme_polar/encoder/8_depth_6/4
touch ./Supervised_Xformer_decoder_Polar_Results/Polar_4_4/Scheme_polar/encoder/8_depth_6/4/output.txt
 
python -u run_models.py --model encoder --N 8 --max_len 8 --K 8 --dec_train_snr -2 --validation_snr -2 --embed_dim 32 --num_steps 10000 --batch_size 8192  --print_freq 300 --code polar --rate_profile polar  --target_K 8 --run 4 > ./Supervised_Xformer_decoder_Polar_Results/Polar_8_8/Scheme_polar/encoder/8_depth_6/4/output.txt
python -u run_models.py --model encoder --N 4 --max_len 4 --K 4 --dec_train_snr -2 --validation_snr -2  --embed_dim 32 --num_steps 10000 --batch_size 8192  --print_freq 300 --code polar --rate_profile polar --target_K 4 --run 4 > ./Supervised_Xformer_decoder_Polar_Results/Polar_4_4/Scheme_polar/encoder/8_depth_6/4/output.txt
