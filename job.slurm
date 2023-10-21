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
conda activate ml-env
 
python -u run_models.py --model encode --N 8 --max_len 8 --K 8 --dec_train_snr 4 --embed_dim 32 --num_steps 10000 --batch_size 10000 --lr 1e-3 --mult 1 --num_restarts 1 --print_freq 500 --code polar --rate_profile polar --validation_snr 4 --target_K 8 --run 2 >> output8-8.txt
python -u run_models.py --model encode --N 4 --max_len 4 --K 4 --dec_train_snr 4 --embed_dim 32 --num_steps 10000 --batch_size 10000 --lr 1e-3 --mult 1 --num_restarts 1 --print_freq 500 --code polar --rate_profile polar --validation_snr 4 --target_K 4 --run 1 >> output4-4.txt