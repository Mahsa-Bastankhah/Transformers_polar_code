#!/bin/bash
#SBATCH --job-name=<name>         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=mig
#SBATCH --mail-type=fail        # send email when job fails
#SBATCH --mail-user=mb6458@princeton.edu

module purge
module load anaconda3/2023.3
source activate mlenv

run=3
n_head=1                                         
n_layers=2
num_steps=300000
print_freq=100
N=16
K=8
snr=3
batch_size=8192
embed_dim=64
model_save_per=1000


# mkdir ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}
# mkdir ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar
# mkdir ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder
# mkdir ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}
# mkdir ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}
# touch ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt


#python -u run_models.py --code_table --model_iters 5000 --embed_dim ${embed_dim} --N ${N} --K ${K} --max_len ${N} --dec_train_snr ${snr}  --n_head ${n_head} --n_layers ${n_layers}  --target_K ${K} --run ${run} >> ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
#python -u run_models.py --gradual_testing --model_save_per ${model_save_per} --embed_dim ${embed_dim}  --load_previous --model_iters 10000 --previous_id 0  --model encoder --N ${N} --K ${K} --max_len ${N} --dec_train_snr ${snr} --validation_snr ${snr} --n_head ${n_head} --n_layers ${n_layers} --num_steps ${num_steps} --batch_size ${batch_size} --print_freq ${print_freq} --code polar --rate_profile polar  --target_K ${K} --run ${run} > ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt

#sleep 3
#python run_models.py --test --embed_dim ${embed_dim}  --model encoder --dec_train_snr ${snr} --run ${run} --N ${N} --K ${K} --max_len ${N} --n_head ${n_head} --n_layers ${n_layers} --print_freq ${print_freq} --code polar --rate_profile polar --target_K ${K} --test_snr_start -10 --test_snr_end 30  --test_batch_size 100 --test_size 10000 >> ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
#python run_models.py --test --model_iters 15000 --n_head ${n_head} --n_layers ${n_layers} --embed_dim ${embed_dim} --model encoder --dec_train_snr ${snr} --run ${run} --N ${N} --K ${K} --max_len ${N}  --print_freq 100 --code polar --rate_profile polar --target_K ${K} --test_snr_start -10 --test_snr_end 30 --test_batch_size 1000 --test_size 10000 >> ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt

#mkdir ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/plots
#python PlotBitwiseErr.py ${N} ${K} ${snr} ${run} ${print_freq} --n_head ${n_head} --n_layers ${n_layers}  --rng 500 >> ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt



################### REPORT TEST BER ###################
python -u report_test_ber.py 25000 50000 ${N} ${K} ${run} ${print_freq}  --n_head ${n_head} --n_layers ${n_layers} >> ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
python -u run_models.py --code_table --model_iters 10 --embed_dim ${embed_dim} --N ${N} --K ${K} --max_len ${N} --dec_train_snr ${snr}  --n_head ${n_head} --n_layers ${n_layers}  --target_K ${K} --run ${run} >> ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
python -u run_models.py --code_table --model_iters 1000 --embed_dim ${embed_dim} --N ${N} --K ${K} --max_len ${N} --dec_train_snr ${snr}  --n_head ${n_head} --n_layers ${n_layers}  --target_K ${K} --run ${run} >> ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
python -u run_models.py --code_table --model_iters 3000 --embed_dim ${embed_dim} --N ${N} --K ${K} --max_len ${N} --dec_train_snr ${snr}  --n_head ${n_head} --n_layers ${n_layers}  --target_K ${K} --run ${run} >> ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
python -u run_models.py --code_table --model_iters 5000 --embed_dim ${embed_dim} --N ${N} --K ${K} --max_len ${N} --dec_train_snr ${snr}  --n_head ${n_head} --n_layers ${n_layers}  --target_K ${K} --run ${run} >> ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
python -u run_models.py --code_table --model_iters 10000 --embed_dim ${embed_dim} --N ${N} --K ${K} --max_len ${N} --dec_train_snr ${snr}  --n_head ${n_head} --n_layers ${n_layers}  --target_K ${K} --run ${run} >> ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
python -u run_models.py --code_table --model_iters 12000 --embed_dim ${embed_dim} --N ${N} --K ${K} --max_len ${N} --dec_train_snr ${snr}  --n_head ${n_head} --n_layers ${n_layers}  --target_K ${K} --run ${run} >> ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
python -u run_models.py --code_table --model_iters 50000 --embed_dim ${embed_dim} --N ${N} --K ${K} --max_len ${N} --dec_train_snr ${snr}  --n_head ${n_head} --n_layers ${n_layers}  --target_K ${K} --run ${run} >> ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
python -u run_models.py --code_table --model_iters 100000 --embed_dim ${embed_dim} --N ${N} --K ${K} --max_len ${N} --dec_train_snr ${snr}  --n_head ${n_head} --n_layers ${n_layers}  --target_K ${K} --run ${run} >> ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
################### REPORT BER OVER SNRS ###################
# for model_iter in $(seq $model_save_per $model_save_per $num_steps); do
#     echo "Running with model_iter=$model_iter"
#     python run_models.py --test --model_iters $model_iter --n_head ${n_head} --n_layers ${n_layers} --embed_dim ${embed_dim} --model encoder --dec_train_snr ${snr} --run ${run} --N ${N} --K ${K} --max_len ${N}  --print_freq 100 --code polar --rate_profile polar --target_K ${K} --test_snr_start -10 --test_snr_end 20 --test_batch_size 1000 --test_size 10000 >> ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
#     sleep 3
# done

#python processAttnData.py --oe_test ${N} ${K} ${snr} ${run} ${print_freq} --rng 80000 --n_head ${n_head} --n_layers ${n_layers} 
#python processAttnData.py --oe_train ${N} ${K} ${snr} ${run} ${print_freq} --rng 80000 --n_head ${n_head} --n_layers ${n_layers} 
#python processAttnData.py --compos ${N} ${K} ${snr} ${run} ${print_freq} --n_head ${n_head} --n_layers ${n_layers} 


# (16,8), (8,4) >> SNR = 1.2, 15000 steps needed
# (8,8) , (4,4) >> SNR = 5.5, 5000 steps is enough