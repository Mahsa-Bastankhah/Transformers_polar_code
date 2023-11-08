#!/bin/bash


run=1
n_head=8
n_layers=6
num_steps=10
print_freq=10
N=16
K=8


#mkdir ./Supervised_Xformer_decoder_Polar_Results/Polar_8_8/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}
#touch ./Supervised_Xformer_decoder_Polar_Results/Polar_8_8/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
# mkdir ./Supervised_Xformer_decoder_Polar_Results/Polar_4_4/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}
# touch ./Supervised_Xformer_decoder_Polar_Results/Polar_4_4/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
#mkdir ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}
#touch ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt

#python -u run_models.py --model encoder --N ${N} --K ${K} --max_len ${N} --dec_train_snr 100 --validation_snr 100 --n_head ${n_head} --n_layers ${n_layers} --num_steps ${num_steps} --batch_size 8192  --print_freq ${print_freq} --code polar --rate_profile polar  --target_K ${K} --run ${run} > ./Supervised_Xformer_decoder_Polar_Results/Polar_${K}_${N}/Scheme_polar/encoder/${n_head}_depth_${n_layers}/${run}/output.txt
python run_models.py --test --model encoder --run ${run} --N ${N} --K ${K} --max_len ${N}  --print_freq 100 --code polar --rate_profile polar --target_K ${K} --test_snr_start -3 --test_snr_end 10 --test_batch_size 100 --test_size 100