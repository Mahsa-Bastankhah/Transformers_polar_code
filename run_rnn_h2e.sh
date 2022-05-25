python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 8 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 10000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id run1 --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 9 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 5000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id init_864_run1 --load_path Supervised_RNN_Polar_Results/final_nets/Scheme_rev_polar_22/N64_K8_y_input_onehot_GRU_depth_2_fsize_512_y_depth_0_hsize_0_snr_0.0_bs_4096_tfr_1.0_activ_selu_init_He_optim_AdamW_lr_0.001_decay_2000_step_loss_MSE_run1.pt --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 10 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 5000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id init_964_run1 --load_path Supervised_RNN_Polar_Results/final_nets/Scheme_rev_polar_22/N64_K9_y_input_onehot_GRU_depth_2_fsize_512_y_depth_0_hsize_0_snr_0.0_bs_4096_tfr_1.0_activ_selu_init_He_optim_AdamW_lr_0.001_decay_2000_step_loss_MSE_init_864_run1.pt --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 11 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 5000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id init_1064_run1 --load_path Supervised_RNN_Polar_Results/final_nets/Scheme_rev_polar_22/N64_K10_y_input_onehot_GRU_depth_2_fsize_512_y_depth_0_hsize_0_snr_0.0_bs_4096_tfr_1.0_activ_selu_init_He_optim_AdamW_lr_0.001_decay_2000_step_loss_MSE_init_964_run1.pt --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 12 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 5000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id init_1164_run1 --load_path Supervised_RNN_Polar_Results/final_nets/Scheme_rev_polar_22/N64_K11_y_input_onehot_GRU_depth_2_fsize_512_y_depth_0_hsize_0_snr_0.0_bs_4096_tfr_1.0_activ_selu_init_He_optim_AdamW_lr_0.001_decay_2000_step_loss_MSE_init_1064_run1.pt --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 13 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 5000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id init_1264_run1 --load_path Supervised_RNN_Polar_Results/final_nets/Scheme_rev_polar_22/N64_K12_y_input_onehot_GRU_depth_2_fsize_512_y_depth_0_hsize_0_snr_0.0_bs_4096_tfr_1.0_activ_selu_init_He_optim_AdamW_lr_0.001_decay_2000_step_loss_MSE_init_1164_run1.pt --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 14 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 5000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id init_1364_run1 --load_path Supervised_RNN_Polar_Results/final_nets/Scheme_rev_polar_22/N64_K13_y_input_onehot_GRU_depth_2_fsize_512_y_depth_0_hsize_0_snr_0.0_bs_4096_tfr_1.0_activ_selu_init_He_optim_AdamW_lr_0.001_decay_2000_step_loss_MSE_init_1264_run1.pt --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 15 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 5000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id init_1464_run1 --load_path Supervised_RNN_Polar_Results/final_nets/Scheme_rev_polar_22/N64_K14_y_input_onehot_GRU_depth_2_fsize_512_y_depth_0_hsize_0_snr_0.0_bs_4096_tfr_1.0_activ_selu_init_He_optim_AdamW_lr_0.001_decay_2000_step_loss_MSE_init_1364_run1.pt --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 16 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 5000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id init_1564_run1 --load_path Supervised_RNN_Polar_Results/final_nets/Scheme_rev_polar_22/N64_K15_y_input_onehot_GRU_depth_2_fsize_512_y_depth_0_hsize_0_snr_0.0_bs_4096_tfr_1.0_activ_selu_init_He_optim_AdamW_lr_0.001_decay_2000_step_loss_MSE_init_1464_run1.pt --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 17 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 5000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id init_1664_run1 --load_path Supervised_RNN_Polar_Results/final_nets/Scheme_rev_polar_22/N64_K16_y_input_onehot_GRU_depth_2_fsize_512_y_depth_0_hsize_0_snr_0.0_bs_4096_tfr_1.0_activ_selu_init_He_optim_AdamW_lr_0.001_decay_2000_step_loss_MSE_init_1564_run1.pt --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 18 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 5000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id init_1764_run1 --load_path Supervised_RNN_Polar_Results/final_nets/Scheme_rev_polar_22/N64_K17_y_input_onehot_GRU_depth_2_fsize_512_y_depth_0_hsize_0_snr_0.0_bs_4096_tfr_1.0_activ_selu_init_He_optim_AdamW_lr_0.001_decay_2000_step_loss_MSE_init_1664_run1.pt --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 19 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 5000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id init_1864_run1 --load_path Supervised_RNN_Polar_Results/final_nets/Scheme_rev_polar_22/N64_K18_y_input_onehot_GRU_depth_2_fsize_512_y_depth_0_hsize_0_snr_0.0_bs_4096_tfr_1.0_activ_selu_init_He_optim_AdamW_lr_0.001_decay_2000_step_loss_MSE_init_1764_run1.pt --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 20 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 5000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id init_1964_run1 --load_path Supervised_RNN_Polar_Results/final_nets/Scheme_rev_polar_22/N64_K19_y_input_onehot_GRU_depth_2_fsize_512_y_depth_0_hsize_0_snr_0.0_bs_4096_tfr_1.0_activ_selu_init_He_optim_AdamW_lr_0.001_decay_2000_step_loss_MSE_init_1864_run1.pt --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 21 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 5000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id init_2064_run1 --load_path Supervised_RNN_Polar_Results/final_nets/Scheme_rev_polar_22/N64_K20_y_input_onehot_GRU_depth_2_fsize_512_y_depth_0_hsize_0_snr_0.0_bs_4096_tfr_1.0_activ_selu_init_He_optim_AdamW_lr_0.001_decay_2000_step_loss_MSE_init_1964_run1.pt --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
python3 rnn_all.py --code Polar --rate_profile rev_polar --target_K 22 --N 64 --K 22 --decoding_type y_input --rnn_feature_size 512 --y_hidden_size 128 --y_depth 3 --num_steps 100000 --batch_size 4096 --rnn_depth 2 --model_save_per 10000 --tfr_min 1 --tfr_max 1 --dec_train_snr 0 --lr 0.001 --scheduler step --lr_decay 2000 --lr_decay_gamma 0.95 --onehot --id init_2164_run1 --load_path Supervised_RNN_Polar_Results/final_nets/Scheme_rev_polar_22/N64_K21_y_input_onehot_GRU_depth_2_fsize_512_y_depth_0_hsize_0_snr_0.0_bs_4096_tfr_1.0_activ_selu_init_He_optim_AdamW_lr_0.001_decay_2000_step_loss_MSE_init_2064_run1.pt --test_codes y --test_bitwise --testing_snr 0 --progressive_path Supervised_RNN_Polar_Results/progressive/N64_K22_H2E --gpu 3 --test_snr_start -3 --test_snr_end 3
