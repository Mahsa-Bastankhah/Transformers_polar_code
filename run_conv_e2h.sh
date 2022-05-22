#!/bin/bash
python xformer_all.py --model conv --N 64 --max_len 64 --K 1 --dec_train_snr -6 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 1000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 16 --previous_N 64 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 2 --dec_train_snr -6 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 1000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 1 --previous_N 64 --load_previous --model_iters 1000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 3 --dec_train_snr -5 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 1000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 2 --previous_N 64 --load_previous --model_iters 1000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 4 --dec_train_snr -5 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 1000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 3 --previous_N 64 --load_previous --model_iters 1000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 5 --dec_train_snr -4 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 1000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 4 --previous_N 64 --load_previous --model_iters 1000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 6 --dec_train_snr -4 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 1000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 5 --previous_N 64 --load_previous --model_iters 1000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 7 --dec_train_snr -4 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 1000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 6 --previous_N 64 --load_previous --model_iters 1000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 8 --dec_train_snr -4 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 5000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 7 --previous_N 64 --load_previous --model_iters 1000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 9 --dec_train_snr -3 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 5000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 8 --previous_N 64 --load_previous --model_iters 5000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 10 --dec_train_snr -3 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 5000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 9 --previous_N 64 --load_previous --model_iters 5000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 11 --dec_train_snr -2 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 5000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 10 --previous_N 64 --load_previous --model_iters 5000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 12 --dec_train_snr -2 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 5000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 11 --previous_N 64 --load_previous --model_iters 5000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 13 --dec_train_snr -1 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 5000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 12 --previous_N 64 --load_previous --model_iters 5000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 14 --dec_train_snr -1 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 5000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 13 --previous_N 64 --load_previous --model_iters 5000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 15 --dec_train_snr -1 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 5000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 14 --previous_N 64 --load_previous --model_iters 5000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 16 --dec_train_snr -1 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 5000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 15 --previous_N 64 --load_previous --model_iters 5000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 17 --dec_train_snr 0 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 5000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 16 --previous_N 64 --load_previous --model_iters 5000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 18 --dec_train_snr 0 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 5000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 17 --previous_N 64 --load_previous --model_iters 5000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 19 --dec_train_snr 0 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 5000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 18 --previous_N 64 --load_previous --model_iters 5000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 20 --dec_train_snr 0 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 5000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 19 --previous_N 64 --load_previous --model_iters 5000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 21 --dec_train_snr 0 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 5000 --lr 1e-3 --batch_size 8192 --mult 1 --num_restarts 1 --print_freq 200 --code polar --previous_K 20 --previous_N 64 --load_previous --model_iters 5000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 22 --dec_train_snr 0 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 150000 --lr 1e-3 --batch_size 8192 --mult 1 --cosine --num_restarts 1 --print_freq 200 --code polar --previous_K 21 --previous_N 64 --load_previous --model_iters 5000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101
python xformer_all.py --model conv --N 64 --max_len 64 --K 22 --dec_train_snr 0 --n_layers 6 --embed_dim 128 --n_head 8 --num_steps 150000 --lr 1e-3 --batch_size 8192 --mult 1 --cosine --num_restarts 1 --print_freq 200 --code polar --previous_K 21 --previous_N 64 --load_previous --model_iters 150000 --rate_profile polar --prog_mode r2l --id r2l --previous_id r2l --validation_snr 1 --target_K 22 --run 101 --test --test_snr_start -3 --test_snr_end 3 --test_batch_size 1000 --test_size 100000