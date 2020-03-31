#!/bin/bash

# ## PROBLEM-SPECIFIC INSTRUCTIONS:
#    - For Problem 3.1 the hyperparameter settings you should run are as follows
#            --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=40 --save_best
#            --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=20  --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=40
#            --model=RNN --optimizer=SGD --initial_lr=10.0 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=40
#            --model=RNN --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=40
#            --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=40

#    - For Problem 3.2 the hyperparameter settings you should run are as follows
#            --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=40 --save_best
#            --model=GRU --optimizer=SGD  --initial_lr=10.0  --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=40
#            --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=40

#    - For Problem 3.3 the hyperparameter settings you should run are as follows
#            --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=2 --dp_keep_prob=0.2  --num_epochs=40
#            --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=2048 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=40
#            --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=4 --dp_keep_prob=0.5  --num_epochs=40

#    - For Problem 3.4 the hyperparameter settings you should run are as follows
#            --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512  --num_layers=6 --dp_keep_prob=0.9 --num_epochs=40
#            --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512  --num_layers=2 --dp_keep_prob=0.9 --num_epochs=40
#            --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=2048 --num_layers=2 --dp_keep_prob=0.6 --num_epochs=40
#            --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=1024 --num_layers=6 --dp_keep_prob=0.9 --num_epochs=40


# python run_exp.py --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_best --save_dir='Exp1' &
# python run_exp.py --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=20  --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_dir='Exp2' &
# python run_exp.py --model=RNN --optimizer=SGD --initial_lr=10.0 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_dir='Exp3' &
# python run_exp.py --model=RNN --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_dir='Exp4' &
# python run_exp.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_dir='Exp5' &
# python run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20 --save_best --save_dir='Exp6' &
# python run_exp.py --model=GRU --optimizer=SGD  --initial_lr=10.0  --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20 --save_dir='Exp7' &
# python run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20 --save_dir='Exp8' &
# python run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=2 --dp_keep_prob=0.2  --num_epochs=20 --save_dir='Exp9' &
# python run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=2048 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20 --save_dir='Exp10' &
python run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=4 --dp_keep_prob=0.5  --num_epochs=20 --save_dir='Exp11' &
python run_exp.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512  --num_layers=6 --dp_keep_prob=0.9 --num_epochs=20 --save_dir='Exp12' &
python run_exp.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512  --num_layers=2 --dp_keep_prob=0.9 --num_epochs=20 --save_dir='Exp13' &
python run_exp.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=2048 --num_layers=2 --dp_keep_prob=0.6 --num_epochs=20 --save_dir='Exp14' &
python run_exp.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=1024 --num_layers=6 --dp_keep_prob=0.9 --num_epochs=20 --save_dir='Exp15' &