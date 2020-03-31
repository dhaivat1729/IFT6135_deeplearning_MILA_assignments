import matplotlib.pyplot as plt
import numpy as np
import os

exp1 = '/network/home/bhattdha/IFT6135H20_assignment/Assignment2/practical/Exp1/RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_save_best_save_dir=Exp1_0'
exp2 = '/network/home/bhattdha/IFT6135H20_assignment/Assignment2/practical/Exp3/RNN_SGD_model=RNN_optimizer=SGD_initial_lr=10.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_save_dir=Exp3_0'
exp3 = '/network/home/bhattdha/IFT6135H20_assignment/Assignment2/practical/Exp4/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_save_dir=Exp4_0'
exp4 = '/network/home/bhattdha/IFT6135H20_assignment/Assignment2/practical/Exp5/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_save_dir=Exp5_0'
# GRU_SGD_LR = 'GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0'
# TX_SGD_LR = 'TRANSFORMER_SGD_LR_SCHEDULE_model=TRANSFORMER_optimizer=SGD_LR_SCHEDULE_initial_lr=20_batch_size=128_seq_len=35_hidden_size=512_num_layers=6_dp_keep_prob=0.9_save_best_0'

dirs = [exp1, exp2, exp4, exp4]#, GRU_SGD_LR, TX_SGD_LR]
labels = ['RNN_SGD_model', 'RNN_SGD_model', 'RNN_ADAM_model', 'RNN_ADAM_model']#, 'GRU_SGD_LR', 'TX_SGD_LR']
colors = ['C'+str(i+1) for i in range(len(dirs))]

# Read learning_curves
train_ppls_all, val_ppls_all, epochs_all, times_all = [], [], [], []
for d in dirs:
    lc_path = os.path.join(d, 'learning_curves.npy')
    a = np.load(lc_path, allow_pickle=True)[()]
    train_ppls, val_ppls, times = a['train_ppls'], a['val_ppls'], a['times']
    epochs = np.arange(len(times)) + 1
    times = np.cumsum(times)
    train_ppls_all.append(train_ppls)
    val_ppls_all.append(val_ppls)
    epochs_all.append(epochs)
    times_all.append(times)

# Plot
plt.figure(figsize=(10, 12))

# vs epochs
plt.subplot(211)
for i in range(len(dirs)):
    plt.plot(epochs_all[i], train_ppls_all[i], '--o', mfc='none', color=colors[i], alpha=0.7, label=labels[i]+" Train PPL")
    plt.plot(epochs_all[i], val_ppls_all[i], '-o', alpha=0.7, color=colors[i], label=labels[i]+" Val PPL")
    plt.legend()
    # plt.yscale("symlog")
    plt.ylim([0, 1000])
    plt.title("PPL sv Epochs")
    # plt.ylabel("PPL (in log scale)")
    plt.ylabel("PPL")
    plt.xlabel("Epochs")

# vs times
plt.subplot(212)
for i in range(len(dirs)):
    plt.plot(times_all[i], train_ppls_all[i], '--o', mfc='none', color=colors[i], alpha=0.7, label=labels[i]+" Train PPL")
    plt.plot(times_all[i], val_ppls_all[i], '-o', alpha=0.7, color=colors[i], label=labels[i]+" Val PPL")
    plt.legend()
    # plt.yscale("symlog")
    plt.ylim([0, 1000])
    plt.title("PPL vs Wall clock time")
    # plt.ylabel("PPL (in log scale)")
    plt.ylabel("PPL")
    plt.xlabel("Wall clock time (seconds)")

# plt.subplots_adjust(hspace=.5)

plt.savefig('1_model_compare.png', bbox_inches='tight', pad_inches=0.2)
plt.clf()
plt.close()
