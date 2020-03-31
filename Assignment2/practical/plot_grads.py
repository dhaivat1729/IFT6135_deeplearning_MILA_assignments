
"""
load results stored in numpy file and plot losses. 

Author: Dhaivat Jitendra Bhatt
UdeM ID: 20146667

How to run?

python plot_losses.py

"""

import numpy as np
import matplotlib.pyplot as plt

## loading the results
rnn = np.load('RNN_hidden_grads.npy', allow_pickle=True)[()]
gru = np.load('GRU_hidden_grads.npy', allow_pickle=True)[()]

rnn_train = rnn['train_hidden_grads']
rnn_val = rnn['val_hidden_grads']

rnn_val = rnn_val/np.max(rnn_val)

gru_train = gru['train_hidden_grads']
gru_val = gru['val_hidden_grads']
gru_val = gru_val/np.max(gru_val)


colors = ['C'+str(i+1) for i in range(4)]
## plot for validation loss

# plt.plot(rnn_train, '--o', mfc='none', color=colors[0], alpha=0.7, label="train data + RNN")
plt.plot(rnn_val, '-o', alpha=0.7, color=colors[1], label="val data + RNN")
# plt.plot(gru_train, '--o', mfc='none', color=colors[2], alpha=0.7, label="train data + GRU")
plt.plot(gru_val, '-o', alpha=0.7, color=colors[3], label="val data + GRU")
plt.legend()
	

plt.legend(loc='upper left')
plt.xlabel('timestep ->')
plt.ylabel('gradient norm -> ')
plt.title('gradient norm for GRU and RNN')
plt.savefig('grad_norm.png')

# plt.clf()

# ## plot for training loss
# for key in output[0].keys():
# 	plt.plot(output[0][key]['train_loss'], label=f'{key} initialization')
	

# plt.legend(loc='center right')
# plt.xlabel('epoch ->')
# plt.ylabel('training loss -> ')
# plt.title('Training losses for different weight initialization schemes')
# plt.savefig('train_losses.png')

