
"""
load results stored in numpy file and plot losses. 

Author: Dhaivat Jitendra Bhatt
UdeM ID: 20146667

How to run?

python plot_CNN_losses.py

Check out 2 png files created! 

"""

import numpy as np
import matplotlib.pyplot as plt

## loading the results
vanilla_CNN = np.load('vanilla_CNN.npy', allow_pickle=True)



plt.plot(vanilla_CNN[()]['val_loss'], 'o-', label=f'validation loss')
plt.plot(vanilla_CNN[()]['train_loss'], 'x-', label=f'training loss')
plt.legend(loc='upper right')
plt.xlabel('epoch ->')
plt.ylabel('loss value -> ')
plt.title('Loss profile of vanilla CNN (763,094 parameters)')
plt.savefig('vanilla_CNN_losses.png')


#clear figure
plt.clf()

vanilla_CNN = np.load('CNN_with_reg.npy', allow_pickle=True)

plt.plot(vanilla_CNN[()]['val_loss'], 'o-', label=f'validation loss')
plt.plot(vanilla_CNN[()]['train_loss'], 'x-', label=f'training loss')
plt.legend(loc='upper right')
plt.xlabel('epoch ->')
plt.ylabel('loss value -> ')
plt.title('Loss profile for regularized CNN (763,094 parameters)')
plt.savefig('CNN_with_reg_losses.png')

## plotting equivalent MLP result
## loading the results
output = np.load('hyperparameter_search.npy', allow_pickle=True)
for config in output:
	if config['validation_accuracy'][9] == 0.9764:
		train_accuracy = config['train_loss']
		validation_accuracy = config['validation_loss']
		
plt.clf()


plt.plot(validation_accuracy, 'o-', label=f'validation loss')
plt.plot(train_accuracy, 'x-', label=f'training loss')
plt.legend(loc='upper right')
plt.xlabel('epoch ->')
plt.ylabel('loss value -> ')
plt.title('Loss profile of MLP (748,261 parameters)')
plt.savefig('MLP_loss_profile.png')		