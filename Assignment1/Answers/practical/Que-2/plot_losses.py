
"""
load results stored in numpy file and plot losses. 

Author: Dhaivat Jitendra Bhatt
UdeM ID: 20146667

How to run?

python plot_losses.py

Check out 2 png files created! 

"""

import numpy as np
import matplotlib.pyplot as plt

## loading the results
output = np.load('different_initialization.npy', allow_pickle=True)


## plot for validation loss
for key in output[0].keys():
	plt.plot(output[0][key]['validation_loss'], label=f'{key} initialization')
	

plt.legend(loc='center right')
plt.xlabel('epoch ->')
plt.ylabel('validation loss -> ')
plt.title('Validation losses for different weight initialization schemes')
plt.savefig('validation_losses.png')

plt.clf()

## plot for training loss
for key in output[0].keys():
	plt.plot(output[0][key]['train_loss'], label=f'{key} initialization')
	

plt.legend(loc='center right')
plt.xlabel('epoch ->')
plt.ylabel('training loss -> ')
plt.title('Training losses for different weight initialization schemes')
plt.savefig('train_losses.png')

