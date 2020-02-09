import numpy as np

hype = np.load('hyperparameter_search.npy', allow_pickle=True)

for config in hype:
	if config['validation_accuracy'][9] > 0.97:
		print(config)	