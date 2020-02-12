import numpy as np
import csv


hype = np.load('hyperparameter_search.npy', allow_pickle=True)

f = open('hyperparams_best.csv', 'w')


rows = []
for config in hype:
	if config['validation_accuracy'][9] >= 0.97:
		out = [config['hidden_dims'], config['lr'], config['activation'], config['train_accuracy'][9], config['validation_accuracy'][9], config['model_params']]
		rows.append(out)

main_row = ['hidden_dims', 'learning rate', 'activation function', 'training accuracy', 'validation_accuracy', 'model parameters']

with f:
	writer = csv.writer(f)
	writer.writerow(main_row)
	for row in rows:
		writer.writerow(row)

f = open('hyperparams_full.csv', 'w')


rows = []
for config in hype:
	out = [config['hidden_dims'], config['lr'], config['activation'], config['train_accuracy'][9], config['validation_accuracy'][9], config['model_params']]
	rows.append(out)

main_row = ['hidden_dims', 'learning rate', 'activation function', 'training accuracy', 'validation_accuracy', 'model parameters']

with f:
	writer = csv.writer(f)
	writer.writerow(main_row)
	for row in rows:
		writer.writerow(row)