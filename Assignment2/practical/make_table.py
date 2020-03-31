import numpy as np
import csv
import glob

files = glob.glob('*/*/*.npy',recursive=True)
files.sort()
f = open('table.csv', 'w')

label = ['train_epoch','3.1.1(RNN+SGD)','3.1.2(RNN+SGD)','3.1.3(RNN+SGD)','3.1.4(RNN+ADAM)','3.1.5(RNN+ADAM)','3.2.1(GRU+ADAM)','3.2.2(GRU+SGD)','3.2.3(GRU+ADAM)','3.3.1(GRU+ADAM)','3.3.2(GRU+ADAM)','3.3.3(GRU+ADAM)','3.4.1(TF+ADAM)','3.4.2(TF+ADAM)','3.4.3(TF+ADAM)','3.4.4(TF+ADAM)']
epoch = 20
rows = []
for e in range(epoch):
	row = ['Epoch #'+str(e).zfill(2)]
	for l, file in enumerate(files):
		str_is = 'Exp'+str(l)+'/*/*.npy'
		# import ipdb; ipdb.set_trace()
		a = np.load(glob.glob('Exp'+str(l+1)+'/*/*.npy')[0], allow_pickle=True)[()]
		train_ppls = a['train_ppls']
		row.append(round(train_ppls[e], 2))
	rows.append(row)


# rows = []
# for config in hype:
# 	if config['validation_accuracy'][9] >= 0.97:
# 		out = [config['hidden_dims'], config['lr'], config['activation'], config['train_accuracy'][9], config['validation_accuracy'][9], config['model_params']]
# 		rows.append(out)

# main_row = ['hidden_dims', 'learning rate', 'activation function', 'training accuracy', 'validation_accuracy', 'model parameters']

rows.append(['val_epoch','','','','','','','','','','','','','','',''])

for e in range(epoch):
	row = ['Epoch #'+str(e).zfill(2)]
	for l, file in enumerate(files):
		a = np.load(glob.glob('Exp'+str(l+1)+'/*/*.npy')[0], allow_pickle=True)[()]
		val_ppls = a['val_ppls']
		row.append(round(val_ppls[e], 2))
	rows.append(row)

with f:
	writer = csv.writer(f)
	writer.writerow(label)
	for row in rows:
		writer.writerow(row)




