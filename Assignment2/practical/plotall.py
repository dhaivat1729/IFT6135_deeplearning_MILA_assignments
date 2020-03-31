import matplotlib.pyplot as plt
import numpy as np
import os
import glob


exps = ['Exp1','Exp2','Exp3','Exp4','Exp5','Exp6','Exp7','Exp8','Exp9','Exp10','Exp11','Exp12','Exp13','Exp14','Exp15']
# exps = ['Exp1','Exp3','Exp4','Exp5']
# label = ['RNN_SGD_LR_1_BS_128_SL_35_HS_512_NL_2_DKP_8E-1', 'RNN_SGD_LR_10_BS_128_SL_35_HS_512_NL_2_DKP_8E-1', 'RNN_ADAM_LR_1e-3_BS_128_SL_35_HS_512_NL_2_DKP_8E-1', 'RNN_ADAM_LR_1e-4_BS_128_SL_35_HS_512_NL_2_DKP_8E-1', '']
# 'GRU_ADAM_LR_1e-3_BS_128_NL_2'

label = ['3.1.1','3.1.2','3.1.3','3.1.4','3.1.5','3.2.1','3.2.2','3.2.3','3.3.1','3.3.2','3.3.3','3.4.1','3.4.2','3.4.3','3.4.4']

for ind, exp in enumerate(exps):
	dirs = glob.glob(exp+'/*/*.npy',recursive=True)#, GRU_SGD_LR, TX_SGD_LR]
	if len(dirs) == 0:
		continue
	labels = label[ind]
	# labels = ['RNN_SGD_model', 'RNN_SGD_model', 'RNN_ADAM_model', 'RNN_ADAM_model']#, 'GRU_SGD_LR', 'TX_SGD_LR']
	colors = ['C'+str(i+1) for i in range(len(dirs))]

	# Read learning_curves
	train_ppls_all, val_ppls_all, epochs_all, times_all = [], [], [], []
	for d in dirs:
		lc_path = os.path.join(d)
		a = np.load(lc_path, allow_pickle=True)[()]
		train_ppls, val_ppls, times = a['train_ppls'], a['val_ppls'], a['times']
		epochs = np.arange(len(times)) + 1
		times = np.cumsum(times)
		train_ppls_all.append(train_ppls)
		val_ppls_all.append(val_ppls)
		epochs_all.append(epochs)
		times_all.append(times)

	# Plot
	plt.figure(figsize=(12, 6))

	# vs epochs
	plt.subplot(121)
	for i in range(len(dirs)):
		plt.plot(epochs_all[i], train_ppls_all[i], '--o', mfc='none', color=colors[i], alpha=0.7, label=labels+" Train PPL")
		plt.plot(epochs_all[i], val_ppls_all[i], '-o', alpha=0.7, color=colors[i], label=labels+" Val PPL")
		plt.legend()
		# plt.yscale("symlog")
		# plt.ylim([0, 1000])
		plt.title('Experiment ' + labels)
		# plt.ylabel("PPL (in log scale)")
		plt.ylabel("PPL")
		plt.xlabel("Epochs")

	# vs times
	plt.subplot(122)
	for i in range(len(dirs)):
		plt.plot(times_all[i], train_ppls_all[i], '--o', mfc='none', color=colors[i], alpha=0.7, label=labels+" Train PPL")
		plt.plot(times_all[i], val_ppls_all[i], '-o', alpha=0.7, color=colors[i], label=labels+" Val PPL")
		plt.legend()
		# plt.yscale("symlog")
		# plt.ylim([0, 1000])
		plt.title('Experiment ' + labels)
		# plt.ylabel("PPL (in log scale)")
		plt.ylabel("PPL")
		plt.xlabel("Wall clock time (seconds)")

	# plt.subplots_adjust(hspace=.5)

	plt.savefig(exp+'_model_compare.png', bbox_inches='tight', pad_inches=0.2)
	plt.clf()
	plt.close()


