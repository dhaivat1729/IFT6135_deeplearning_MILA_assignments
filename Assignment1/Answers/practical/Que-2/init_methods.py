"""
Author: Dhaivat Jitendra Bhatt
ID no: 20146667
Date: 09/02/2020
Status: Passed all tests on gradescope
"""


import pickle
import numpy as np
import gzip

def one_hot(y, n_classes=10):
	return np.eye(n_classes)[y]

def load_mnist():
	data_file = gzip.open("mnist.pkl.gz", "rb")
	train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
	data_file.close()

	train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
	train_results = [one_hot(y, 10) for y in train_data[1]]
	train_data = np.array(train_inputs).reshape(-1, 784), np.array(train_results).reshape(-1, 10)

	val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
	val_results = [one_hot(y, 10) for y in val_data[1]]
	val_data = np.array(val_inputs).reshape(-1, 784), np.array(val_results).reshape(-1, 10)

	test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
	test_data = list(zip(test_inputs, test_data[1]))

	return train_data, val_data, test_data

# train_data_, val_data_, test_data_ = load_mnist()

class NN(object):
	def __init__(self,
				 hidden_dims=(784, 256),
				 epsilon=1e-6,
				 lr=7e-4,
				 init_method = 'glorot',
				 batch_size=64,
				 seed=500,
				 activation="relu",
				 data=None
				 ):

		self.hidden_dims = hidden_dims
		self.n_hidden = len(hidden_dims)
		self.lr = lr
		self.batch_size = batch_size
		self.init_method = init_method
		self.seed = seed
		self.activation_str = activation
		self.epsilon = epsilon

		self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

		if data is None:
			# for testing, do NOT remove or modify
			self.train, self.valid, self.test = (
				(np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400))),
				(np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400))),
				(np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400)))
		)
		else:
			self.train, self.valid, self.test = data


	def initialize_weights(self, dims):        
		if self.seed is not None:
			np.random.seed(self.seed)

		self.weights = {}
		# self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
		all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
		for layer_n in range(1, self.n_hidden + 2):

			if self.init_method == 'glorot':
				d_l = np.sqrt(6/(all_dims[layer_n] + all_dims[layer_n - 1]))
				self.weights[f"W{layer_n}"] = np.random.uniform(low = -d_l, high = d_l, size=(all_dims[layer_n - 1], all_dims[layer_n]))
			elif self.init_method == 'zero':
				self.weights[f"W{layer_n}"] = np.zeros((all_dims[layer_n - 1], all_dims[layer_n]))
			elif self.init_method == 'normal':
				self.weights[f"W{layer_n}"] = np.random.normal(0, 1, size=(all_dims[layer_n - 1], all_dims[layer_n]))

			self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

	def relu(self, x, grad=False):

		if grad:
			x[x<0] = 0
			x[x>0] = 1
			return x
			
		x = np.maximum(0, x)
		
		return x

	def sigmoid(self, x, grad=False):
		
		## Sigmoid
		y = 1.0 / (1.0 + np.exp(-x))

		if grad:
			## gradient of sigmoid function
			return y*(1-y)
		
		return y

	def tanh(self, x, grad=False):

		y = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
		if grad:
			return 1.0 - y**2

		return y

	def activation(self, x, grad=False):

		if self.activation_str == "relu":
			return self.relu(x, grad=grad)
		
		elif self.activation_str == "sigmoid":
			return self.sigmoid(x, grad=grad)
		
		elif self.activation_str == "tanh":
			return self.tanh(x, grad=grad)
		
		else:
			raise Exception("invalid")
		
		return 0

	def softmax(self, x):
		# Remember that softmax(x-C) = softmax(x) when C is a constant.
		
		# import ipdb; ipdb.set_trace()
		if len(x.shape) == 2:
			x = x - np.max(x, axis=1, keepdims=True)
			x = np.exp(x)/(np.sum(np.exp(x), axis=1, keepdims=True))
			return x
		else: 
			x = x - np.max(x)
			x = np.exp(x)/(np.sum(np.exp(x)))
			return x

	def lin_forward(self, W, x, b):
		## x -> NxD, N = batchsize and D = input dimension
		## W -> DxM  D = input dimension, M = output dimension
		## b -> 1xM  M = output dimension
		y = np.dot(x, W) + b
		return y

	def lin_backward(self, gradient, W, x, b):
		## gradient: gradient with respect to output of the affine layer
		## W, x, b: corresponds to the affine layer for which we are doing backprop
		## useful: http://cs231n.stanford.edu/handouts/linear-backprop.pdf
		dout = gradient
		dx = np.dot(dout, np.transpose(W)) ## this is dL/dx
		dW = np.dot(np.transpose(x), dout)	## this is dL/dW
		db = np.sum(dout, 0, keepdims=True) ## this is dL/db

		return dx, dW, db

	def cross_entropy_backward(self, probs, labels):
		### Understand from: http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html

		## from one hot encoding to class labels
		labels = np.argmax(labels, axis=1) 

		dupstream = probs
		dupstream[range(labels.shape[0]),labels] -= 1
		dupstream = dupstream/labels.shape[0]

		return dupstream


	def forward(self, x):
		cache = {"Z0": x}
		# cache is a dictionnary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
		# Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
		
		cache['A1'] = self.lin_forward(self.weights['W1'], cache['Z0'], self.weights['b1'])

		for layer in range(1, self.n_hidden+1):

			## activation function
			cache[f'Z{layer}'] = self.activation(cache[f'A{layer}'])

			## forward to next layer
			cache[f'A{layer+1}'] = self.lin_forward(self.weights[f'W{layer+1}'], cache[f'Z{layer}'],self.weights[f'b{layer+1}'])

		
		## Applying softmax function
		cache[f"Z{self.n_hidden + 1}"] = self.softmax(cache[f"A{self.n_hidden + 1}"])

		return cache

	def backward(self, cache, labels):
		output = cache[f"Z{self.n_hidden + 1}"]
		grads = {}
		## dpreact and dpostact are the temp variables to store preactivation and postactivation gradients

		## grads just before applying the softmax function
		grads[f"dA{self.n_hidden + 1}"]= self.cross_entropy_backward(output, labels)

		# grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
		## let's backprop through the network
		for i in range(self.n_hidden+1, 1, -1):

			## backpropping through the linear layer
			grads[f'dZ{i-1}'], grads[f'dW{i}'], grads[f'db{i}'] = self.lin_backward(grads[f'dA{i}'], self.weights[f'W{i}'], cache[f'Z{i-1}'], self.weights[f'b{i}'])

			## backpropping through the non-linearity
			# import ipdb; ipdb.set_trace()
			grads[f'dA{i-1}'] = grads[f'dZ{i-1}']*self.activation(cache[f'A{i-1}'], grad=True)


		## backpropping all the way to the input
		_, grads[f'dW{1}'], grads[f'db{1}'] = self.lin_backward(grads[f'dA{1}'], self.weights[f'W{1}'], cache[f'Z0'], self.weights[f'b{1}'])		

		return grads

	def update(self, grads):
		for layer in range(1, self.n_hidden + 2):

			## updating weights and biases
			self.weights[f'W{layer}'] -= self.lr*grads[f'dW{layer}']
			self.weights[f'b{layer}'] -= self.lr*grads[f'db{layer}']
			


	def loss(self, prediction, labels):
		prediction[np.where(prediction < self.epsilon)] = self.epsilon
		prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
		
		## let's calculate 
		# import ipdb; ipdb.set_trace()

		## from one hot encoding to class labels
		labels = np.argmax(labels, axis=1) 

		## log probabilities of the true class!
		logprobs = -np.log(prediction[range(labels.shape[0]),labels])

		loss = np.sum(logprobs)/labels.shape[0]

		return loss

	def compute_loss_and_accuracy(self, X, y):
		one_y = y
		y = np.argmax(y, axis=1)  # Change y to integers
		cache = self.forward(X)
		predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
		accuracy = np.mean(y == predictions)
		loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
		return loss, accuracy, predictions

	def train_loop(self, n_epochs):
		X_train, y_train = self.train
		y_onehot = y_train
		dims = [X_train.shape[1], y_onehot.shape[1]]
		self.initialize_weights(dims)

		n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

		

		for epoch in range(n_epochs):
			total_epoch_loss = 0
			for batch in range(n_batches):
				minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
				minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
				
				## forward pass
				cache = self.forward(minibatchX)

				## computing loss
				loss = self.loss(cache[f"Z{self.n_hidden + 1}"], minibatchY)
				total_epoch_loss += loss

				grads = self.backward(cache, minibatchY)

				## time for updating the parameters
				self.update(grads)

			print(f"Normalized epoch loss is: {total_epoch_loss / X_train.shape[0]}")

			X_train, y_train = self.train
			train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
			X_valid, y_valid = self.valid
			valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

			self.train_logs['train_accuracy'].append(train_accuracy)
			self.train_logs['validation_accuracy'].append(valid_accuracy)
			self.train_logs['train_loss'].append(train_loss)
			self.train_logs['validation_loss'].append(valid_loss)

		return self.train_logs

	def evaluate(self):
		X_test, y_test = self.test
		test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
		return test_loss, test_accuracy


init_methods = ['glorot', 'zero', 'normal']

output = {}

n_epochs = 10

for init_method in init_methods:

	nn = NN(init_method=init_method, data=load_mnist())
	output[init_method] = nn.train_loop(n_epochs)

import ipdb; ipdb.set_trace()