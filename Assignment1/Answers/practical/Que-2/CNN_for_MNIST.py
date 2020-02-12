
## Importing torch modules
import pickle
import numpy as np
import gzip
import torch
import torchvision 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random

torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
np.random.seed(100)
random.seed(100)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()


		"""
		conv layers
		"""

		## layer 1
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)

		## layer 2
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=[2,2], padding=2)

		## layer 3
		self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)

		## layer 4
		self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=[2,2], padding=2)		

		## layer 4
		self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)		

		## linear layers
		self.lin = nn.Linear(256*3*3,100)
		self.flin = nn.Linear(100, 10)

		"""
		Pooling layers
		"""
		self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)
		self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)


	def forward(self, x):

		
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.pool2(F.relu(self.conv5(x)))
		x = self.lin(x.view(-1,256*3*3))
		x = self.flin(x.view(-1,100))

		return x



## image transform
transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5,), (0.5,))])


## trainset

batch_size = 256

trainset = torchvision.datasets.MNIST(root='~/MNIST/', train=True,
										download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
										  shuffle=True, num_workers=2)

## testset
testset = torchvision.datasets.MNIST(root='~/MNIST/', train=False,
									   download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
										 shuffle=False, num_workers=2)

## let's define a network
net = CNN()

cuda_available = torch.cuda.is_available()
print ('cuda availability: {}'.format(cuda_available))
if cuda_available:
	net = net.cuda()


## defining loss criterion
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9, weight_decay = 0.01)
optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

## counting number of parameters in the model
params_count = sum(p.numel() for p in net.parameters()) 
print(params_count)	

# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# out = net(images)

loss_profile = {}
train_loss = []
train_accuracy = []
val_loss = []
validation_accuracy = []
## let's train the model!
for epoch in range(10):  # loop over the dataset multiple times


	total = 0
	correct = 0
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		if cuda_available:
			inputs, labels = inputs.cuda(), labels.cuda()

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		print('[%d, %5d] Train loss: %.3f' %(epoch + 1, i + 1, running_loss / (i+1)))

		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	
	running_loss = running_loss / len(trainloader)
	train_accuracy.append(100.0 * correct / total)
	train_loss.append(running_loss)

	total = 0
	correct = 0
	
	running_loss = 0

	for i, data in enumerate(testloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		if cuda_available:
			inputs, labels = inputs.cuda(), labels.cuda()


		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)

		# print statistics
		running_loss += loss.item()
		print('[%d, %5d] Validation loss: %.3f' %(epoch + 1, i + 1, running_loss / (i+1)))

		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	running_loss = running_loss / len(testloader)
	validation_accuracy.append(100.0 * correct / total)
	val_loss.append(running_loss)
	running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
	for data in testloader:
		inputs, labels = data
		if cuda_available:
			inputs, labels = inputs.cuda(), labels.cuda()
			
		outputs = net(inputs)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %.5f %%' % (
	100.0 * correct / total))

loss_profile['train_loss'] = train_loss
loss_profile['val_loss'] = val_loss
loss_profile['train_accuracy'] = train_accuracy
loss_profile['val_accuracy'] = validation_accuracy
np.save('vanila_CNN.npy',loss_profile)

import ipdb; ipdb.set_trace()