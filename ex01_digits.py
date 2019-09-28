
#matplotlib inline
#config InlineBackend.figure_format = 'retina'

import torch
import numpy as np
#import helper
import matplotlib.pyplot as plt 
from torchvision import datasets, transforms
torch.manual_seed(7)


def sigmoid(x):
    return 1/(1+torch.exp(-x))

def softmax(x):
    x_exp = torch.exp(x)
    den = x_exp.sum(dim=1, keepdim=True)
    #for each batch output, divide the exp of the element with the sum of all elements' exps
    return x_exp / den

#get and load the dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))
                             ])

trainset = datasets.MNIST('MNIST_data/', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#create a dataset iterator, for each batch
dataiter = iter(trainloader)
image, label = dataiter.next()

#flatten the batch image and get the parameters of the network
batch_size = image.shape[0]
img = image.view(batch_size, -1)
n_input = img.shape[1]
n_hidden = 256
n_output = 10

#create the layers of the network
Wh = torch.randn(n_input, n_hidden)
Bh = torch.randn(n_hidden)
Wo = torch.randn(n_hidden, n_output)
Bo = torch.randn(n_output)

#use the network
h = sigmoid(torch.mm(img, Wh) + Bh)
out = torch.mm(h, Wo) + Bo

#apply the softmax function to the output of the network
probabilities = softmax(out)
print(probabilities.sum(dim=1))
#print(probabilities)
#print(probabilities.shape)
