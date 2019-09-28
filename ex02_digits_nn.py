import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np 
#import helper
import matplotlib.pyplot as plt 
from collections import OrderedDict
from torchvision import datasets, transforms
torch.manual_seed(7)


def import_data():
    #get and load the dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))
                                 ])

    trainset = datasets.MNIST('MNIST_data/', download=False, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    return trainloader
    #create a dataset iterator, for each batch
    #dataiter = iter(trainloader)
    #image, label = dataiter.next()
    #return image, label

#definition of the NN for the MNIST task
class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
        self.init_vars()

    def init_vars(self):
        self.hidden1.weight.data.normal_(std=0.01)
        self.hidden1.bias.data.fill_(0)
        self.hidden2.weight.data.normal_(std=0.01)
        self.hidden2.bias.data.fill_(0)
    
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.softmax(input=x, dim=1)
        return x



def main():
    trainloader = import_data()
    #images, labels = import_data()
    #imgs = images.view(images.shape[0], -1)
    #print(imgs.shape)
    #model = MNIST()
    #out = model.forward(img)
    #print(out)

    model = nn.Sequential(OrderedDict([
            ('hidden1',nn.Linear(784,128)),
            ('relu1',nn.ReLU()),
            ('hidden2',nn.Linear(128,64)),
            ('relu2',nn.ReLU()),
            ('output',nn.Linear(64,10)),
            ('log_softmax',nn.LogSoftmax(dim=1))
            #('softmax',nn.Softmax(dim=1))
        ]))
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.003)
    #torch.set_grad_enabled(False)

    epochs = 5
    for i in range(epochs):
        running_loss = 0
        #optimizer.zero_grad()
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)

            #Training Step Procedure
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            

            running_loss += loss.item()
        else:
            print(f"Training Loss: {running_loss/len(trainloader)}")
            #optimizer.step()

    #logps = model(imgs)
    #loss = criterion(logps, labels)
    #print(model.hidden1)
    #print(loss)


if __name__ == "__main__":
    main()