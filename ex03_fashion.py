import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np 
from collections import OrderedDict

torch.manual_seed(7)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data(batch_size=1, train=True):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])
    trainset = datasets.FashionMNIST('fashion_data/', download=False, train=train, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader

def get_model(input_dim=784, classes=10):
    model = nn.Sequential(OrderedDict([
            ("h1",nn.Linear(input_dim,256)),
            ("relu1",nn.ReLU()),
            ("h2",nn.Linear(256,128)),
            ("relu2",nn.ReLU()),
            ("h3",nn.Linear(128,64)),
            ("relu3",nn.ReLU()),
            ("out",nn.Linear(64,classes))
        ]))
    return model

def get_CNN_model():
    pass


def main():
    train_data = get_data(batch_size=64, train=True)
    test_data = get_data(batch_size=64, train=False)
    
    epochs = 6
    model = get_model()#.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    for e in range(epochs):
        total_loss = 0.0
        for images, labels in train_data:
            images = images.view(images.shape[0], -1)
            #images = images.to(device)
            #labels = images.to(device)

            optimizer.zero_grad()
            outs = model(images)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        else:
            print(f"EPOCH: {e} || Loss: {total_loss/len(train_data)}")


    total_corrects = 0.0
    for images, labels in test_data:
        images = images.view(images.shape[0], -1)
        #images = images.to(device)
        #labels = images.to(device)
        with torch.no_grad():
            probs = F.softmax(model(images), dim=1)
            preds = probs.argmax(dim=1)
        corrects = (preds == labels)
        total_corrects += corrects.sum().float()
    else:
        #print(str(total_corrects)+"#"+str(len(test_data)))
        print(f"Test Accuracy: {total_corrects/float(len(test_data.dataset))}")



if __name__ == "__main__":
    main()
