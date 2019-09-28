import torch
from torch import nn, optim
import torch.nn.functional as F 
from torchvision import datasets, transforms
import numpy as np 
from collections import OrderedDict
import os.path
import matplotlib.pyplot as plt 

torch.manual_seed(6)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(3*224*224, 2048)#the image tensor is 3x224x224, since it has 3 color channels. so the input layer has 3*224*224 input size
        self.h2 = nn.Linear(2048, 512)
        self.h3 = nn.Linear(512, 128)
        self.fl = nn.Linear(128, 2)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.h1(x)))
        x = self.dropout(F.relu(self.h2(x)))
        x = self.dropout(F.relu(self.h3(x)))
        logps = F.log_softmax(self.fl(x), dim=1)
        return logps


def get_data(datadir, train=True):
    if train:
        datadir = datadir+'/train'
        transform = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
    else:
        datadir = datadir+'/test'
        transform = transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
    data = datasets.ImageFolder(datadir, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
    return loader

def save_model(model, epoch, optimizer, ckptfile):
    state_dict = model.state_dict()
    opt_state_dict = optimizer.state_dict()
    checkpoint = {
        'epoch':epoch,
        'optimizer':opt_state_dict,
        'state_dict':state_dict
    }
    torch.save(checkpoint, ckptfile)

def load_model(ckptfile):
    model = Classifier()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    epoch = 0
    if os.path.exists(ckptfile):
        checkpoint = torch.load(ckptfile)
        epoch = checkpoint['epoch'] + 1
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, epoch

def train_model(datapath, epochs, ckptfile):
    train_data = get_data(datapath, train=True)
    valid_data = get_data(datapath, train=False)

    model,optimizer, epoch = load_model(ckptfile)
    criterion = nn.NLLLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.003)

    train_losses, valid_losses = [], []
    for e in range(epoch, epochs):
        total_loss = 0.0
        for images, labels in train_data:
            out = model(images)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        else:
            print(f'Epoch {e+1}, AVG Train Loss {total_loss/len(train_data)}')
            save_model(model, e, optimizer, ckptfile)
            with torch.no_grad():
                model.eval()
                total_correct = 0
                losses = 0.0
                for images, labels in valid_data:
                    out = model(images)
                    loss = criterion(out, labels)
                    losses += loss.item()
                    
                    probs = torch.exp(out)
                    tprobs, tclasses = probs.topk(k=1, dim=1)
                    corrects = tclasses == labels.view(*tclasses.shape)
                    total_correct += corrects.sum()

                else:
                    print(f'>AVG Valid Loss {losses/len(valid_data)}')
                    print(f'>Valid Accuracy {float(total_correct)/len(valid_data.dataset)*100}')

            train_losses.append(total_loss/len(train_data))
            valid_losses.append(losses/len(valid_data))
            model.train()

    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.legend(frameon=False)
    plt.show()


def main():
    train_model(datapath='Cat_Dog_data', epochs=30, ckptfile='ckpt.pth')


if __name__ == '__main__':
    main()


