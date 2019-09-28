import torch
from torch import nn, optim
import torch.nn.functional as F 
from torchvision import datasets, transforms
import numpy as np 
from collections import OrderedDict
import os.path
import matplotlib.pyplot as plt 

torch.manual_seed(7)

class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(784,256)

    def forward(self, x):
        return self.h1(x)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = nn.Sequential(OrderedDict([
        #         ('in', nn.Linear(784, 256)),
        #         ('r1', nn.ReLU()),
        #         ('h2', nn.Linear(256, 128)),
        #         ('r2', nn.ReLU()),
        #         ('h3', nn.Linear(128, 64)),
        #         ('r3', nn.ReLU()),
        #         ('out', nn.Linear(64, 10)),
        #     ]))
        self.h1 = Layer()#nn.Linear(784, 256)
        self.h2 = nn.Linear(256, 128)
        self.h3 = nn.Linear(128, 64)
        self.fl = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        #out = self.model(x)
        x = self.dropout(F.relu(self.h1(x)))
        x = self.dropout(F.relu(self.h2(x)))
        x = self.dropout(F.relu(self.h3(x)))
        logps = F.log_softmax(self.fl(x), dim=1)
        return logps


def save_model(model, epoch, optimizer, filename):
    state_dict = model.state_dict()
    opt_state_dict = optimizer.state_dict()
    checkpoint = {
        'epoch':epoch ,
        'optimizer':opt_state_dict,
        'state_dict':state_dict
    }
    torch.save(checkpoint, filename)

def load_model(path):
    model = Classifier()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    epoch = 0
    if os.path.exists(path):
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']+1#start from the successive epoch of the saved checkpoint
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, epoch



def get_data(batch_size=64, train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    data = datasets.FashionMNIST('fashion_data/', download=False, train=train, transform=transform)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader


def main():
    train = get_data(train=True)
    valid = get_data(train=False)

    epochs = 30
    model, optimizer, epoch = load_model('checkpoint.pth')
    criterion = nn.NLLLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.003)

    train_losses, test_losses = [], []

    print(model)
    print(model.state_dict())

    for e in range(epoch, epochs):
        total_loss = 0.0
        for images, labels in train:
            logps = model(images)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        else:
            print(f'EPOCH: {e} @ Training Loss: {total_loss/len(train)}')
            with torch.no_grad():
                corrects = 0
                test_loss = 0.0

                #set the model in inference mode (set dropout probability to 0)
                model.eval()#;print('test!')

                for images, labels in valid:
                    out = model(images)
                    loss = criterion(out, labels)
                    test_loss += loss.item()

                    ps = torch.exp(out)
                    tvalues, tclasses = ps.topk(k=1, dim=1)
                    equals = tclasses == labels.view(*tclasses.shape)
                    corrects += equals.sum()
                else:
                    print(f'>Test Loss: {test_loss/len(valid)}')
                    print(f'>Test Accuracy: {float(corrects)/len(valid.dataset)*100}')
                    #print(f'>data batches: {len(valid)} ** data points: {len(valid.dataset)}')

                #set the model in training model
                save_model(model, e, optimizer, 'checkpoint.pth')

            model.train()#;print('train!')

        train_losses.append(total_loss/len(train))
        test_losses.append(test_loss/len(valid))
        print('done\n___________________________\n')
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend(frameon=False)
    plt.show()


if __name__ == "__main__":
    main()
