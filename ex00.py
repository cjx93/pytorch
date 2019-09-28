#FIRST PROJECT IN PYTORCH
import torch
torch.manual_seed(7) # set random seed in order to obtain predictable results
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def activation(x):
    """sigmoid activation function definition
    args:
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))

def single_layer_nn():
    #features are 5 random normal variables
    features = torch.randn(size=(1,5))
    weights = torch.randn_like(features)
    bias = torch.randn(size=(1,1))
    
    #output of a single-layer NN
    y = activation(torch.sum(features *  weights) + bias)
    _y = activation(torch.mm(features, weights.view(-1,1))+bias)#.t())+bias)
    __y = activation((features*weights).sum() + bias)
    print(str(y) + " " + str(_y) + " " + str(__y))

def multi_layer_nn():
    features = torch.randn((1,3))#features are 3 random normal variables
    
    #set dimensions of network
    n_input = features.shape[1]
    n_hidden = 2
    n_output = 1

    W1 = torch.randn((n_input, n_hidden))#first layer of the multilayer perceptron
    W2 = torch.randn((n_hidden, n_output))#output layer of the multilayer perceptron

    B1 = torch.randn((1,n_hidden))
    B2 = torch.randn((1,n_output))

    h = activation(torch.mm(features, W1)+B1)
    output =  activation(torch.mm(h, W2)+B2)
    print(output)


def main():
    single_layer_nn()
    multi_layer_nn()

if __name__ == "__main__":
    print("hello")
    main()

