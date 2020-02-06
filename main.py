import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T
import matplotlib.pyplot as plt

import numpy as np
from utils import conv3x3, flatten, Flatten

from CustomNaiveResNet import CustomNaiveResNet
from ResNet import ResNet, BasicBlock
from DenseNet import DenseNet
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import spatial_groupnorm_forward, spatial_groupnorm_backward
from cs231n.layers import fake_groupnorm_forward, fake_groupnorm_backward
USE_GPU = False

dtype = torch.float32 # we will be using float throughout this tutorial
device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)
NUM_TRAIN = 49000

def random_weight(shape):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:]) # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator. 
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w

def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)


def check_accuracy_part34(loader_test_val, loader_test_train, model, verbose=True):
    val_accuracy = None
    train_accuracy = None
    if loader_test_val.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader_test_val:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = None
            scores = model(x)
            preds = None
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        val_accuracy = acc*100
        if verbose:
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        num_correct = 0
        num_samples = 0
        for x, y in loader_test_train:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = None
            scores = model(x)
            preds = None
            preds = scores.max(1)[1]
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        train_accuracy = acc*100
        if verbose:
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return val_accuracy, train_accuracy


def train_part34(model, optimizer, loader_test_val, loader_test_train, epochs=1, verbose=True):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    losses = list()
    val_accuracies = list()
    train_accuracies = list()
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_test_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            losses.append(loss.item())
            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            print_every = 1
            if t % print_every == 0 and verbose:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                val_accuracy, train_accuracy = check_accuracy_part34(loader_test_val, loader_test_train, model, verbose=verbose)
                val_accuracies.append(val_accuracy)
                train_accuracies.append(train_accuracy)
                print()
    return losses, val_accuracies, train_accuracies
   

class TwelveLayerNet(nn.Module):
    def __init__(self):
        super(TwelveLayerNet, self).__init__()
        num_classes = 10
        in_channel = 3
        channel1 = 32
        channel2 = 32
        channel3 = 64
        channel4 = 128
        channel5 = 256
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channel, channel1, kernel_size=3, stride=1, padding=1, bias=False)#32x32
        self.bn1 = nn.BatchNorm2d(channel1)
        self.conv2 = nn.Conv2d(channel1, channel2, kernel_size=3, stride=1, padding=1, bias=False)#32x32
        self.bn2 = nn.BatchNorm2d(channel2)
        self.conv3 = nn.Conv2d(channel2, channel3, kernel_size=3, stride=2, padding=1, bias=False)#16x16
        self.bn3 = nn.BatchNorm2d(channel3)
        self.conv4 = nn.Conv2d(channel3, channel4, kernel_size=3, stride=2, padding=1, bias=False)#8x8
        self.bn4 = nn.BatchNorm2d(channel4)
        self.conv5 = nn.Conv2d(channel4, channel5, kernel_size=3, stride=2, padding=1, bias=False)#4x4
        self.bn5 = nn.BatchNorm2d(channel5)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        """Forward pass of ResNet."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #########
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        ######
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        #############
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)
        #######
        out = self.maxpool(out)
        out = flatten(out)
        out = self.fc(out)

        return out


def getSequentialTwelveLayerNet():
    num_classes = 10
    in_channel = 3
    channel1 = 32
    channel2 = 32
    channel3 = 64
    channel4 = 128
    channel5 = 256
    model = nn.Sequential(
    nn.Conv2d(in_channel, channel1, (3, 3), padding=1, stride=1),#32x32
    nn.BatchNorm2d(channel1),
    nn.ReLU(),
    nn.Conv2d(channel1, channel2, (3, 3), padding=1),#32x32
    nn.BatchNorm2d(channel2),
    nn.ReLU(),
    nn.Conv2d(channel2, channel3, (3, 3), padding=1, stride=2),#16x16
    nn.BatchNorm2d(channel3),
    nn.ReLU(),
    nn.Conv2d(channel3, channel4, (3, 3), padding=1, stride=2),#8x8
    nn.BatchNorm2d(channel4),
    nn.ReLU(),
    nn.Conv2d(channel4, channel5, (3, 3), padding=1, stride=2),#4x4
    nn.BatchNorm2d(channel5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=4, stride=1),#1x1
    Flatten(),
    nn.Linear(channel5 * 1 * 1, num_classes),
    )
    return model

def train():
    count = 1
    results = {}
    best_val = 0.0
    best_optimizier = None
    best_losses = None
    best_train_accuracies = None
    best_val_accuracies = None
    best_model = None
    for i in range(count):
        scale_r = 0.0
        rate_r = 0.0
        weight_scale = 1e-5
        learning_rate = 3e-3
        cifar10_small_test = dset.CIFAR10('./cs231n/datasets', train=True, download=True, transform=transform)
        loader_small_test = DataLoader(cifar10_small_test, batch_size=2, sampler=sampler.SubsetRandomSampler(range(10)))
        model = None
        #model = ResNet(BasicBlock, [2,2,2,2])
        model = DenseNet()
        optimizer = None
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                            momentum=0.9, nesterov=True, weight_decay=weight_scale)
        losses, val_accuracies, train_accuracies = train_part34(model, optimizer, loader_val, loader_small_test, epochs=10, verbose=True)
        val_acc = val_accuracies[-1]
        train_acc = train_accuracies[-1]
        results[(scale_r, rate_r)] = (weight_scale, learning_rate, val_acc, train_acc)
        if (best_val < val_acc):
            best_val = val_acc
            best_optimizier = optimizer
            best_losses = losses
            best_train_accuracies = train_accuracies
            best_val_accuracies = val_accuracies
            print('###############')

    for item in sorted(results):
        scale_r, rate_r = item
        weight_scale, learning_rate, val_acc, train_acc = results[item]
        print('scale_r: %e, rate_r: %e, ws: %f, lr: %f,  train_acc: %f' % (scale_r, rate_r, weight_scale, learning_rate, train_acc))

    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(best_losses)
    plt.xlabel('Iteration')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(best_train_accuracies, '-o', label='train')
    plt.plot(best_val_accuracies, '-o', label='val')
    plt.plot([0.5] * len(best_val_accuracies), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def main():
    # The torchvision.transforms package provides tools for preprocessing data
    # and for performing data augmentation; here we set up a transform to
    # preprocess the data by subtracting the mean RGB value and dividing by the
    # standard deviation of each RGB value; we've hardcoded the mean and std.
    transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])

    # We set up a Dataset object for each split (train / val / test); Datasets load
    # training examples one at a time, so we wrap each Dataset in a DataLoader which
    # iterates through the Dataset and forms minibatches. We divide the CIFAR-10
    # training set into train and val sets by passing a Sampler object to the
    # DataLoader telling how it should sample from the underlying Dataset.
    # cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
    #                             transform=transform)
    # loader_train = DataLoader(cifar10_train, batch_size=64, 
    #                         sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                            transform=transform)
    loader_val = DataLoader(cifar10_val, batch_size=64, 
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

    # cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True, 
    #                             transform=transform)
    # loader_test = DataLoader(cifar10_test, batch_size=64)
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #train()

    np.random.seed(231)
    N, C, H, W = 2, 6, 4, 5
    G = 2
    D = C // G
    x = 5 * np.random.randn(N, C, H, W) + 12
    gamma = np.random.randn(1,C,1,1)
    beta = np.random.randn(1,C,1,1)
    dout = np.random.randn(N, C, H, W)
    
    # x_reshaped = x.reshape((N*G, H*W*D)).T
    # fn = lambda n: fake_groupnorm_forward(x_reshaped)[0]
    # #cache = {'gamma': gamma}
    # dmean = np.ones((1, N*G))
    # dn_num = eval_numerical_gradient_array(fn, x_reshaped, dmean)
    # mean, cache = fake_groupnorm_forward(x_reshaped)
    # dx_reshaped = fake_groupnorm_backward(dmean, cache)
   
    #You should expect errors of magnitudes between 1e-12~1e-07
    gn_param = {}
    fx = lambda x: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0]
    fg = lambda a: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0]
    fb = lambda b: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0]

    dx_num = eval_numerical_gradient_array(fx, x, dout)
    da_num = eval_numerical_gradient_array(fg, gamma, dout)
    db_num = eval_numerical_gradient_array(fb, beta, dout)

    _, cache = spatial_groupnorm_forward(x, gamma, beta, G, gn_param)
    dx, dgamma, dbeta = spatial_groupnorm_backward(dout, cache)
    print('dx error: ', rel_error(dx_num, dx))
    print('dgamma error: ', rel_error(da_num, dgamma))
    print('dbeta error: ', rel_error(db_num, dbeta))


if __name__ == '__main__':
    main()