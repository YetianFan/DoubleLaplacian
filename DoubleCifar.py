####################################################################
##### this code is designed for double Laplcian regularization #####
##### fully connected neural networks #####
##### date: 2023-02-01 #####

import time
import torch
import ssl
import math
import csv
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import sys
import numpy as np

def TenPow(vector):
    answer = []
    for i in vector:
        answer.append(math.pow(10,i))
    return answer
    
ssl._create_default_https_context = ssl._create_unverified_context
sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### input the parameters ####
batch_size = 64            ### the batch size
lamb1 = np.arange(-8,-3)   ### the tunning parameters for attraction Laplacian
lamb1 = TenPow(lamb1)
lamb1.insert(0,0)
lamb2 = np.arange(-17,-12)  ### the tunning parameters for repulsion Laplacian
lamb2 = TenPow(lamb2)
lamb2.insert(0,0)
lr, num_epochs = 0.00003, 100  ### the learning rate, the number of iteration
Accuracy = [[] for i in range(len(lamb1)*len(lamb2))]  ### record the accuracy

train_dataset = datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=True)
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#### the structure of neural networks
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 3, 1,1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(96, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )


    def forward(self, x):
        outputs = []
        for name, module in self.conv.named_children():
            x = module(x)
        x=x.view(x.shape[0],-1)
        for name, module in self.fc.named_children():
            x = module(x)
            if name in ["0","2","4"]:
                outputs.append(x)
        return outputs

#### generate the traning data and calculate the Laplacian matrix
def data_L():
    x1 = []
    y1 = []
    L1 = []
    L2 = []
    for x11, y11 in tqdm(train_iter):
        x11 = x11.to(device)
        y11 = y11.to(device)
        x1.append(x11)
        y1.append(y11)
        x11 = x11.view(x11.shape[0], -1)
        w = torch.randn(x11.shape[0], x11.shape[0])*0.000001
        w = w.to(device)
        sigma = 0
        for i in range(x11.shape[0]):
            s = x11[i] ** 2
            s = s.sum()
            sigma += s
        sigma = sigma / x11.shape[0]

        for i in range(x11.shape[0]):
            for j in range(x11.shape[0]):
                q = x11[i] - x11[j]
                q = q ** 2
                q = -q.sum()
                q = q / sigma
                w[i][j] = torch.exp(q)

        w1 = w.clone()
        w2 = w.clone()
        for i in range(x11.shape[0]):
            for j in range(x11.shape[0]):
                if y11[i] != y11[j]:
                    w1[i][j] = 0
                else:
                    if i != j:
                        w2[i][j] = 0

        k1 = []
        for i in range(x11.shape[0]):
            k1.append(w1[i].sum() - 1)

        k2 = []
        for i in range(x11.shape[0]):
            k2.append(w2[i].sum() - 1)

        for i in range(x11.shape[0]):
            for j in range(x11.shape[0]):
                if i != j:
                    w1[i][j] = -w1[i][j]
                    d11 = torch.pow(k1[i], 1 / 2)
                    d12 = torch.pow(k1[j], 1 / 2)
                    d1112 = d11 * d12
                    if w1[i][j] != 0:
                        w1[i][j] = w1[i][j] / d1112
                    w2[i][j] = -w2[i][j]
                    d21 = torch.pow(k2[i], 1 / 2)
                    d22 = torch.pow(k2[j], 1 / 2)
                    d2122 = d21 * d22
                    if w2[i][j] != 0:
                        w2[i][j] = w2[i][j] / d2122
                else:
                    w1[i][j] = 1
                    w2[i][j] = 1
        L1.append(w1)
        L2.append(w2)
    return x1,y1,L1,L2

def evaluate_accuracy(data_iter, net, device=None):

    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() 
                acc_sum += (net(X.to(device))[2].argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() 
            else: 
                if('is_training' in net.__code__.co_varnames): 
                    acc_sum += (net(X, is_training=False)[2].argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X)[2].argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def net_orignal(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        tt = math.ceil(50000/batch_size)
        for ni in range(tt):
            out_put = net(X[ni])
            l = loss(out_put[2], y[ni])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (out_put[2].argmax(dim=1) == y[ni]).sum().cpu().item()
            n += y[ni].shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        Accuracy[0].append(test_acc)
        print('epoch %d, loss %.4f, train acc %.4f, test acc %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

def net_laplacian(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        tt = math.ceil(50000/batch_size)
        for ni in range(tt):
            out_put = net(X[ni])
            z = []
            for feature_map in out_put[0:2]:
                im = feature_map.view(-1, X[ni].shape[0])
                imT = im.transpose(0, 1)
                Q = torch.mm(torch.mm(im,L[ni]),imT)
                sum = 0
                for i in range(Q.shape[0]):
                    sum += Q[i][i]
                z.append(sum)

            l = loss(out_put[2], y[ni])
            l = l + z[0] + z[1]
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (out_put[2].argmax(dim=1) == y[ni]).sum().cpu().item()
            n += y[ni].shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        Accuracy[lambda1ID*(len(lamb2))+lambda2ID].append(test_acc)
        print('epoch %d, loss %.4f, train acc %.4f, test acc %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

X,y,L1,L2 = data_L()
for lambda1ID in range(len(lamb1)):
    for lambda2ID in range(len(lamb2)):
        if (lambda1ID+lambda2ID) == 0:
            net = AlexNet()
            path = 'Double+Cifar.pth'
            torch.save(net.state_dict(), path)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            net_orignal(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
        else:
            lambda1 = lamb1[lambda1ID]
            lambda2 = lamb2[lambda2ID]
            net = AlexNet()
            path = 'Double+Cifar.pth'
            net.load_state_dict(torch.load(path))
            L = []
            for i in range(len(L1)):
                L.append(lambda1 * L1[i] - lambda2 * L2[i])
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            net_laplacian(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

with open('results.csv','w',newline='') as f:
    writer = csv.writer(f)
    for i in Accuracy:
        writer.writerow(i)
    f.close()


