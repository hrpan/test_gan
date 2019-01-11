# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 22:32:00 2019

@author: Rick
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

input_dim = 100
class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)

d = D()
g = G()
batch_size = 1024

optim_d = torch.optim.Adam(d.parameters(),lr=1e-3)
optim_g = torch.optim.Adam(g.parameters(),lr=1e-3)

ones = torch.ones(batch_size, 1)
zeros = torch.zeros(batch_size, 1)

rint_low = 0
rint_high = 3


plot_bins = 100
plot_range = (rint_low - 1, rint_high + 1)

linspace = torch.linspace(plot_range[0], plot_range[1], dtype=torch.float32)
linspace = linspace.unsqueeze(1)
z_dist = torch.randn

loss = torch.nn.BCELoss()
for i in range(10000):
    for j in range(1):
        optim_d.zero_grad()
        fake = g(z_dist(batch_size, input_dim))
        real = 0.1 * torch.randn((batch_size, 1)) + torch.randint(rint_low, rint_high, (batch_size, 1),dtype=torch.float32)
        loss_r = loss(d(torch.Tensor(real)), ones)
        loss_f = loss(d(fake), zeros)
        loss_d = loss_r + loss_f 
        loss_d.backward()
        optim_d.step()
    for j in range(1):
        optim_g.zero_grad()
        loss_g = loss(d(g(z_dist(batch_size, input_dim))), ones)
        loss_g.backward()
        optim_g.step()
        _loss = loss_g.data
    if i % 1 == 0:
        plt.ylim(0, 3)
        print(loss_d.data, loss_g.data)
        w = d(linspace).detach().numpy()[:,0]
        test = g(z_dist(batch_size, input_dim)).detach().numpy()
        plt.hist(test,bins=plot_bins,range=plot_range,density=True)
        plt.hist(real.data[:,0],bins=plot_bins,range=plot_range,histtype='step',fill=False,density=True)
        plt.plot(linspace.numpy()[:,0], w)
        plt.pause(1)
        
        plt.cla()