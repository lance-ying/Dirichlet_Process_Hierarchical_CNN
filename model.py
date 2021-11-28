import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


### DCNN model
class CNN(nn.Module):   
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv1d(in_channels=40, out_channels=128, kernel_size=15, bias=False, padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, bias=False, dilation=2, padding=2+2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(p=0.2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

    # Defining the forward pass    
    def forward(self, x):
        h = self.cnn_layers(x)
        h = h.view(h.size(0), -1)
#         print(x.shape)
        h = self.linear_layers(h)

        return h



### NHNN model
class NHNN(nn.Module):   
    def __init__(self,patience=5):
        super(NHNN, self).__init__()

        self.loss=0
        self.epoch=0
        self.patience=patience

        self.cnn_layers = nn.Sequential(
            # Defining a 1D convolution layer
            nn.Conv1d(in_channels=40, out_channels=128, kernel_size=15, bias=False, padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, bias=False, dilation=2, padding=2+2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(p=0.2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

    # Defining the forward pass    
    def forward(self, x):
        h = self.cnn_layers(x)
        h = h.view(h.size(0), -1)
#         print(x.shape)
        h = self.linear_layers(h)
        return h

    def fit(self, optimizer, criterion, train_loader, val_loader):
            loss_train = 0
            loss_valid = 0

            for step in range(1,len(train_loader)+1):
                mfbs, label = next(iter(train_loader))
                mfbs=mfbs.cuda(0)
                label=label.cuda(0)
                optimizer.zero_grad()
                prediction=self.forward(mfbs)
                loss=criterion(prediction, label)
                loss.backward()
                optimizer.step()
                loss_train+=loss.item()
                
                new_train_loss=loss_train/len(train_loader)

            with torch.no_grad():
                for step in range(1,len(val_loader)+1):
                    mfbs, label = next(iter(val_loader))
                    mfbs=mfbs.cuda(0)
                    label=label.cuda(0)
                    prediction=self.forward(mfbs)
                    loss=criterion(prediction, label)
                    loss_valid+=loss.item()
            new_val_loss=loss_valid/len(val_loader)

            print("epoch ", self.epoch, "train_loss=",new_train_loss,"val_loss=",new_val_loss)

            self.loss=new_val_loss

