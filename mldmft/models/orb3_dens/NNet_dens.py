import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


   
class FeedforwardNet_dens(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(435, 1024)
        self.resblock1 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )
        self.resblock2 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )
        self.fc2 = nn.Linear(1024, 432)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        residual = x
        x = self.resblock1(x) + residual  # First residual connection
        residual = x
        x = self.resblock2(x) + residual  # Second residual connection
        x = self.fc2(x)
        return x

class NetProcess_dens(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(432, 256)
        self.fc2 = nn.Linear(256, 56)
        self.fc3 = nn.Linear(56, 3)
        #self.fc4 = nn.Linear(512, 512)
        #self.fc5 = nn.Linear(512, 256)
        #self.fc6 = nn.Linear(256, 128)
        #self.fc7 = nn.Linear(128, 128)
        #self.fc8 = nn.Linear(128, 108)
        
        

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

def Gfuncloss_dens(outputs, labels, taulen, alpha = 0.0001):
    #criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)   
    return (torch.norm(-2*outputs[:,12*(taulen-1) + 0]-labels[:,0]) + torch.norm(-2*outputs[:,12*(taulen-1) + 3]-labels[:,1]) + torch.norm(-2*outputs[:,12*(taulen-1) + 5]-labels[:,2])) / float(bs)

def Gfuncloss_dens_processed(outputs, labels, taulen, alpha = 0.0001):
    #criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)   
    #print('outputs shape in loss: ', outputs.shape)
    #print('labels shape in loss: ', labels.shape)
    return (torch.norm(outputs-labels)) / float(bs)

def Gfuncloss_G(outputs, labels, alpha = 0.0001):
    #criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    
    return (torch.norm(outputs-labels)) / float(bs)
