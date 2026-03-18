import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


   
class FeedforwardNet(nn.Module):
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

def Gfuncloss(outputs, labels, omegas, alpha = 0.0001):
    #criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    #id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    #id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    #if outputs.is_cuda:
    #    id3x3=id3x3.cuda()
    #    id64x64=id64x64.cuda()
    #diff1 = torch.norm(output_real-labels_real)
    #diff2 = torch.norm(output_imag-labels_imag)
    
    return (torch.norm(outputs-labels)) / float(bs)