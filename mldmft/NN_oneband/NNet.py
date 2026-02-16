import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


   
class FeedforwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(74, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 72)
        
        

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = F.gelu(self.fc4(x))
        x = F.gelu(self.fc5(x))

        return self.fc6(x)

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