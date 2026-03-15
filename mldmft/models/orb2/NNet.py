import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class FeedforwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(328, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 1024)
        self.fc7 = nn.Linear(1024, 512)
        self.fc8 = nn.Linear(512, 324)

        self.alpha1 = nn.Linear(4, 10)
        self.alpha2= nn.Linear(10,512)
        self.alpha3=nn.Linear(10,1024)

        self.eps1 = nn.Parameter(torch.tensor(0.5))
        self.eps3 = nn.Parameter(torch.tensor(0.5))

        self.eps4 = nn.Parameter(torch.tensor(0.5))
        self.eps6 = nn.Parameter(torch.tensor(0.5))
        
        

    def forward(self, input):
        params= input[:,-4:]
        residual = self.fc1(input)
        x_params = F.gelu(self.alpha1(params))
        x = F.gelu(self.fc1(input))
        x = F.gelu((1-self.eps1)*self.fc2(x)+ self.eps1*(self.alpha3(x_params))) #2048
        x = F.gelu((1-self.eps3)*self.fc4(x)+ self.eps3*(self.alpha2(x_params))) #1024

        x = x + residual

        x = F.gelu((1-self.eps4)*self.fc5(x)+ self.eps4*(self.alpha3(x_params))) #2048
        x = F.gelu((1-self.eps6)*self.fc7(x)+ self.eps6*(self.alpha2(x_params))) #1024

        x = x + residual

        x = self.fc8(x)

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