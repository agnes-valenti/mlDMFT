import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


   
class FeedforwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(121, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 118)

        self.alpha1 = nn.Linear(3, 10)
        self.alpha2= nn.Linear(10,256)
        self.alpha3=nn.Linear(10,512)

        self.eps1 = nn.Parameter(torch.tensor(0.5))
        self.eps2 = nn.Parameter(torch.tensor(0.5))
        self.eps3 = nn.Parameter(torch.tensor(0.5))
        
        

    def forward(self, input):
        params= input[:,-3:]
        x_params = F.relu(self.alpha1(params))
        x = F.relu(self.fc1(input))
        x = F.relu((1-self.eps1)*self.fc2(x)+ self.eps1*(self.alpha2(x_params)))
        x = F.gelu((1-self.eps2)*self.fc3(x)+ self.eps2*(self.alpha3(x_params)))
        x = F.gelu((1-self.eps3)*self.fc4(x)+ self.eps3*(self.alpha3(x_params)))
   

        return self.fc5(x)

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