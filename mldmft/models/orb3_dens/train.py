import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from NNet_dens import NetProcess_dens, Gfuncloss_dens_processed, Gfuncloss_G
import matplotlib.pyplot as plt
import sys 

#import plotly.graph_objects as go
#import plotly.express as px


#NsamplesToUse = 3000
#NsamplesToUseDmft1 = 400
#NsamplesToUseTest = 100
#NsamplesToUseDmft1Test = 100

folder ='/mnt/home/avalenti/ceph/ml-dmft/train_model_3orbs_B3/'
sys.path.append(folder)
from NNet import FeedforwardNet, Gfuncloss
import argparse

import matplotlib.pyplot as plt
#from triqs.plot.mpl_interface import oplot,plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

feedforwardnet = FeedforwardNet()
feedforwardnet.to(device); 
feedforwardnet.load_state_dict(torch.load(folder+"save_9500.pth", map_location=device, weights_only=True)) #3800

net_process = NetProcess_dens()
net_process.to(device)
# Print the parameters of the feedforward network
y=0
for name, param in feedforwardnet.named_parameters():
    if param.requires_grad:
        y+=param.numel()
        #print(f"Parameter: {name}, Shape: {param.shape}")
        #param.data.zero_()

print("Number of trainable parameters in FeedforwardNet: ", y)
#sys.exit()

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)

def get_data(folder1, nsamples_to_use_train=-1, nsamples_to_use_test=-1, flag_test=True, ind_up=0):
    taus = np.load(os.path.join(folder1, "tau", "tauvals.npy"))
    print("num taus: ",np.shape(taus)[0])

    train_data_real = np.load(os.path.join(folder1, "tau", "Delta_tau_real_train.npy"))[:nsamples_to_use_train]
    print("train_data_real shape: ", train_data_real.shape)
    train_data_imag = np.load(os.path.join(folder1, "tau", "Delta_tau_imag_train.npy"))[:nsamples_to_use_train]
    train_data = np.zeros((train_data_real.shape[0], train_data_real.shape[1], 6*2))
    train_data[:, :, 0] = train_data_real[:,:,ind_up,0,0]
    train_data[:, :, 1] = train_data_real[:,:,ind_up,0,1]
    train_data[:, :, 2] = train_data_real[:,:,ind_up,0,2]
    train_data[:,:,3] = train_data_real[:,:,ind_up,1,1]
    train_data[:,:,4] = train_data_real[:,:,ind_up,1,2]
    train_data[:,:,5] = train_data_real[:,:,ind_up,2,2]
    train_data[:,:,6] = train_data_imag[:,:,ind_up,0,0]
    train_data[:,:,7] = train_data_imag[:,:,ind_up,0,1]
    train_data[:,:,8] = train_data_imag[:,:,ind_up,0,2]
    train_data[:,:,9] = train_data_imag[:,:,ind_up,1,1]
    train_data[:,:,10] = train_data_imag[:,:,ind_up,1,2]
    train_data[:,:,11] = train_data_imag[:,:,ind_up,2,2]
    
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
    train_data_1 = np.zeros((np.shape(train_data)[0], np.shape(train_data)[1]+3))
    train_data_1[:, :-3] = train_data
    Umuvals = None
    
    Umuvals = np.load(os.path.join(folder1, "Umu_train.npy"))[:train_data.shape[0]]
    print("Umuvals shape: ", Umuvals.shape)
    train_data_1[:, -3] = Umuvals[:,0]
    train_data_1[:, -2] = Umuvals[:,1]/Umuvals[:,0]
    train_data_1[:, -1] = Umuvals[:,2] #Umuvals[:,0]

    if flag_test:
        test_data_real = np.load(os.path.join(folder1, "tau", "Delta_tau_real_test.npy"))[:nsamples_to_use_test]
        test_data_imag = np.load(os.path.join(folder1, "tau", "Delta_tau_imag_test.npy"))[:nsamples_to_use_test]
        test_data = np.zeros((test_data_real.shape[0], test_data_real.shape[1], 6*2))
        test_data[:, :, 0] = test_data_real[:,:,ind_up,0,0]
        test_data[:, :, 1] = test_data_real[:,:,ind_up,0,1]
        test_data[:, :, 2] = test_data_real[:,:,ind_up,0,2]
        test_data[:,:,3] = test_data_real[:,:,ind_up,1,1]
        test_data[:,:,4] = test_data_real[:,:,ind_up,1,2]
        test_data[:,:,5] = test_data_real[:,:,ind_up,2,2]
        test_data[:,:,6] = test_data_imag[:,:,ind_up,0,0]
        test_data[:,:,7] = test_data_imag[:,:,ind_up,0,1]
        test_data[:,:,8] = test_data_imag[:,:,ind_up,0,2]
        test_data[:,:,9] = test_data_imag[:,:,ind_up,1,1]
        test_data[:,:,10] = test_data_imag[:,:,ind_up,1,2]
        test_data[:,:,11] = test_data_imag[:,:,ind_up,2,2] 
        
        test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))
        test_data_1 = np.zeros((np.shape(test_data)[0], np.shape(test_data)[1]+3))
        test_data_1[:, :-3] = test_data

        Umuvals = np.load(os.path.join(folder1, "Umu_test.npy"))[:test_data.shape[0]]
        #Umuvals = np.load(os.path.join(folder1, "Umuvals.npy"))[:nsamples_to_use]
        test_data_1[:, -3] = Umuvals[:,0]
        test_data_1[:, -2] = Umuvals[:,1]/Umuvals[:,0]
        test_data_1[:, -1] = Umuvals[:,2] #Umuvals[:,0]

    train_labels_real = np.load(os.path.join(folder1, "tau", "G_tau_real_train.npy"))[:nsamples_to_use_train]
    train_labels_imag = np.load(os.path.join(folder1, "tau", "G_tau_imag_train.npy"))[:nsamples_to_use_train]
    train_labels = np.zeros((train_labels_real.shape[0], train_labels_real.shape[1], 6*2))
    train_labels[:,:,0] = train_labels_real[:,:,ind_up,0,0]
    train_labels[:,:,1] = train_labels_real[:,:,ind_up,0,1]
    train_labels[:,:,2] = train_labels_real[:,:,ind_up,0,2]
    train_labels[:,:,3] = train_labels_real[:,:,ind_up,1,1]
    train_labels[:,:,4] = train_labels_real[:,:,ind_up,1,2]
    train_labels[:,:,5] = train_labels_real[:,:,ind_up,2,2]            
    train_labels[:,:,6] = train_labels_imag[:,:,ind_up,0,0]
    train_labels[:,:,7] = train_labels_imag[:,:,ind_up,0,1]
    train_labels[:,:,8] = train_labels_imag[:,:,ind_up,0,2]
    train_labels[:,:,9] = train_labels_imag[:,:,ind_up,1,1]
    train_labels[:,:,10] = train_labels_imag[:,:,ind_up,1,2]
    train_labels[:,:,11] = train_labels_imag[:,:,ind_up,2,2]

    train_labels = np.reshape(train_labels, (train_labels.shape[0], train_labels.shape[1]*train_labels.shape[2]))

    if flag_test:
        test_labels_real = np.load(os.path.join(folder1, "tau", "G_tau_real_test.npy"))[:nsamples_to_use_test]
        test_labels_imag = np.load(os.path.join(folder1, "tau", "G_tau_imag_test.npy"))[:nsamples_to_use_test]
        test_labels = np.zeros((test_labels_real.shape[0], test_labels_real.shape[1], 6*2))
        test_labels[:,:,0] = test_labels_real[:,:,ind_up,0,0]
        test_labels[:,:,1] = test_labels_real[:,:,ind_up,0,1]
        test_labels[:,:,2] = test_labels_real[:,:,ind_up,0,2]      
        test_labels[:,:,3] = test_labels_real[:,:,ind_up,1,1]
        test_labels[:,:,4] = test_labels_real[:,:,ind_up,1,2]
        test_labels[:,:,5] = test_labels_real[:,:,ind_up,2,2]
        test_labels[:,:,6] = test_labels_imag[:,:,ind_up,0,0]
        test_labels[:,:,7] = test_labels_imag[:,:,ind_up,0,1]
        test_labels[:,:,8] = test_labels_imag[:,:,ind_up,0,2]
        test_labels[:,:,9] = test_labels_imag[:,:,ind_up,1,1]
        test_labels[:,:,10] = test_labels_imag[:,:,ind_up,1,2]
        test_labels[:,:,11] = test_labels_imag[:,:,ind_up,2,2]
        
        test_labels = np.reshape(test_labels, (test_labels.shape[0], test_labels.shape[1]*test_labels.shape[2]))
        #test_labels1_mean = np.mean(train_labels1, axis=0)
        #test_labels1_mean = np.broadcast_to(test_labels1_mean, (np.shape(test_labels1)[0], np.shape(test_labels1)[1]))
        
        return train_data_1, train_labels, test_data_1, test_labels, taus
    return train_data_1, train_labels, taus  


#folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2_U10/'
#train_dataU10, train_labelsU10, test_dataU10, test_labelsU10, test_labels_meanU10, tausU10 = get_data(folder, getU=False, Uval=10.0)

#folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2/'
#train_dataU1, train_labelsU1, test_dataU1, test_labelsU1, test_labels_meanU1, tausU1 = get_data(folder, getU=False, Uval=1.0)



def get_data_dens(folder_root, folder_root_ina, folder1, nsamples_to_use_train=-1, nsamples_to_use_test=-1, flag_test=True, ind_up=0):
    taus = np.load(os.path.join(folder_root_ina, folder1, "tauvals.npy"))
    print("num taus: ",np.shape(taus)[0])

    train_data_real = np.load(os.path.join(folder_root, folder1, "tau", "Delta_tau_real_train.npy"))[:nsamples_to_use_train]
    print("train_data_real shape: ", train_data_real.shape)
    train_data_imag = np.load(os.path.join(folder_root, folder1, "tau", "Delta_tau_imag_train.npy"))[:nsamples_to_use_train]
    train_data = np.zeros((train_data_real.shape[0], train_data_real.shape[1], 6*2))
    train_data[:, :, 0] = train_data_real[:,:,ind_up,0,0]
    train_data[:, :, 1] = train_data_real[:,:,ind_up,0,1]
    train_data[:, :, 2] = train_data_real[:,:,ind_up,0,2]
    train_data[:,:,3] = train_data_real[:,:,ind_up,1,1]
    train_data[:,:,4] = train_data_real[:,:,ind_up,1,2]
    train_data[:,:,5] = train_data_real[:,:,ind_up,2,2]
    train_data[:,:,6] = train_data_imag[:,:,ind_up,0,0]
    train_data[:,:,7] = train_data_imag[:,:,ind_up,0,1]
    train_data[:,:,8] = train_data_imag[:,:,ind_up,0,2]
    train_data[:,:,9] = train_data_imag[:,:,ind_up,1,1]
    train_data[:,:,10] = train_data_imag[:,:,ind_up,1,2]
    train_data[:,:,11] = train_data_imag[:,:,ind_up,2,2]
    
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
    train_data_1 = np.zeros((np.shape(train_data)[0], np.shape(train_data)[1]+3))
    train_data_1[:, :-3] = train_data
    Umuvals = None
    
    Umuvals = np.load(os.path.join(folder_root, folder1, "Umu_train.npy"))[:train_data.shape[0]]
    print("Umuvals shape: ", Umuvals.shape)
    train_data_1[:, -3] = Umuvals[:,0]
    train_data_1[:, -2] = Umuvals[:,1]/Umuvals[:,0]
    train_data_1[:, -1] = Umuvals[:,2] #Umuvals[:,0]

    if flag_test:
        test_data_real = np.load(os.path.join(folder_root, folder1, "tau", "Delta_tau_real_test.npy"))[:nsamples_to_use_test]
        test_data_imag = np.load(os.path.join(folder_root, folder1, "tau", "Delta_tau_imag_test.npy"))[:nsamples_to_use_test]
        test_data = np.zeros((test_data_real.shape[0], test_data_real.shape[1], 6*2))
        test_data[:, :, 0] = test_data_real[:,:,ind_up,0,0]
        test_data[:, :, 1] = test_data_real[:,:,ind_up,0,1]
        test_data[:, :, 2] = test_data_real[:,:,ind_up,0,2]
        test_data[:,:,3] = test_data_real[:,:,ind_up,1,1]
        test_data[:,:,4] = test_data_real[:,:,ind_up,1,2]
        test_data[:,:,5] = test_data_real[:,:,ind_up,2,2]
        test_data[:,:,6] = test_data_imag[:,:,ind_up,0,0]
        test_data[:,:,7] = test_data_imag[:,:,ind_up,0,1]
        test_data[:,:,8] = test_data_imag[:,:,ind_up,0,2]
        test_data[:,:,9] = test_data_imag[:,:,ind_up,1,1]
        test_data[:,:,10] = test_data_imag[:,:,ind_up,1,2]
        test_data[:,:,11] = test_data_imag[:,:,ind_up,2,2] 
        
        test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))
        test_data_1 = np.zeros((np.shape(test_data)[0], np.shape(test_data)[1]+3))
        test_data_1[:, :-3] = test_data

        Umuvals = np.load(os.path.join(folder_root, folder1, "Umu_test.npy"))[:test_data.shape[0]]
        #Umuvals = np.load(os.path.join(folder1, "Umuvals.npy"))[:nsamples_to_use]
        test_data_1[:, -3] = Umuvals[:,0]
        test_data_1[:, -2] = Umuvals[:,1]/Umuvals[:,0]
        test_data_1[:, -1] = Umuvals[:,2] #Umuvals[:,0]

    train_labels_real = np.load(os.path.join(folder_root_ina, folder1, "n_from_Gtau_3den_train.npy"))[:nsamples_to_use_train]
    #train_labels_imag = np.load(os.path.join(folder1, "tau", "G_tau_imag_train.npy"))[:nsamples_to_use_train]
    train_labels = np.zeros((train_labels_real.shape[0], 3))
    train_labels = np.copy(train_labels_real)
    #print(np.shape(train_labels))

    #train_labels = np.reshape(train_labels, (train_labels.shape[0], train_labels.shape[1]*train_labels.shape[2]))

    if flag_test:
        test_labels_real = np.load(os.path.join(folder_root_ina, folder1, "n_from_Gtau_3den_test.npy"))[:nsamples_to_use_test]
        test_labels = np.zeros((test_labels_real.shape[0], 3))
        test_labels = np.copy(test_labels_real)
        #test_labels_imag = np.load(os.path.join(folder1, "tau", "G_tau_imag_test.npy"))[:nsamples_to_use_test]
        
        return train_data_1, train_labels, test_data_1, test_labels, taus
    return train_data_1, train_labels, taus  


#folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2_U10/'
#train_dataU10, train_labelsU10, test_dataU10, test_labelsU10, test_labels_meanU10, tausU10 = get_data(folder, getU=False, Uval=10.0)

#folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2/'
#train_dataU1, train_labelsU1, test_dataU1, test_labelsU1, test_labels_meanU1, tausU1 = get_data(folder, getU=False, Uval=1.0)



#folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJhund/'
#train_datamixedU, train_labelsmixedU, test_datamixedU, test_labelsmixedU, taus = get_data(folder_root, folder_root_ina, folder) #, nsamples_to_use_test=2, nsamples_to_use_train=2)
#print("train_data shape: ", np.shape(train_datamixedU))
#print("test_data shape: ", np.shape(test_datamixedU))

folder_root = '/mnt/home/avalenti/ceph/ml-dmft/'
folder_root_ina = '/mnt/ceph/users/ipark/ML/n_solver/3den/training_dataset/'
folders =[
'traindata_3orbs_offdiag_beta10_mixedUmuJhund/',
'traindata_3orbs_offdiag_beta10_mixedUmuJhund_dmft1/',
'traindata_3orbs_offdiag_beta10_mixedUmuJhund_dmft2/',  
'traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm/',
'traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_dmft1/', 
'traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_halffilled/',
'traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_halffilled_dmft1/',
'traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_halffilled_smallscale/',
'traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_halffilled_smallscale_dmft1/',
'traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_largemurange/',
'traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_largemurange_dmft1/',
'traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_largemurange_smallscale/',
'traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_largemurange_smallscale_dmft1/',
'traindata_3orbs_offdiag_beta10_mixedUmuJ_largemurange_smallscale/',
'traindata_3orbs_offdiag_beta10_mixedUmuJ_largemurange_smallscale_dmft1/']

train_datamixedU = None
train_labelsmixedU = None 
test_datamixedU = None
test_labelsmixedU = None
train_labelsmixedU_dens = None
test_labelsmixedU_dens = None
taus = None  
for folder in folders:
    print("Loading data from folder: ", folder)
    if train_datamixedU is None:
       train_datamixedU, train_labelsmixedU, test_datamixedU, test_labelsmixedU, taus = get_data(folder_root + folder) 
       t1, train_labelsmixedU_dens, _, test_labelsmixedU_dens, _ = get_data_dens(folder_root, folder_root_ina, folder)
       print(t1[:5]-train_datamixedU[:5])
       #abort()
    else:
        train_datadmft1, train_labelsdmft1, test_datadmft1, test_labeldmft1, taus = get_data(folder_root + folder) 
        _, train_labelsdmft1_dens, _, test_labeldmft1_dens, _ = get_data_dens(folder_root, folder_root_ina, folder)
        train_datamixedU = np.concatenate((train_datamixedU, train_datadmft1), axis=0) 
        train_labelsmixedU = np.concatenate((train_labelsmixedU, train_labelsdmft1), axis=0)
        test_datamixedU = np.concatenate((test_datamixedU, test_datadmft1), axis=0)
        test_labelsmixedU = np.concatenate((test_labelsmixedU, test_labeldmft1), axis=0)
        train_labelsmixedU_dens = np.concatenate((train_labelsmixedU_dens, train_labelsdmft1_dens), axis=0)
        test_labelsmixedU_dens = np.concatenate((test_labelsmixedU_dens, test_labeldmft1_dens), axis=0)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

#feedforwardnet = FeedforwardNet()
#feedforwardnet.to(device);

optimizer = torch.optim.Adam(net_process.parameters(), lr=0.0002, weight_decay=1e-5)
taus_tensor = torch.tensor(taus).to(device).float()
taulen = len(taus)
def train(model, train_loader, val_loader=None, val_train_loader=None, numepochs=1200, epochs_start=0, save=True, saveexamples=True):
    test_loss_vals = []
    train_loss_vals = []
    train_loss_its = []
    test_loss_its = []
    train_loss_vals_G = []
    train_loss_vals_dens = []
    test_loss_vals_dens = []
    test_loss_vals_G = []

    #test_loss_compare_vals = []
    for epoch in range(epochs_start, epochs_start+numepochs): 
        feedforwardnet.train()
        running_loss = 0.0
        running_loss_G = 0.0
        running_loss_dens = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels_G, labels_dens = data['DeltaHyb'].to(device).float(), data['Gfunc'].to(device), data['Dens'].to(device) #, data['compareoutput'].to(device)
            optimizer.zero_grad()
            # Freeze weights in feedforwardnet
            #for param in feedforwardnet.parameters():
            #    param.requires_grad = False

            outputs_G = feedforwardnet(inputs)
            outputs_dens = net_process(outputs_G)

            loss_G = Gfuncloss_G(outputs_G, labels_G)
            loss_dens = Gfuncloss_dens_processed(outputs_dens, labels_dens, taulen)

            loss = loss_dens
            loss.backward()

            # Only update weights in net_process
            #optimizer = torch.optim.Adam(net_process.parameters(), lr=0.0002, weight_decay=1e-5)
            optimizer.step()

            # print statistics
            #print('epoch, batch: ', epoch, i, loss, loss.item())
            running_loss += loss.item()
            running_loss_G += loss_G.item()
            running_loss_dens += loss_dens.item()
            if i % 200 == 0:    # print every 100 mini-batches
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.5f, loss_G: %.5f, loss_dens: %.5f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10, running_loss_G / 10, running_loss_dens / 10))
                    train_loss_vals.append(running_loss / 10)
                    train_loss_vals_G.append(running_loss_G / 10)
                    train_loss_vals_dens.append(running_loss_dens / 10)
                    train_loss_its.append(i+epoch*len(train_loader))
                    running_loss = 0.0
                    running_loss_G = 0.0
                    running_loss_dens = 0.0
                    np.save('trainloss.npy', np.array(train_loss_vals))
                    np.save('trainloss_G.npy', np.array(train_loss_vals_G))
                    np.save('trainloss_dens.npy', np.array(train_loss_vals_dens))
                    np.save('train_its.npy', np.array(train_loss_its))

             

        feedforwardnet.eval()
        correct = total = 0

        # validation
        if val_loader and epoch % 100 == 0:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels_G, labels_dens= data['DeltaHyb'].to(device).float(), data['Gfunc'].to(device), data['Dens'].to(device) #, data['compareoutput'].to(device)
                    outputs_G = feedforwardnet(inputs)
                    outputs_dens = net_process(outputs_G)
                    
                    test_loss_G = Gfuncloss_G(outputs_G, labels_G) #/len(val_loader)
                    test_loss_dens = Gfuncloss_dens_processed(outputs_dens, labels_dens, taulen) #/len(val_loader)
                    test_loss = test_loss_G/432.0 + test_loss_dens
                                       #test_loss_compare = Gfuncloss(comparelabels, labels, taus)
                    print('Valid loss: %.3f' % test_loss.item())
                    test_loss_vals_dens.append(test_loss_dens.item())
                    test_loss_vals_G.append(test_loss_G.item())
                    test_loss_vals.append(test_loss.item())
                    #test_loss_compare_vals.append(test_loss_compare.item())
                    test_loss_its.append(epoch*len(train_loader) + i)
                    #plt.figure()

                    if saveexamples and epoch % 100 == 0:
                        inputs = inputs.cpu().detach().numpy()
                        Uinput = inputs[:,-3]
                        muinput = inputs[:,-2]
                        Jhundinput = inputs[:,-1]
                        inputs = inputs[:,:-3]
                        inputs = np.reshape(inputs, (inputs.shape[0], len(taus), 6*2))
                        outputs_dens = outputs_dens.cpu().detach().numpy()
                        outputs_G = outputs_G.cpu().detach().numpy()
                        #outputs = -(outputs[:,0]+outputs[:,3]+outputs[:,5]) # trace only
                        outputs_G = np.reshape(outputs_G, (outputs_G.shape[0], len(taus), 6*2))
                        #outputs = outputs
                        labels_dens = labels_dens.cpu().detach().numpy()
                        labels_G = labels_G.cpu().detach().numpy()
                        labels_G = np.reshape(labels_G, (labels_G.shape[0], len(taus), 6*2))
                        if not os.path.exists('plot_examples'):
                            os.makedirs('plot_examples')
                        np.save('plot_examples/inputs_test_epoch' + str(epoch) +  '.npy', inputs)
                        np.save('plot_examples/U_test_epoch' + str(epoch) +  '.npy', Uinput)
                        np.save('plot_examples/mu_test_epoch' + str(epoch) +  '.npy', muinput)
                        np.save('plot_examples/Jhund_test_epoch' + str(epoch) +  '.npy', Jhundinput)
                        np.save('plot_examples/outputs_dens_test_epoch' + str(epoch) +  '.npy', outputs_dens)
                        np.save('plot_examples/labels_dens_test_epoch' + str(epoch) +  '.npy', labels_dens)
                        np.save('plot_examples/outputs_G_test_epoch' + str(epoch) +  '.npy', outputs_G)
                        np.save('plot_examples/labels_G_test_epoch' + str(epoch) +  '.npy', labels_G)
                        np.save('plot_examples/taus.npy', taus)
                        np.save('testloss.npy', np.array(test_loss_vals))
                        np.save('testloss_dens.npy', np.array(test_loss_vals_dens))
                        np.save('testloss_G.npy', np.array(test_loss_vals_G))
                        np.save('testloss_its.npy', np.array(test_loss_its))
   

        # save the model
        if save and epoch % 100 == 0:
            torch.save(feedforwardnet.state_dict(), "save_"+str(epoch)+".pth")
            torch.save(net_process.state_dict(), "save_process_"+str(epoch)+".pth")
        
    return train_loss_vals, train_loss_its, test_loss_vals, test_loss_its


# Convert numpy arrays to PyTorch tensors
#train_data_fixedU_tensor = torch.from_numpy(train_datafixedU).float()
#train_labels_fixedU_tensor = torch.from_numpy(train_labelsfixedU).float()
#test_data_fixedU_tensor = torch.from_numpy(test_datafixedU).float()
#test_labels_fixedU_tensor = torch.from_numpy(test_labelsfixedU).float()
#test_labels_fixedU_mean_tensor = torch.from_numpy(test_labels_meanfixedU).float()

train_data_mixedU_tensor = torch.from_numpy(train_datamixedU).float()
train_labels_mixedU_tensor = torch.from_numpy(train_labelsmixedU).float()
train_labels_mixedU_dens_tensor = torch.from_numpy(train_labelsmixedU_dens).float()
test_data_mixedU_tensor = torch.from_numpy(test_datamixedU).float()
test_labels_mixedU_tensor = torch.from_numpy(test_labelsmixedU).float()
test_labels_mixedU_dens_tensor = torch.from_numpy(test_labelsmixedU_dens).float()
#test_labels_mixedU_mean_tensor = torch.from_numpy(test_labels_meanmixedU).float()
train_test_data_mixedU_tensor = torch.from_numpy(np.concatenate((train_datamixedU[:50], train_datamixedU[-50:]),axis=0)).float()
train_test_labels_mixedU_tensor = torch.from_numpy(np.concatenate((train_labelsmixedU[:50] + train_labelsmixedU[-50:]),axis=0)).float()
train_test_labels_mixedU_dens_tensor = torch.from_numpy(np.concatenate((train_labelsmixedU_dens[:50] + train_labelsmixedU_dens[-50:]),axis=0)).float()
# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels, labelsdens, compareoutput=None):
        self.data = data
        self.labels = labels
        self.labelsdens = labelsdens
        self.compareoutput = compareoutput

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        DeltaHyb = self.data[idx]
        Gfunc = self.labels[idx]
        Dens = self.labelsdens[idx]
        if self.compareoutput is not None:
            compareoutput = self.compareoutput[idx]
            sample = {'DeltaHyb': DeltaHyb, 'Gfunc': Gfunc, 'Dens': Dens, 'compareoutput': compareoutput}
        else:
            sample = {'DeltaHyb': DeltaHyb, 'Gfunc': Gfunc, 'Dens': Dens}
        return sample

# Create train and test datasets and dataloaders
#train_dataset_fixedU = CustomDataset(train_data_fixedU_tensor, train_labels_fixedU_tensor)
#test_dataset_fixedU = CustomDataset(test_data_fixedU_tensor, test_labels_fixedU_tensor, test_labels_fixedU_mean_tensor)

#train_loader_fixedU = DataLoader(train_dataset_fixedU, batch_size=32, shuffle=True)
#test_loader_fixedU = DataLoader(test_dataset_fixedU, batch_size=np.shape(test_datafixedU)[0], shuffle=False)

train_dataset_mixedU = CustomDataset(train_data_mixedU_tensor, train_labels_mixedU_tensor, train_labels_mixedU_dens_tensor)
test_dataset_mixedU = CustomDataset(test_data_mixedU_tensor, test_labels_mixedU_tensor, test_labels_mixedU_dens_tensor)
train_test_dataset_mixedU = CustomDataset(train_test_data_mixedU_tensor, train_test_labels_mixedU_tensor, train_test_labels_mixedU_dens_tensor)

train_loader_mixedU = DataLoader(train_dataset_mixedU, batch_size=128, shuffle=True)
test_loader_mixedU = DataLoader(test_dataset_mixedU, batch_size=np.shape(test_datamixedU)[0], shuffle=False)
train_test_loader_mixedU = DataLoader(train_test_dataset_mixedU, batch_size=50, shuffle=False)
#print("Train dataset size:", len(train_dataset))
#print("Train dataset shape:", train_dataset.data.shape, train_dataset.__getitem__(1))

# Call the train function
#trainloss1, train_its1, testloss1, test_its1, testloss_compare1 =  train(feedforwardnet, train_loader_fixedU, val_loader=test_loader_fixedU, save=True, numepochs=200, epochs_start=0)
trainloss2, train_its2, testloss2, test_its2 =  train(feedforwardnet, train_loader_mixedU, val_loader=test_loader_mixedU, save=True, numepochs=5000, epochs_start=0)

np.save('trainloss2.npy', trainloss2)
np.save('train_its2.npy', train_its2)
np.save('testloss2.npy', testloss2)
np.save('test_its2.npy', test_its2)
trainloss = trainloss2 #trainloss1 + trainloss2
train_its = train_its2 #train_its1 + train_its2
testloss = testloss2 #testloss1 + testloss2
test_its = test_its2 #test_its1 + test_its2
#testloss_compare = testloss_compare2 #testloss_compare1 + testloss_compare2

plt.figure()
plt.plot(train_its, trainloss, label='train loss')
plt.plot(test_its, testloss, label='test loss')
#plt.plot(test_its, testloss_compare, label='test loss compare')
plt.legend()
plt.savefig('loss.png')