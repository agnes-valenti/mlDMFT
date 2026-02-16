import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from NNet import FeedforwardNet, Gfuncloss, NeuralBackflow
import sys
import matplotlib.pyplot as plt

nstart = 0 

restart = False
if nstart > 0:
    restart = True


class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)

def get_data(folder1, betaval = 10): 
    taus = np.load(os.path.join(folder1, "tau", "tauvals.npy"))
    numtaus = np.shape(taus)[0]

    train_data_real = np.load(os.path.join(folder1, "tau", "Delta_tau_real_train.npy"))[:,:,0]
    nsamples_to_use_train = np.shape(train_data_real)[0]
    train_data = np.copy(train_data_real)
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1)) 
   
    Umuvals = np.load(os.path.join(folder1, "Umu_train.npy"))[:nsamples_to_use_train]
    train_Umu_data = np.zeros((np.shape(train_data)[0],4))
    train_Umu_data[:, 0] = Umuvals[:,0]
    train_Umu_data[:, 1] = Umuvals[:,1]/Umuvals[:,0]
    train_Umu_data[:, 2] = betaval
    train_Umu_data[:, 3] = numtaus

    test_data_real = np.load(os.path.join(folder1, "tau", "Delta_tau_real_test.npy"))[:,:,0]
    nsamples_to_use_test = np.shape(test_data_real)[0]
    test_data = np.copy(test_data_real)
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1],1 ))

    Umuvals = np.load(os.path.join(folder1, "Umu_test.npy"))[:nsamples_to_use_test]
    test_Umu_data = np.zeros((np.shape(test_data)[0],4))
    test_Umu_data[:, 0] = Umuvals[:,0]
    test_Umu_data[:, 1] = Umuvals[:,1]/Umuvals[:,0]
    test_Umu_data[:, 2] = betaval
    test_Umu_data[:, 3] = numtaus

    train_labels_real = np.load(os.path.join(folder1, "tau", "G_tau_real_train.npy"))[:nsamples_to_use_train,:,0]
    train_labels1 = np.copy(train_labels_real)
    train_labels1 = np.reshape(train_labels1, (train_labels1.shape[0], train_labels1.shape[1], 1)) 

    test_labels_real = np.load(os.path.join(folder1, "tau", "G_tau_real_test.npy"))[:nsamples_to_use_test,:,0]
    test_labels1 = np.copy(test_labels_real)
    test_labels1 = np.reshape(test_labels1, (test_labels1.shape[0], test_labels1.shape[1], 1)) 

    test_labels1_mean = np.mean(test_labels1, axis=0)
    test_labels1_mean = np.broadcast_to(test_labels1_mean, (np.shape(test_labels1)[0], np.shape(test_labels1)[1], 1))
    
   
    return train_data, train_Umu_data, train_labels1, test_data, test_Umu_data, test_labels1, test_labels1_mean, taus




folders = ['/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2_moresteps_mixedU_mixedmu', 
'/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2_moresteps_mixedU_mixedmu_dmft1', 
'/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2_moresteps_mixedU_mixedmu_beta50', 
'/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2_moresteps_mixedU_mixedmu_beta50_dmft1',
'/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2_moresteps_mixedU_mixedmu_beta20',
'/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2_moresteps_mixedU_mixedmu_beta30',
'/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2_moresteps_mixedU_mixedmu_beta60']
betavals = [10, 10, 50, 50, 20, 30, 60]
train_datamixedU = None
train_data_Umu = None
train_labelsmixedU = None
test_datamixedU = None
test_data_Umu = None    
test_labelsmixedU = None
test_labels_meanmixedU = None
tau_train = None
tau_test = None

# determine maximum tau length from all folders
taulen = 0
for folder in folders:
    tauvals = np.load(os.path.join(folder, "tau", "tauvals.npy"))
    if np.shape(tauvals)[0] > taulen:
        taulen = np.shape(tauvals)[0]
print("Maximum tau length: ", taulen)
#taulen=50
    
#sys.exit("Program aborted by user.")

for folder, betaval_it in zip(folders, betavals):
    trD, trUmuD, trL, teD, teUmuD, teL, temeanD, tauD = get_data(folder, betaval=betaval_it, getU=True, getmu=True) #, nsamples_to_use_train=ntrain, nsamples_to_use_test=ntest)
    data_size = np.shape(trD)[0]
    data_size_test = np.shape(teD)[0]
    taulen_batch = np.shape(tauD)[0]
    trD_full = np.zeros((data_size, taulen, 1))
    trD_full[:, :taulen_batch, :] = trD
    trL_full = np.zeros((data_size, taulen, 1))
    trL_full[:, :taulen_batch, :] = trL
    teD_full = np.zeros((data_size_test, taulen, 1))
    teD_full[:, :taulen_batch, :] = teD
    teL_full = np.zeros((data_size_test, taulen, 1))
    teL_full[:, :taulen_batch, :] = teL
    tauD_train = np.tile(tauD, (data_size, 1)) 
    tauD_test = np.tile(tauD, (data_size_test, 1))
    tauD_train_full = np.zeros((data_size, taulen))
    tauD_train_full[:, :taulen_batch] = tauD_train
    tauD_test_full = np.zeros((data_size_test, taulen))
    tauD_test_full[:, :taulen_batch] = tauD_test

    if train_datamixedU is None:
        train_datamixedU = trD_full
        train_data_Umu = trUmuD
        train_labelsmixedU = trL_full
        test_datamixedU = teD_full
        test_data_Umu = teUmuD
        test_labelsmixedU = teL_full
        tau_train = tauD_train_full
        tau_test = tauD_test_full
    else:
        train_datamixedU = np.concatenate((train_datamixedU, trD_full), axis=0) 
        train_labelsmixedU = np.concatenate((train_labelsmixedU, trL_full), axis=0)
        train_data_Umu = np.concatenate((train_data_Umu, trUmuD), axis=0)
        test_datamixedU = np.concatenate((test_datamixedU, teD_full), axis=0)
        test_data_Umu = np.concatenate((test_data_Umu, teUmuD), axis=0)
        test_labelsmixedU = np.concatenate((test_labelsmixedU, teL_full), axis=0)
        tau_train = np.concatenate((tau_train, tauD_train_full), axis=0)
        tau_test = np.concatenate((tau_test, tauD_test_full), axis=0)
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

feedforwardnet = NeuralBackflow()
feedforwardnet.to(device);


if restart:
    print("Loading saved model from save_{}.pth".format(nstart))
    #feedforwardnet.load_state_dict(torch.load("save_{}.pth".format(nstart), map_location=device))
    feedforwardnet.load_state_dict(torch.load("save_{}.pth".format(nstart), map_location=device))

optimizer = torch.optim.Adam(feedforwardnet.parameters(), lr=0.001)
# Create the folder "plot_examples" if it doesn't already exist
if not os.path.exists("plot_examples"):
    os.makedirs("plot_examples")
def train(model, train_loader, val_loader=None,  numepochs=1200, epochs_start=0, save=True, saveexamples=True):
    test_loss_vals = []
    train_loss_vals = []
    train_loss_its = []
    test_loss_its = []
    for epoch in range(epochs_start, epochs_start+numepochs): 
        feedforwardnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, extra_params, tauvals_to_train, labels = data['DeltaHyb'].to(device).float(), data['extra_params'].to(device).float(), data['tauvals'].to(device).float() ,data['Gfunc'].to(device)
            optimizer.zero_grad()
            outputs = feedforwardnet(inputs, tauvals_to_train, extra_params)
       
            loss = Gfuncloss(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:    
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                    train_loss_vals.append(running_loss / 10)
                    train_loss_its.append(i+epoch*len(train_loader))
                    running_loss = 0.0

             

        feedforwardnet.eval()

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, extra_params, tauvals_to_test, labels, = data['DeltaHyb'].to(device).float(), data['extra_params'].to(device).float(), data['tauvals'].to(device).float() , data['Gfunc'].to(device)
                    outputs = feedforwardnet(inputs, tauvals_to_test, extra_params)

                    test_loss = Gfuncloss(outputs, labels) 
                    print('Valid loss: %.3f' % test_loss.item())
                    test_loss_vals.append(test_loss.item())
                    test_loss_its.append(epoch*len(train_loader) + i)

                    if saveexamples:
                        inputs = inputs.cpu().detach().numpy()
                        Umuinput = extra_params.cpu().detach().numpy()
                        Uinput = Umuinput[:,0]
                        muinput = Umuinput[:,1]
                        betainput = Umuinput[:,2]
                        tauleninput = Umuinput[:,3]
                        
                        inputs = np.reshape(inputs, (inputs.shape[0], taulen, 1))
                        outputs = outputs.cpu().detach().numpy()
                        outputs = np.reshape(outputs, (outputs.shape[0], taulen, 1))
                        labels = labels.cpu().detach().numpy()
                        labels = np.reshape(labels, (labels.shape[0], taulen, 1))

                        np.save('plot_examples/inputs_test_epoch' + str(epoch) +  '.npy', inputs)
                        np.save('plot_examples/U_test_epoch' + str(epoch) +  '.npy', Uinput)
                        np.save('plot_examples/mu_test_epoch' + str(epoch) +  '.npy', muinput)
                        np.save('plot_examples/beta_test_epoch' + str(epoch) +  '.npy', betainput)
                        np.save('plot_examples/taulen_test_epoch' + str(epoch) +  '.npy', tauleninput)
                        np.save('plot_examples/outputs_test_epoch' + str(epoch) +  '.npy', outputs)
                        np.save('plot_examples/labels_test_epoch' + str(epoch) +  '.npy', labels)
                        np.save('plot_examples/taus.npy', tauvals_to_test.cpu().detach().numpy())
                           
        # save the model
        if save and epoch % 10 == 9:
            torch.save(feedforwardnet.state_dict(), "save_"+str(epoch)+".pth")
    return train_loss_vals, train_loss_its, test_loss_vals, test_loss_its #, test_loss_compare_vals

train_data_mixedU_tensor = torch.from_numpy(train_datamixedU).float()
train_labels_mixedU_tensor = torch.from_numpy(train_labelsmixedU).float()
test_data_mixedU_tensor = torch.from_numpy(test_datamixedU).float()
test_labels_mixedU_tensor = torch.from_numpy(test_labelsmixedU).float()
train_Umu_tensor = torch.from_numpy(train_data_Umu).float()
test_Umu_tensor = torch.from_numpy(test_data_Umu).float()
tau_train_tensor = torch.from_numpy(tau_train).float()
tau_test_tensor = torch.from_numpy(tau_test).float()

class CustomDataset(Dataset):
    def __init__(self, data, Umu, tau_vals, labels, compareoutput=None):
        self.data = data
        self.Umu = Umu
        self.labels = labels
        self.compareoutput = compareoutput
        self.taus = tau_vals

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        DeltaHyb = self.data[idx]
        Gfunc = self.labels[idx]
        extra_params = self.Umu[idx, :]
        taus_ = self.taus[idx, :]
        sample = {'DeltaHyb': DeltaHyb, 'extra_params' : extra_params, 'Gfunc': Gfunc, 'tauvals': taus_,}
        return sample

train_dataset_mixedU = CustomDataset(train_data_mixedU_tensor, train_Umu_tensor, tau_train_tensor, train_labels_mixedU_tensor)
test_dataset_mixedU = CustomDataset(test_data_mixedU_tensor, test_Umu_tensor, tau_test_tensor, test_labels_mixedU_tensor)

train_loader_mixedU = DataLoader(train_dataset_mixedU, batch_size=32, shuffle=True)
test_loader_mixedU = DataLoader(test_dataset_mixedU, batch_size=np.shape(test_datamixedU)[0], shuffle=False)

trainloss2, train_its2, testloss2, test_its2 =  train(feedforwardnet, train_loader_mixedU, val_loader=test_loader_mixedU, save=True, numepochs=10200, epochs_start=nstart)

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