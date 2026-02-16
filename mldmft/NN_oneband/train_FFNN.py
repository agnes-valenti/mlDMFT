import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from NNet import FeedforwardNet, Gfuncloss
import matplotlib.pyplot as plt
import mldmft.NN_oneband.config 

#import plotly.graph_objects as go
#import plotly.express as px


NsamplesToUse = 3000
NsamplesToUseDmft1 = 400
NsamplesToUseTest = 100
NsamplesToUseDmft1Test = 100


class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)

def get_data(folder1, getU=False, getmu=False, Uval=10.0, muval=0.4, nsamples_to_use_train=NsamplesToUse, nsamples_to_use_test=NsamplesToUseTest):
    taus = np.load(os.path.join(folder1, "tau", "tauvals.npy"))
    print("num taus: ",np.shape(taus)[0])

    train_data_real = np.load(os.path.join(folder1, "tau", "Delta_tau_real_train.npy"))[:nsamples_to_use_train]
    train_data_imag = np.load(os.path.join(folder1, "tau", "Delta_tau_imag_train.npy"))[:nsamples_to_use_train]
    train_data = np.copy(train_data_real)
    train_data[:, :, 1] = train_data_imag[:,:,0]
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
    train_data_1 = np.zeros((np.shape(train_data)[0], np.shape(train_data)[1]+2))
    train_data_1[:, :-2] = train_data
    Umuvals = None
    if not getU:
        Umuvals = np.ones((np.shape(train_data)[0], 2))*Uval
        Umuvals[:,1] = muval
    else:
       Umuvals = np.load(os.path.join(folder1, "Umu_train.npy"))[:nsamples_to_use_train]
    train_data_1[:, -2] = Umuvals[:,0]
    train_data_1[:, -1] = Umuvals[:,1]/Umuvals[:,0]

    test_data_real = np.load(os.path.join(folder1, "tau", "Delta_tau_real_test.npy"))[:nsamples_to_use_test]
    test_data_imag = np.load(os.path.join(folder1, "tau", "Delta_tau_imag_test.npy"))[:nsamples_to_use_test]
    test_data = np.copy(test_data_real)
    test_data[:, :, 1] = test_data_imag[:,:,0]
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))
    test_data_1 = np.zeros((np.shape(test_data)[0], np.shape(test_data)[1]+2))
    test_data_1[:, :-2] = test_data
    if not getU:
        Umuvals = np.ones((np.shape(test_data)[0], 2))*Uval
        Umuvals[:,1] = muval
    else:
        Umuvals = np.load(os.path.join(folder1, "Umu_test.npy"))[:nsamples_to_use_test]
    #Umuvals = np.load(os.path.join(folder1, "Umuvals.npy"))[:nsamples_to_use]
    test_data_1[:, -2] = Umuvals[:,0]
    test_data_1[:, -1] = Umuvals[:,1]/Umuvals[:,0]

    train_labels_real = np.load(os.path.join(folder1, "tau", "G_tau_real_train.npy"))[:nsamples_to_use_train]
    train_labels_imag = np.load(os.path.join(folder1, "tau", "G_tau_imag_train.npy"))[:nsamples_to_use_train]
    train_labels1 = np.copy(train_labels_real)
    train_labels1[:, :, 1] = train_labels_imag[:,:,0]
    train_labels1 = np.reshape(train_labels1, (train_labels1.shape[0], train_labels1.shape[1]*train_labels1.shape[2]))
    #print(train_labels)

    test_labels_real = np.load(os.path.join(folder1, "tau", "G_tau_real_test.npy"))[:nsamples_to_use_test]
    test_labels_imag = np.load(os.path.join(folder1, "tau", "G_tau_imag_test.npy"))[:nsamples_to_use_test]
    test_labels1 = np.copy(test_labels_real)
    test_labels1[:, :, 1] = test_labels_imag[:,:,0]
    test_labels1 = np.reshape(test_labels1, (test_labels1.shape[0], test_labels1.shape[1]*test_labels1.shape[2]))

    test_labels1_mean = np.mean(train_labels1, axis=0)
    test_labels1_mean = np.broadcast_to(test_labels1_mean, (np.shape(test_labels1)[0], np.shape(test_labels1)[1]))
    
    return train_data_1, train_labels1, test_data_1, test_labels1, test_labels1_mean, taus



#folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2_U10/'
#train_dataU10, train_labelsU10, test_dataU10, test_labelsU10, test_labels_meanU10, tausU10 = get_data(folder, getU=False, Uval=10.0)

#folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2/'
#train_dataU1, train_labelsU1, test_dataU1, test_labelsU1, test_labels_meanU1, tausU1 = get_data(folder, getU=False, Uval=1.0)

#train_datafixedU = np.concatenate((train_dataU1, train_dataU10), axis=0)
#train_labelsfixedU = np.concatenate((train_labelsU1, train_labelsU10), axis=0)
#test_datafixedU = np.concatenate((test_dataU1, test_dataU10), axis=0)
#test_labelsfixedU = np.concatenate((test_labelsU1, test_labelsU10), axis=0)
#test_labels_meanfixedU = np.concatenate((test_labels_meanU1, test_labels_meanU10), axis=0)

folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2_moresteps_mixedU_mixedmu'
train_datamixedU, train_labelsmixedU, test_datamixedU, test_labelsmixedU, test_labels_meanmixedU, taus = get_data(folder, getU=True, getmu=True)

folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2_moresteps_mixedU_mixedmu_dmft1'
train_datadmft1, train_labelsdmft1, test_datadmft1, test_labelsdmft1, test_labels_meandmft1, tausdmft1 = get_data(folder, getU=True, nsamples_to_use_train=NsamplesToUseDmft1, nsamples_to_use_test=NsamplesToUseDmft1Test)
train_datamixedU = np.concatenate((train_datamixedU, train_datadmft1), axis=0) 
train_labelsmixedU = np.concatenate((train_labelsmixedU, train_labelsdmft1), axis=0)
#test_datamixedU = np.concatenate((test_dataU1, test_datamixedU, test_datadmft1), axis=0)
#test_labelsmixedU = np.concatenate((test_labelsU1, test_labelsmixedU, test_labelsdmft1), axis=0)
#test_labels_meanmixedU = np.concatenate((test_labels_meanU1,test_labels_meanmixedU, test_labels_meandmft1), axis=0)

#omegas = np.load("/mnt/home/avalenti/ceph/ml-dmft/traindata_newparams/omegas.npy")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

feedforwardnet = FeedforwardNet()
feedforwardnet.to(device);

optimizer = torch.optim.Adam(feedforwardnet.parameters(), lr=0.001)

def train(model, train_loader, val_loader=None,  numepochs=1200, epochs_start=0, save=True, saveexamples=True):
    test_loss_vals = []
    train_loss_vals = []
    train_loss_its = []
    test_loss_its = []
    test_loss_compare_vals = []
    for epoch in range(epochs_start, epochs_start+numepochs): 
        feedforwardnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['DeltaHyb'].to(device).float(), data['Gfunc'].to(device)
            optimizer.zero_grad()
            outputs = feedforwardnet(inputs)

            loss = Gfuncloss(outputs, labels, taus)
            loss.backward()
            optimizer.step()

            # print statistics
            #print('epoch, batch: ', epoch, i, loss, loss.item())
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                    train_loss_vals.append(running_loss / 10)
                    train_loss_its.append(i+epoch*len(train_loader))
                    running_loss = 0.0

             

        feedforwardnet.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels, comparelabels = data['DeltaHyb'].to(device).float(), data['Gfunc'].to(device), data['compareoutput'].to(device)
                    outputs = feedforwardnet(inputs)
                    print('outputs: ', outputs.shape)
                    test_loss = Gfuncloss(outputs, labels, taus) #/len(val_loader)
                    test_loss_compare = Gfuncloss(comparelabels, labels, taus)
                    print('Valid loss: %.3f' % test_loss.item())
                    test_loss_vals.append(test_loss.item())
                    test_loss_compare_vals.append(test_loss_compare.item())
                    test_loss_its.append(epoch*len(train_loader) + i)
                    #plt.figure()

                    if saveexamples:
                        inputs = inputs.cpu().detach().numpy()
                        Uinput = inputs[:,-2]
                        muinput = inputs[:,-1]
                        inputs = inputs[:,:-2]
                        inputs = np.reshape(inputs, (inputs.shape[0], len(taus), 2))
                        outputs = outputs.cpu().detach().numpy()
                        outputs = np.reshape(outputs, (outputs.shape[0], len(taus), 2))
                        labels = labels.cpu().detach().numpy()
                        labels = np.reshape(labels, (labels.shape[0], len(taus), 2))
                        np.save('plot_examples/inputs_test_epoch' + str(epoch) +  '.npy', inputs)
                        np.save('plot_examples/U_test_epoch' + str(epoch) +  '.npy', Uinput)
                        np.save('plot_examples/mu_test_epoch' + str(epoch) +  '.npy', muinput)
                        np.save('plot_examples/outputs_test_epoch' + str(epoch) +  '.npy', outputs)
                        np.save('plot_examples/labels_test_epoch' + str(epoch) +  '.npy', labels)
                        np.save('plot_examples/taus.npy', taus)
                    #plt.plot(omegas, outputs[0,:,0], 'x', label='output real')
                    #plt.plot(omegas, labels[0,:,0], label='label real')
                    #plt.plot(omegas, outputs[0,:,1],  'x',label='output imag')
                    #plt.plot(omegas, labels[0,:,1], label='label imag')
                    #plt.savefig('plots/epoch' + str(epoch) + '_batch' + str(i) + '.png')
                    #running_loss += test_loss.item()
                    #if i % 10 == 9:

       

        # save the model
        if save and epoch % 10 == 9:
            torch.save(feedforwardnet.state_dict(), "save_"+str(epoch)+".pth")
    return train_loss_vals, train_loss_its, test_loss_vals, test_loss_its, test_loss_compare_vals


# Convert numpy arrays to PyTorch tensors
#train_data_fixedU_tensor = torch.from_numpy(train_datafixedU).float()
#train_labels_fixedU_tensor = torch.from_numpy(train_labelsfixedU).float()
#test_data_fixedU_tensor = torch.from_numpy(test_datafixedU).float()
#test_labels_fixedU_tensor = torch.from_numpy(test_labelsfixedU).float()
#test_labels_fixedU_mean_tensor = torch.from_numpy(test_labels_meanfixedU).float()

train_data_mixedU_tensor = torch.from_numpy(train_datamixedU).float()
train_labels_mixedU_tensor = torch.from_numpy(train_labelsmixedU).float()
test_data_mixedU_tensor = torch.from_numpy(test_datamixedU).float()
test_labels_mixedU_tensor = torch.from_numpy(test_labelsmixedU).float()
test_labels_mixedU_mean_tensor = torch.from_numpy(test_labels_meanmixedU).float()

# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels, compareoutput=None):
        self.data = data
        self.labels = labels
        self.compareoutput = compareoutput

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        DeltaHyb = self.data[idx]
        Gfunc = self.labels[idx]
        if self.compareoutput is not None:
            compareoutput = self.compareoutput[idx]
            sample = {'DeltaHyb': DeltaHyb, 'Gfunc': Gfunc, 'compareoutput': compareoutput}
        else:
            sample = {'DeltaHyb': DeltaHyb, 'Gfunc': Gfunc}
        return sample

# Create train and test datasets and dataloaders
#train_dataset_fixedU = CustomDataset(train_data_fixedU_tensor, train_labels_fixedU_tensor)
#test_dataset_fixedU = CustomDataset(test_data_fixedU_tensor, test_labels_fixedU_tensor, test_labels_fixedU_mean_tensor)

#train_loader_fixedU = DataLoader(train_dataset_fixedU, batch_size=32, shuffle=True)
#test_loader_fixedU = DataLoader(test_dataset_fixedU, batch_size=np.shape(test_datafixedU)[0], shuffle=False)

train_dataset_mixedU = CustomDataset(train_data_mixedU_tensor, train_labels_mixedU_tensor)
test_dataset_mixedU = CustomDataset(test_data_mixedU_tensor, test_labels_mixedU_tensor, test_labels_mixedU_mean_tensor)

train_loader_mixedU = DataLoader(train_dataset_mixedU, batch_size=32, shuffle=True)
test_loader_mixedU = DataLoader(test_dataset_mixedU, batch_size=np.shape(test_datamixedU)[0], shuffle=False)

#print("Train dataset size:", len(train_dataset))
#print("Train dataset shape:", train_dataset.data.shape, train_dataset.__getitem__(1))

# Call the train function
#trainloss1, train_its1, testloss1, test_its1, testloss_compare1 =  train(feedforwardnet, train_loader_fixedU, val_loader=test_loader_fixedU, save=True, numepochs=200, epochs_start=0)
trainloss2, train_its2, testloss2, test_its2, testloss_compare2 =  train(feedforwardnet, train_loader_mixedU, val_loader=test_loader_mixedU, save=True, numepochs=2200, epochs_start=0)

trainloss = trainloss2 #trainloss1 + trainloss2
train_its = train_its2 #train_its1 + train_its2
testloss = testloss2 #testloss1 + testloss2
test_its = test_its2 #test_its1 + test_its2
testloss_compare = testloss_compare2 #testloss_compare1 + testloss_compare2

plt.figure()
plt.plot(train_its, trainloss, label='train loss')
plt.plot(test_its, testloss, label='test loss')
plt.plot(test_its, testloss_compare, label='test loss compare')
plt.legend()
plt.savefig('loss.png')