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

#import plotly.graph_objects as go
#import plotly.express as px


#NsamplesToUse = 3000
#NsamplesToUseDmft1 = 400
#NsamplesToUseTest = 100
#NsamplesToUseDmft1Test = 100


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



folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJhund/'
train_datamixedU, train_labelsmixedU, test_datamixedU, test_labelsmixedU, taus = get_data(folder) #, nsamples_to_use_test=2, nsamples_to_use_train=2)
print("train_data shape: ", np.shape(train_datamixedU))
print("test_data shape: ", np.shape(test_datamixedU))


folders =[
'/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJhund_dmft1/',
'/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJhund_dmft2/',  
'/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm/',
'/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_dmft1/', 
'/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_halffilled/',
'/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_halffilled_dmft1/',
'/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_halffilled_smallscale/',
'/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_halffilled_smallscale_dmft1/',
'/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_largemurange/',
'/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_largemurange_dmft1/',
'/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_largemurange_smallscale/',
'/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_largemurange_smallscale_dmft1/',
'/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJ_largemurange_smallscale/',
'/mnt/home/avalenti/ceph/ml-dmft/traindata_3orbs_offdiag_beta10_mixedUmuJ_largemurange_smallscale_dmft1/',
'/mnt/home/avalenti/ceph/ml-dmft/3orbitals/traindata_3orbs_offdiag_beta10_mixedUmuJhund_dmft1hyb/',
'/mnt/home/avalenti/ceph/ml-dmft/3orbitals/traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_halffilled_dmft1hyb/',
'/mnt/home/avalenti/ceph/ml-dmft/3orbitals/traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_largemurange_dmft1hyb/',
'/mnt/home/avalenti/ceph/ml-dmft/3orbitals/traindata_3orbs_offdiag_beta10_mixedUmuJhund_symm_largemurange_smallscale_dmft1hyb/'
]

for folder in folders:
    print("Loading data from folder: ", folder)
    train_datadmft1, train_labelsdmft1, test_datadmft1, test_labeldmft1, tausdmft1 = get_data(folder)
    test_datadmft1 = test_datadmft1[:20]
    test_labeldmft1 = test_labeldmft1[:20]
    train_datamixedU = np.concatenate((train_datamixedU, train_datadmft1), axis=0) 
    train_labelsmixedU = np.concatenate((train_labelsmixedU, train_labelsdmft1), axis=0)
    test_datamixedU = np.concatenate((test_datamixedU, test_datadmft1), axis=0)
    test_labelsmixedU = np.concatenate((test_labelsmixedU, test_labeldmft1), axis=0)

    random_indices = np.arange(train_datamixedU.shape[0])
    np.random.shuffle(random_indices)
    random_indices = random_indices[:int(0.1*train_datamixedU.shape[0])]
    noisy_traindata = train_datamixedU[random_indices] + np.random.normal(0, 0.005, train_datamixedU[random_indices].shape)
    noisy_trainlabels = train_labelsmixedU[random_indices]
    train_datamixedU = np.concatenate((train_datamixedU, noisy_traindata), axis=0)
    train_labelsmixedU = np.concatenate((train_labelsmixedU, train_labelsmixedU[random_indices]), axis=0)
    
   
#test_datamixedU = np.concatenate((test_datamixedU[:20], test_datamixedU[-20:]), axis=0)
#test_labelsmixedU = np.concatenate((test_labelsmixedU[:20], test_labelsmixedU[-20:]), axis=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

feedforwardnet = FeedforwardNet()
feedforwardnet.to(device);

optimizer = torch.optim.Adam(feedforwardnet.parameters(), lr=0.001, weight_decay=1e-5)

def train(model, train_loader, val_loader=None, val_train_loader=None, numepochs=1200, epochs_start=0, save=True, saveexamples=True):
    test_loss_vals = []
    train_loss_vals = []
    train_loss_its = []
    test_loss_its = []
    #test_loss_compare_vals = []
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
            if i % 200 == 0:    # print every 100 mini-batches
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.5f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                    train_loss_vals.append(running_loss / 10)
                    train_loss_its.append(i+epoch*len(train_loader))
                    running_loss = 0.0
                    np.save('trainloss.npy', np.array(train_loss_vals))
                    np.save('train_its.npy', np.array(train_loss_its))

             

        feedforwardnet.eval()
        correct = total = 0

        # validation
        if val_loader and epoch % 100 == 0:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels= data['DeltaHyb'].to(device).float(), data['Gfunc'].to(device) #, data['compareoutput'].to(device)
                    outputs = feedforwardnet(inputs)
                    print('outputs: ', outputs.shape)
                    test_loss = Gfuncloss(outputs, labels, taus) #/len(val_loader)
                    #test_loss_compare = Gfuncloss(comparelabels, labels, taus)
                    print('Valid loss: %.3f' % test_loss.item())
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
                        outputs = outputs.cpu().detach().numpy()
                        outputs = np.reshape(outputs, (outputs.shape[0], len(taus), 6*2))
                        labels = labels.cpu().detach().numpy()
                        labels = np.reshape(labels, (labels.shape[0], len(taus), 6*2))
                        if not os.path.exists('plot_examples'):
                            os.makedirs('plot_examples')
                        np.save('plot_examples/inputs_test_epoch' + str(epoch) +  '.npy', inputs)
                        np.save('plot_examples/U_test_epoch' + str(epoch) +  '.npy', Uinput)
                        np.save('plot_examples/mu_test_epoch' + str(epoch) +  '.npy', muinput)
                        np.save('plot_examples/Jhund_test_epoch' + str(epoch) +  '.npy', Jhundinput)
                        np.save('plot_examples/outputs_test_epoch' + str(epoch) +  '.npy', outputs)
                        np.save('plot_examples/labels_test_epoch' + str(epoch) +  '.npy', labels)
                        np.save('plot_examples/taus.npy', taus)
                        np.save('testloss.npy', np.array(test_loss_vals))
                        np.save('testloss_its.npy', np.array(test_loss_its))
   

        # save the model
        if save and epoch % 100 == 0:
            torch.save(feedforwardnet.state_dict(), "save_"+str(epoch)+".pth")
    return train_loss_vals, train_loss_its, test_loss_vals, test_loss_its


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
#test_labels_mixedU_mean_tensor = torch.from_numpy(test_labels_meanmixedU).float()
train_test_data_mixedU_tensor = torch.from_numpy(np.concatenate((train_datamixedU[:50], train_datamixedU[-50:]),axis=0)).float()
train_test_labels_mixedU_tensor = torch.from_numpy(np.concatenate((train_labelsmixedU[:50] + train_labelsmixedU[-50:]),axis=0)).float()

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
test_dataset_mixedU = CustomDataset(test_data_mixedU_tensor, test_labels_mixedU_tensor)
train_test_dataset_mixedU = CustomDataset(train_test_data_mixedU_tensor, train_test_labels_mixedU_tensor)

train_loader_mixedU = DataLoader(train_dataset_mixedU, batch_size=512, shuffle=True)
test_loader_mixedU = DataLoader(test_dataset_mixedU, batch_size=np.shape(test_datamixedU)[0], shuffle=False)
train_test_loader_mixedU = DataLoader(train_test_dataset_mixedU, batch_size=50, shuffle=False)
#print("Train dataset size:", len(train_dataset))
#print("Train dataset shape:", train_dataset.data.shape, train_dataset.__getitem__(1))

# Call the train function
#trainloss1, train_its1, testloss1, test_its1, testloss_compare1 =  train(feedforwardnet, train_loader_fixedU, val_loader=test_loader_fixedU, save=True, numepochs=200, epochs_start=0)
trainloss2, train_its2, testloss2, test_its2 =  train(feedforwardnet, train_loader_mixedU, val_loader=test_loader_mixedU, save=True, numepochs=10200, epochs_start=0)

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