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

def get_data(folder1, nsamples_to_use_train=-1, nsamples_to_use_test=-1, flag_test=True, ind_up=0, ind_dn=1):
    taus = np.load(os.path.join(folder1, "tau", "tauvals.npy"))
    print("num taus: ",np.shape(taus)[0])

    train_data_real = np.load(os.path.join(folder1, "tau", "Delta_tau_real_train.npy"))[:nsamples_to_use_train]
    print("train_data_real shape: ", train_data_real.shape)
    train_data_imag = np.load(os.path.join(folder1, "tau", "Delta_tau_imag_train.npy"))[:nsamples_to_use_train]
    train_data = np.zeros((train_data_real.shape[0], train_data_real.shape[1], 3*2))
    train_data[:, :, 0] = train_data_real[:,:,ind_up,0,0]
    train_data[:, :, 1] = train_data_real[:,:,ind_up,0,1]
    train_data[:, :, 2] = train_data_real[:,:,ind_up,1,1]
    train_data[:, :, 3] = train_data_imag[:,:,ind_up,0,0]
    train_data[:, :, 4] = train_data_imag[:,:,ind_up,0,1]
    train_data[:, :, 5] = train_data_imag[:,:,ind_up,1,1]
    #train_data[:,:,3] = train_data_real[:,:,ind_dn,1,1]
    #train_data[:,:,4] = train_data_real[:,:,ind_dn,1,2]
    #train_data[:,:,5] = train_data_real[:,:,ind_dn,2,2]
    
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
    train_data_1 = np.zeros((np.shape(train_data)[0], np.shape(train_data)[1]+4))
    train_data_1[:, :-4] = train_data
    Umuvals = None
    
    Umuvals = np.load(os.path.join(folder1, "Umu_train.npy"))[:train_data.shape[0]]
    print("Umuvals shape: ", Umuvals.shape)
    train_data_1[:, -4] = Umuvals[:,0]
    train_data_1[:, -3] = Umuvals[:,1]/Umuvals[:,0]
    train_data_1[:, -2] = Umuvals[:,2] #Umuvals[:,0]
    train_data_1[:, -1] = Umuvals[:,3] #Umuvals[:,0]

    if flag_test:
        test_data_real = np.load(os.path.join(folder1, "tau", "Delta_tau_real_test.npy"))[:nsamples_to_use_test]
        test_data_imag = np.load(os.path.join(folder1, "tau", "Delta_tau_imag_test.npy"))[:nsamples_to_use_test]
        test_data = np.zeros((test_data_real.shape[0], test_data_real.shape[1], 3*2))
        test_data[:, :, 0] = test_data_real[:,:,ind_up,0,0]
        test_data[:, :, 1] = test_data_real[:,:,ind_up,0,1]
        test_data[:, :, 2] = test_data_real[:,:,ind_up,1,1]
        test_data[:, :, 3] = test_data_imag[:,:,ind_up,0,0]
        test_data[:, :, 4] = test_data_imag[:,:,ind_up,0,1]
        test_data[:, :, 5] = test_data_imag[:,:,ind_up,1,1]
        
      
        
        test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))
        test_data_1 = np.zeros((np.shape(test_data)[0], np.shape(test_data)[1]+4))
        test_data_1[:, :-4] = test_data

        Umuvals = np.load(os.path.join(folder1, "Umu_test.npy"))[:test_data.shape[0]]
        #Umuvals = np.load(os.path.join(folder1, "Umuvals.npy"))[:nsamples_to_use]
        test_data_1[:, -4] = Umuvals[:,0]
        test_data_1[:, -3] = Umuvals[:,1]/Umuvals[:,0]
        test_data_1[:, -2] = Umuvals[:,2] #Umuvals[:,0]
        test_data_1[:, -1] = Umuvals[:,3] #Umuvals[:,0]

    train_labels_real = np.load(os.path.join(folder1, "tau", "G_tau_real_train.npy"))[:nsamples_to_use_train]
    train_labels_imag = np.load(os.path.join(folder1, "tau", "G_tau_imag_train.npy"))[:nsamples_to_use_train]
    train_labels = np.zeros((train_labels_real.shape[0], train_labels_real.shape[1], 3*2))
    train_labels[:,:,0] = train_labels_real[:,:,ind_up,0,0]
    train_labels[:,:,1] = train_labels_real[:,:,ind_up,0,1]
    train_labels[:,:,2] = train_labels_real[:,:,ind_up,1,1]
    train_labels[:,:,3] = train_labels_imag[:,:,ind_up,0,0]
    train_labels[:,:,4] = train_labels_imag[:,:,ind_up,0,1]
    train_labels[:,:,5] = train_labels_imag[:,:,ind_up,1,1]

    train_labels = np.reshape(train_labels, (train_labels.shape[0], train_labels.shape[1]*train_labels.shape[2]))

    if flag_test:
        test_labels_real = np.load(os.path.join(folder1, "tau", "G_tau_real_test.npy"))[:nsamples_to_use_test]
        test_labels_imag = np.load(os.path.join(folder1, "tau", "G_tau_imag_test.npy"))[:nsamples_to_use_test]
        test_labels = np.zeros((test_labels_real.shape[0], test_labels_real.shape[1], 3*2))
        test_labels[:,:,0] = test_labels_real[:,:,ind_up,0,0]
        test_labels[:,:,1] = test_labels_real[:,:,ind_up,0,1]
        test_labels[:,:,2] = test_labels_real[:,:,ind_up,1,1]      
        test_labels[:,:,3] = test_labels_imag[:,:,ind_up,0,0]
        test_labels[:,:,4] = test_labels_imag[:,:,ind_up,0,1]
        test_labels[:,:,5] = test_labels_imag[:,:,ind_up,1,1]
        
        test_labels = np.reshape(test_labels, (test_labels.shape[0], test_labels.shape[1]*test_labels.shape[2]))
        #test_labels1_mean = np.mean(train_labels1, axis=0)
        #test_labels1_mean = np.broadcast_to(test_labels1_mean, (np.shape(test_labels1)[0], np.shape(test_labels1)[1]))
        
        return train_data_1, train_labels, test_data_1, test_labels, taus
    return train_data_1, train_labels, taus  


#folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2_U10/'
#train_dataU10, train_labelsU10, test_dataU10, test_labelsU10, test_labels_meanU10, tausU10 = get_data(folder, getU=False, Uval=10.0)

#folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_tau_freq_MC2/'
#train_dataU1, train_labelsU1, test_dataU1, test_labelsU1, test_labels_meanU1, tausU1 = get_data(folder, getU=False, Uval=1.0)

#train_datafixedU = np.concatenate((train_dataU1, train_dataU10), axis=0)
#train_labelsfixedU = np.concatenate((train_labelsU1, train_labelsU10), axis=0)
#test_datafixedU = np.concatenate((test_dataU1, test_dataU10), axis=0)
#test_labelsfixedU = np.concatenate((test_labelsU1, test_labelsU10), axis=0)
#test_labels_meanfixedU = np.concatenate((test_labels_meanU1, test_labels_meanU10), axis=0)

folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_2orbs_beta50_mixedUmuJhund_halffilled_mixedfields/'
train_datamixedU, train_labelsmixedU, test_datamixedU, test_labelsmixedU, taus = get_data(folder) #, nsamples_to_use_test=2, nsamples_to_use_train=2)
test_datamixedU = test_datamixedU[:min(10, np.shape(test_datamixedU)[0])]
test_labelsmixedU = test_labelsmixedU[:min(10, np.shape(test_labelsmixedU)[0])]
#test_labels_meanmixedU = test_labels_meanmixedU[:
print("train_data shape: ", np.shape(train_datamixedU))
print("test_data shape: ", np.shape(test_datamixedU))

folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_2orbs_beta50_mixedUmuJhund_halffilled_mixedfields_dmft1/'
train_datamixedU2, train_labelsmixedU2, test_datamixedU2, test_labelsmixedU2, taus = get_data(folder) #, nsamples_to_use_test=2, nsamples_to_use_train=2)
train_datamixedU = np.concatenate((train_datamixedU, train_datamixedU2), axis=0)
train_labelsmixedU = np.concatenate((train_labelsmixedU, train_labelsmixedU2), axis=0)
test_datamixedU2 = test_datamixedU2[:min(10, np.shape(test_datamixedU2)[0])]
test_labelsmixedU2 = test_labelsmixedU2[:min(10, np.shape(test_labelsmixedU2)[0])]
#test_datamixedU = np.concatenate((test_datamixedU, test_datamixedU2), axis=0)
#test_labelsmixedU = np.concatenate((test_labelsmixedU, test_labelsmixedU2), axis=0)
#test_labels_meanmixedU = np.concatenate((test_labels_meanmixedU, test_labels_meanmixedU2), axis=0)

folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_2orbs_beta50_mixedUmuJhund_halffilled_mixedfields_dmft2/'
train_datamixedU3, train_labelsmixedU3, test_datamixedU3, test_labelsmixedU3, taus = get_data(folder) #, nsamples_to_use_test=2, nsamples_to_use_train=2)
train_datamixedU = np.concatenate((train_datamixedU, train_datamixedU3), axis=0)
train_labelsmixedU = np.concatenate((train_labelsmixedU, train_labelsmixedU3), axis=0)
test_datamixedU3 = test_datamixedU3[:min(10, np.shape(test_datamixedU3)[0])]
test_labelsmixedU3 = test_labelsmixedU3[:min(10, np.shape(test_labelsmixedU3)[0])]

#test_datamixedU = np.concatenate((test_datamixedU, test_datamixedU2), axis=0)
#test_labelsmixedU = np.concatenate((test_labelsmixedU, test_labelsmixedU2), axis=0)
#test_labels_meanmixedU = np.concatenate((test_labels_meanmixedU, test_labels_meanmixedU2), axis=0)

folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_2orbs_beta50_mixedUmuJhund_halffilled_mixedfields_dmft3/'
train_datamixedU4, train_labelsmixedU4, test_datamixedU4, test_labelsmixedU4, taus = get_data(folder) #, nsamples_to_use_test=2, nsamples_to_use_train=2)
train_datamixedU = np.concatenate((train_datamixedU, train_datamixedU4), axis=0)
train_labelsmixedU = np.concatenate((train_labelsmixedU, train_labelsmixedU4), axis=0)
test_datamixedU4 = test_datamixedU4[:min(10, np.shape(test_datamixedU4)[0])]
test_labelsmixedU4 = test_labelsmixedU4[:min(10, np.shape(test_labelsmixedU4)[0])]
#test_datamixedU = np.concatenate((test_datamixedU, test_datamixedU2), axis=0)
#test_labelsmixedU = np.concatenate((test_labelsmixedU, test_labelsmixedU2), axis=0)
#test_labels_meanmixedU = np.concatenate((test_labelsmixedU, test_labelsmixedU2), axis=0)


folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_2orbs_beta50_mixedUmuJhund_halffilled_mixedfields_smalldelta_dmft1/'
train_datamixedU5, train_labelsmixedU5, test_datamixedU5, test_labelsmixedU5, taus = get_data(folder) #, nsamples_to_use_test=2, nsamples_to_use_train=2)
train_datamixedU = np.concatenate((train_datamixedU, train_datamixedU5), axis=0)
train_labelsmixedU = np.concatenate((train_labelsmixedU, train_labelsmixedU5), axis=0)

folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_2orbs_beta50_mixedUmuJhund_halffilled_mixedfields_smalldelta/'
train_datamixedU6, train_labelsmixedU6, test_datamixedU6, test_labelsmixedU6, taus = get_data(folder) #, nsamples_to_use_test=2, nsamples_to_use_train=2)
train_datamixedU = np.concatenate((train_datamixedU, train_datamixedU6), axis=0)
train_labelsmixedU = np.concatenate((train_labelsmixedU, train_labelsmixedU6), axis=0)

folder = '/mnt/home/avalenti/ceph/ml-dmft/2orbitals/traindata_2orbs_beta50_mixedUmuJhund_halffilled_mixedfields_dmft1hyb/'
train_datamixedU7, train_labelsmixedU7, test_datamixedU7, test_labelsmixedU7, taus = get_data(folder) #, nsamples_to_use_test=2, nsamples_to_use_train=2)
train_datamixedU = np.concatenate((train_datamixedU, train_datamixedU7), axis=0)
train_labelsmixedU = np.concatenate((train_labelsmixedU, train_labelsmixedU7), axis=0)

folder = '/mnt/home/avalenti/ceph/ml-dmft/2orbitals/traindata_2orbs_beta50_mixedUmuJhund_halffilled_mixedfields_dmft1hyb_dmft1/'
train_datamixedU7, train_labelsmixedU7, test_datamixedU7, test_labelsmixedU7, taus = get_data(folder) #, nsamples_to_use_test=2, nsamples_to_use_train=2)
train_datamixedU = np.concatenate((train_datamixedU, train_datamixedU7), axis=0)
train_labelsmixedU = np.concatenate((train_labelsmixedU, train_labelsmixedU7), axis=0)

folder = '/mnt/home/avalenti/ceph/ml-dmft/2orbitals/traindata_2orbs_beta50_mixedUmuJhund_halffilled_mixedfields_smalldelta_dmft1hyb/'
train_datamixedU7, train_labelsmixedU7, test_datamixedU7, test_labelsmixedU7, taus = get_data(folder) #, nsamples_to_use_test=2, nsamples_to_use_train=2)
train_datamixedU = np.concatenate((train_datamixedU, train_datamixedU7), axis=0)
train_labelsmixedU = np.concatenate((train_labelsmixedU, train_labelsmixedU7), axis=0)
folder = '/mnt/home/avalenti/ceph/ml-dmft/2orbitals/traindata_2orbs_beta50_mixedUmuJhund_halffilled_mixedfields_smalldelta_dmft1_dmft1hyb_smalldelta/'
train_datamixedU7, train_labelsmixedU7, test_datamixedU7, test_labelsmixedU7, taus = get_data(folder) #, nsamples_to_use_test=2, nsamples_to_use_train=2)
train_datamixedU = np.concatenate((train_datamixedU, train_datamixedU7), axis=0)
train_labelsmixedU = np.concatenate((train_labelsmixedU, train_labelsmixedU7), axis=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

feedforwardnet = FeedforwardNet()
feedforwardnet.to(device);

optimizer = torch.optim.Adam(feedforwardnet.parameters(), lr=0.0001, weight_decay=1e-5)

def train(model, train_loader, val_loader=None, val_loader2=None, val_loader3=None, val_loader4=None, numepochs=1200, epochs_start=0, save=True, saveexamples=True):
    test_loss_vals = []
    train_loss_vals = []
    test2_loss_vals =[]
    test3_loss_vals =[]
    test4_loss_vals =[]
    train_loss_its = []
    test_loss_its = []
    test2_loss_its = []
    test3_loss_its = []
    test4_loss_its = []

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
            if i % 10 == 9:    # print every 10 mini-batches
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                    train_loss_vals.append(running_loss / 10)
                    train_loss_its.append(i+epoch*len(train_loader))
                    running_loss = 0.0

             

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
                        Uinput = inputs[:,-4]
                        muinput = inputs[:,-3]
                        Jhundinput = inputs[:,-2]
                        cfieldinput = inputs[:,-1]
                        inputs = inputs[:,:-4]
                        inputs = np.reshape(inputs, (inputs.shape[0], len(taus), 3*2))
                        outputs = outputs.cpu().detach().numpy()
                        outputs = np.reshape(outputs, (outputs.shape[0], len(taus), 3*2))
                        labels = labels.cpu().detach().numpy()
                        labels = np.reshape(labels, (labels.shape[0], len(taus), 3*2))
                        if not os.path.exists('plot_examples'):
                            os.makedirs('plot_examples')
                        np.save('plot_examples/inputs_test_epoch' + str(epoch) +  '.npy', inputs)
                        np.save('plot_examples/U_test_epoch' + str(epoch) +  '.npy', Uinput)
                        np.save('plot_examples/mu_test_epoch' + str(epoch) +  '.npy', muinput)
                        np.save('plot_examples/Jhund_test_epoch' + str(epoch) +  '.npy', Jhundinput)
                        np.save('plot_examples/cfield_test_epoch' + str(epoch) +  '.npy', cfieldinput)
                        np.save('plot_examples/outputs_test_epoch' + str(epoch) +  '.npy', outputs)
                        np.save('plot_examples/labels_test_epoch' + str(epoch) +  '.npy', labels)
                        np.save('plot_examples/taus.npy', taus)
                    np.save('testloss.npy', np.array(test_loss_vals))
                    np.save('trainloss.npy', np.array(train_loss_vals))
                    np.save('testloss_its.npy', np.array(test_loss_its))
                    np.save('trainloss_its.npy', np.array(train_loss_its))

        if val_loader2 and epoch % 100 == 0:
            with torch.no_grad():
                for data in val_loader2:
                    inputs, labels= data['DeltaHyb'].to(device).float(), data['Gfunc'].to(device) #, data['compareoutput'].to(device)
                    outputs = feedforwardnet(inputs)
                    print('outputs: ', outputs.shape)
                    test_loss = Gfuncloss(outputs, labels, taus) #/len(val_loader)
                    #test_loss_compare = Gfuncloss(comparelabels, labels, taus)
                    print('Valid2 loss: %.3f' % test_loss.item())
                    test2_loss_vals.append(test_loss.item())
                    #test_loss_compare_vals.append(test_loss_compare.item())
                    test2_loss_its.append(epoch*len(train_loader) + i)
                    #plt.figure()
                    np.save('test2loss.npy', np.array(test2_loss_vals))
                    np.save('test2loss_its.npy', np.array(test2_loss_its))


        if val_loader3 and epoch % 100 == 0:
            with torch.no_grad():
                for data in val_loader3:
                    inputs, labels= data['DeltaHyb'].to(device).float(), data['Gfunc'].to(device) #, data['compareoutput'].to(device)
                    outputs = feedforwardnet(inputs)
                    print('outputs: ', outputs.shape)
                    test_loss = Gfuncloss(outputs, labels, taus) #/len(val_loader)
                    #test_loss_compare = Gfuncloss(comparelabels, labels, taus)
                    print('Valid3 loss: %.3f' % test_loss.item())
                    test3_loss_vals.append(test_loss.item())
                    #test_loss_compare_vals.append(test_loss_compare.item())
                    test3_loss_its.append(epoch*len(train_loader) + i)
                    #plt.figure()
                    np.save('test3loss.npy', np.array(test3_loss_vals))
                    np.save('test3loss_its.npy', np.array(test3_loss_its))

        if val_loader4 and epoch % 100 == 0:
            with torch.no_grad():
                for data in val_loader4:
                    inputs, labels= data['DeltaHyb'].to(device).float(), data['Gfunc'].to(device) #, data['compareoutput'].to(device)
                    outputs = feedforwardnet(inputs)
                    print('outputs: ', outputs.shape)
                    test_loss = Gfuncloss(outputs, labels, taus) #/len(val_loader)
                    #test_loss_compare = Gfuncloss(comparelabels, labels, taus)
                    print('Valid4 loss: %.3f' % test_loss.item())
                    test4_loss_vals.append(test_loss.item())
                    #test_loss_compare_vals.append(test_loss_compare.item())
                    test4_loss_its.append(epoch*len(train_loader) + i)
                    #plt.figure()
                    np.save('test4loss.npy', np.array(test4_loss_vals))
                    np.save('test4loss_its.npy', np.array(test4_loss_its))
   

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
test_data2_tensor = torch.from_numpy(test_datamixedU2).float()
test_labels2_tensor = torch.from_numpy(test_labelsmixedU2).float()
test_data3_tensor = torch.from_numpy(test_datamixedU3).float()
test_labels3_tensor = torch.from_numpy(test_labelsmixedU3).float()
test_data4_tensor = torch.from_numpy(test_datamixedU4).float()
test_labels4_tensor = torch.from_numpy(test_labelsmixedU4).float()
#test_labels_mixedU_mean_tensor = torch.from_numpy(test_labels_meanmixedU).float()
#train_test_data_mixedU_tensor = torch.from_numpy(np.concatenate((train_datamixedU[:50], train_datamixedU[-50:]),axis=0)).float()
#train_test_labels_mixedU_tensor = torch.from_numpy(np.concatenate((train_labelsmixedU[:50] + train_labelsmixedU[-50:]),axis=0)).float()


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
#train_test_dataset_mixedU = CustomDataset(train_test_data_mixedU_tensor, train_test_labels_mixedU_tensor)
test_dataset_mixedU2 = CustomDataset(test_data2_tensor, test_labels2_tensor)
test_dataset_mixedU3 = CustomDataset(test_data3_tensor, test_labels3_tensor)
test_dataset_mixedU4 = CustomDataset(test_data4_tensor, test_labels4_tensor)

train_loader_mixedU = DataLoader(train_dataset_mixedU, batch_size=32, shuffle=True)
test_loader_mixedU = DataLoader(test_dataset_mixedU, batch_size=np.shape(test_datamixedU)[0], shuffle=False)
test_loader_mixedU2 = DataLoader(test_dataset_mixedU2, batch_size=np.shape(test_data2_tensor)[0], shuffle=False)
test_loader_mixedU3 = DataLoader(test_dataset_mixedU3, batch_size=np.shape(test_data3_tensor)[0], shuffle=False)
test_loader_mixedU4 = DataLoader(test_dataset_mixedU4, batch_size=np.shape(test_data4_tensor)[0], shuffle=False)
#train_test_loader_mixedU = DataLoader(train_test_dataset_mixedU, batch_size=100, shuffle=False)
#print("Train dataset size:", len(train_dataset))
#print("Train dataset shape:", train_dataset.data.shape, train_dataset.__getitem__(1))

# Call the train function
#trainloss1, train_its1, testloss1, test_its1, testloss_compare1 =  train(feedforwardnet, train_loader_fixedU, val_loader=test_loader_fixedU, save=True, numepochs=200, epochs_start=0)
trainloss2, train_its2, testloss2, test_its2 =  train(feedforwardnet, train_loader_mixedU, val_loader=test_loader_mixedU, val_loader2 = test_loader_mixedU2, val_loader3= test_loader_mixedU3, val_loader4=test_loader_mixedU4, save=True, numepochs=20200, epochs_start=0)

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