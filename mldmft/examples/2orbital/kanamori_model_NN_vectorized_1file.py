import os
import sys
from triqs.gf import *
from triqs.gf.descriptors import Function
from h5 import *
from triqs.operators import *
from numpy import array,zeros, sinh,cosh, matrix
import triqs.utility.mpi as mpi
#import triqs_ctseg as ctseg_new
from triqs.plot.mpl_interface import *
import numpy as np
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#folder ='/mnt/home/avalenti/ceph/ml-dmft/train_model_2orbs_halffilled_beta50_newrange_semicircular/' 
import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from triqs.plot.mpl_interface import oplot,plt

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', '-e', type=int, default=6000, help='epoch number to load')
parser.add_argument('--model', '-m', type=str, default='C2', help='model name to load')
args = parser.parse_args()

ckpt_epoch = args.epoch
ckpt_epoch = str(ckpt_epoch)

model_name = args.model
folder ='/mnt/home/avalenti/ceph/ml-dmft/2orbitals/train_model_' + model_name + '/'
sys.path.append(folder)
from NNet import FeedforwardNet, Gfuncloss

n_dmft_loops = 150

tauvals = np.load("tauvals.npy") 
#device = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

feedforwardnet = FeedforwardNet()
feedforwardnet.to(device); 
feedforwardnet.load_state_dict(torch.load(folder+"save_" + ckpt_epoch + ".pth", map_location=device, weights_only=True)) #3800
feedforwardnet.eval()
NNsolver = {}


Deltas ={
      #'values': [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.4, 1.8], #1.0], 
      'values': [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.4, 1.8, 2.0, 2.2, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,3.1,3.2], #1.0], 
    }

folder = '/mnt/home/avalenti/ceph/ml-dmft/dmft_NN_2band_Fig1_efficient/semicircular_startingpoint' #/mnt/home/avalenti/ceph/ml-dmft/traindata_2orbs_beta50_mixedUmuJhund_halffilled_mixedfields/'
#folder = '/mnt/home/avalenti/ceph/ml-dmft/traindata_2orbs_beta50_mixedUmuJhund_halffilled_mixedfields/tau/'
taus = np.load(os.path.join(folder, "dlr_taus.npy"))
#taus = np.load(os.path.join(folder, "taus.npy"))
np.save("taus.npy", taus)
train_data_real = np.load(os.path.join(folder, "Delta_tau_batch_real.npy")) #"G_tau_real_test.npy"))
train_data_imag = np.load(os.path.join(folder, "Delta_tau_batch_imag.npy")) #G_tau_imag_test.npy"))
#train_data_real = np.load(os.path.join(folder, "G_tau_real_test.npy"))
#train_data_imag = np.load(os.path.join(folder, "G_tau_imag_test.npy"))
train_data = np.zeros((train_data_real.shape[0], train_data_real.shape[1], 3*2))
ind_up = 0
train_data[:, :, 0] = train_data_real[:,:,ind_up,0,0]
train_data[:, :, 1] = train_data_real[:,:,ind_up,0,1]
train_data[:, :, 2] = train_data_real[:,:,ind_up,1,1]
train_data[:,:,3] = train_data_imag[:,:,ind_up,0,0]
train_data[:,:,4] = train_data_imag[:,:,ind_up,0,1]
train_data[:,:,5] = train_data_imag[:,:,ind_up,1,1]

train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
train_data_1 = np.zeros((np.shape(train_data)[0], np.shape(train_data)[1]+4))
train_data_1[:, :-4] = train_data


#cfield_splitting = 0.5 #1.0 #0.25
fig, ax = plt.subplots()  # create a single Axes object instead of an ndarray
Jval_factors = [0.0, 0.25] #, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25] #0.0
#Uvals = np.linspace(0.05, 9.0, 100)


# Prepare the grid of parameters
cfield_splittings = np.arange(0.0,3.2,0.05) #np.linspace(0.0,3.2,50) #np.array(Deltas['values'])
Jval_factors = np.array(Jval_factors)
Uvals = np.arange(0.5,9.0,0.5) #np.linspace(0.05, 9.0, 10)

# Create a meshgrid for all combinations of cfield_splitting, Jval_factor, and Uval
Jval_factor_grid, Uval_grid, cfield_grid = np.meshgrid(Jval_factors, Uvals, cfield_splittings, indexing='ij')

# Flatten the grids to create a batch of inputs
cfield_flat = cfield_grid.flatten()
Jval_factor_flat = Jval_factor_grid.flatten()
Uval_flat = Uval_grid.flatten()

print(Jval_factor_flat)

# Compute Jval and muval for the entire batch
Jval_flat = Uval_flat * Jval_factor_flat
muval_flat = 3.0 / 2.0 * Uval_flat - 5.0 / 2.0 * Jval_flat

# Initialize the input batch for the neural network
batch_size = len(Uval_flat)
Deltatau_batch = np.zeros((batch_size, train_data_1.shape[1]))
Deltatau_batch[:, :-4] = train_data_1[0, :-4]
Deltatau_batch[:, -4] = Uval_flat
Deltatau_batch[:, -3] = muval_flat / Uval_flat
Deltatau_batch[:, -2] = Jval_flat
Deltatau_batch[:, -1] = cfield_flat

outputs_old = None
# Perform the DMFT loops
eps_vals = []
plt.figure()
eps_mixing=0.1
for iloop in range(n_dmft_loops):
  # Process the entire batch through the neural network
  print(f"DMFT loop {iloop+1}/{n_dmft_loops}")
  Goutput_batch = feedforwardnet((torch.from_numpy(Deltatau_batch).float()).to(device)).detach().cpu().numpy()

  # Update the Deltatau_batch with the outputs
  outputvals_batch = np.zeros_like(Deltatau_batch)
  outputvals_batch[:, :-4] = Goutput_batch
  outputvals_batch[:, -4:] = Deltatau_batch[:, -4:]
  Deltatau_batch = eps_mixing*np.copy(outputvals_batch)+ (1-eps_mixing)*np.copy(Deltatau_batch)

  # Reshape outputs for saving
  outputs = outputvals_batch[:, :-4]
  outputs = np.reshape(outputs, (batch_size, -1, 3 * 2))
  if outputs_old is not None:
    print(outputs[1,1,0], outputs_old[1,1,0])
    eps_conv = np.linalg.norm(outputs[:,:,0] - outputs_old[:,:,0], axis = 1)
    eps_vals.append(eps_conv[1])
    if iloop % 5 ==0 and iloop>50:
       plt.plot(tauvals,outputs[1,:,0], alpha = iloop*1.0/n_dmft_loops, color = cm.get_cmap('viridis')(iloop*1.0/n_dmft_loops), label=f"Loop {iloop+1}")
       plt.plot(tauvals,outputs_old[1,:,0], 'x', alpha = iloop*1.0/n_dmft_loops, color = cm.get_cmap('viridis')(iloop*1.0/n_dmft_loops))
    #plt.plot(eps_conv)
    #plt.savefig(f"convergence_dmftloop{iloop+1}_" + ckpt_epoch + model_name + "_batched_horiz.png")
    #print(f"Max change in outputs this loop: {np.max(eps_conv)}")
  outputs_old = outputs.copy()

  # Save the outputs for each combination of parameters
  if not os.path.exists("data_" + ckpt_epoch + model_name + "_batched_horiz"):
    os.makedirs("data_" + ckpt_epoch + model_name + "_batched_horiz")

  if 1==1: #iloop == n_dmft_loops - 1:
      np.save(
        f"data_" + ckpt_epoch + model_name + f"_batched_horiz/dmft_NN{iloop}.npy",
        outputs
      )
      np.save(
        f"data_" + ckpt_epoch + model_name + f"_batched_horiz/dmft_NN{iloop}_Uval_flat.npy",
        Uval_flat
      )
      np.save(
        f"data_" + ckpt_epoch + model_name + f"_batched_horiz/dmft_NN{iloop}_Jval_factor_flat.npy",
        Jval_factor_flat
      )
      np.save(
        f"data_" + ckpt_epoch + model_name + f"_batched_horiz/dmft_NN{iloop}_cfield_flat.npy",
        cfield_flat
      )
plt.legend()
plt.savefig("test_conv.png")
#eps_vals = np.array(eps_vals)
plt.figure()
plt.plot(range(1, n_dmft_loops), eps_vals[:])
plt.xlabel('DMFT Loop')
plt.ylabel('Change in Outputs')
plt.title('Convergence of DMFT Loops')
plt.savefig(f"convergence_batched_horiz.png")

