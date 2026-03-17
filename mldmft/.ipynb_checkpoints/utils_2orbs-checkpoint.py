import os
import sys
from triqs.gf import *
from triqs.gf.descriptors import Function
from h5 import *
from triqs.operators import *
import triqs.utility.mpi as mpi
#import triqs_ctseg as ctseg_new
from triqs.plot.mpl_interface import *
import numpy as np
import matplotlib.pyplot as plt


def _make_1d_to_2d(data_1d, n_orb = 2, sample_ind=0):
    lentaus = np.shape(data_1d)[1]
    n_orb = np.sqrt(0.25+np.shape(data_1d)[2])-0.5
    assert abs(n_orb - round(n_orb)) < 1e-15, "n_orb is not close to an integer within the tolerance"
    n_orb = int(round(n_orb))
    data_2d =np.zeros((lentaus, n_orb, n_orb), dtype=np.complex128)
    it = 0
    for i in range(n_orb):
        for j in range(i+1):
            data_2d[:,j,i] = data_1d[sample_ind, :, it] + 1j *data_1d[sample_ind, :, it+n_orb*(n_orb+1)//2]
            it += 1
            if i != j:
                data_2d[:,i,j] = data_2d[:,j,i]
    return data_2d

def _make_2d_to_1d(data_2d, n_orb = 2):
    lentaus = np.shape(data_2d)[0]
    n_orb = np.shape(data_2d)[1]
    data_1d = np.zeros((1, lentaus, ((n_orb*n_orb-n_orb)//2 + n_orb)*2))
    it = 0
    for i in range(n_orb):
        for j in range(i+1):
            data_1d[0, :, it] = data_2d[:,j,i].real
            data_1d[0, :, it+n_orb*(n_orb+1)//2] = data_2d[:,j,i].imag
            it += 1
    return data_1d

def NNoutput_to_DLR(Goutput, dlr_iw_mesh, sample_ind=0, gf_struct = [('up', 2), ('down', 2)]):
    n_orb = gf_struct[0][1]
    Goutput = np.reshape(Goutput, (Goutput.shape[0], -1, n_orb*n_orb + n_orb)) #lentaus
    Goutput = _make_1d_to_2d(Goutput, sample_ind) 
    Goutput_iw_dlr = BlockGf(mesh=dlr_iw_mesh, gf_struct=gf_struct)
    Goutput_dlr = make_gf_dlr(Goutput_iw_dlr)
    Goutputtau_dlr = make_gf_dlr_imtime(Goutput_dlr)
    it = 0
    for tau in Goutputtau_dlr.mesh:
        Goutputtau_dlr['up'][tau][:,:]=Goutput[it,:,:]
        Goutputtau_dlr['down'][tau][:,:]=Goutput[it,:,:]
        it += 1
    Goutput_dlr = make_gf_dlr(Goutputtau_dlr)
    return Goutput_dlr

def DLR_to_NNinput(Delta_dlr, dlr_tau_mesh_loaded, U, mu, J, cfield):
    """Convert DLR representation to neural network input format.

    Args:
        Delta_dlr (BlockGf): The DLR representation to convert.
        dlr_tau_mesh_loaded (triqs mesh): The mesh in imaginary time to use for the DLR representation.
        U (float): The interaction strength.
        mu (float): The chemical potential.
        J (float): Hund's coupling.
        cfield (fload): crystal field splitting.

    Returns:
        np.ndarray: The neural network input in the correct format.
    """
    taus = []
    ntaus_dlr = 0 
    for tau in dlr_tau_mesh_loaded:
        taus.append(tau.value)
        ntaus_dlr += 1
    #assert(np.linalg.norm(np.array(taus) - mesh_saved) < 1e-6)
    
    Delta_tau_dlr = make_gf_dlr_imtime(Delta_dlr)
    n_orb = 2
    Deltavals_up = np.zeros((ntaus_dlr, n_orb, n_orb)) + 0j 
   
    it = 0 
    for tau in Delta_tau_dlr['up'].mesh:
        Deltavals_up[it, 0, 0] = (Delta_tau_dlr['up'][tau][0,0])
        Deltavals_up[it, 0, 1] = (Delta_tau_dlr['up'][tau][0,1])
        Deltavals_up[it, 1, 0] = (Delta_tau_dlr['up'][tau][1,0])
        Deltavals_up[it, 1, 1] = (Delta_tau_dlr['up'][tau][1,1])
        it += 1

    Deltavals_NN = _make_2d_to_1d(Deltavals_up)
    Deltavals_NN = Deltavals_NN.reshape((Deltavals_NN.shape[0], Deltavals_NN.shape[1]*Deltavals_NN.shape[2]))
    nn_input = np.zeros((Deltavals_NN.shape[0], Deltavals_NN.shape[1]+4))
    nn_input[:,:-4] = Deltavals_NN[:,:] 
    nn_input[:,-4] = U
    nn_input[:,-3] = mu/U
    nn_input[:,-2] = J
    nn_input[:,-1] = cfield

    return nn_input

