import os
import sys
from triqs.gf import *
from triqs.gf.descriptors import Function
from h5 import *
from triqs.operators import *
#import triqs.utility.mpi as mpi
#import triqs_ctseg as ctseg_new
from triqs.plot.mpl_interface import *
import numpy as np
import matplotlib.pyplot as plt

mesh_saved = np.array([
    0.0006579972660322427, 0.01552912716736704, 0.048112196024241626, 0.14547652832105934,
    0.22532530397575837, 0.3355869082085571, 0.5637530521867528, 0.768053434675183,
    1.044531793039739, 1.4033941755170067, 1.8778558244829933, 2.21514064190638,
    2.572397568193933, 3.2111877734315253, 3.6258279309176786, 4.178127172158956,
    5.144795136387866, 6.144427477401464, 7.251655861835357, 8.501533965322126,
    9.886956661843824, 11.746688276329285, 13.405248906273899, 15.511218937326174,
    17.229951165011954, 19.773913323687648, 22.454306808272108, 27.922290090394142,
    31.920819454448537, 34.95788817497393, 38.97756212534765, 42.07770990960586,
    45.42229009039414, 47.54569319172789, 51.95990233002391, 55.49668827632929,
    57.1552489062739, 58.772846595863946, 60.113043338156174, 61.13943743237448,
    63.019427477401464, 63.855572522598536, 64.63060946866308, 65.61447204374349,
    66.24428835103402, 66.64868777343152, 67.31530473433155, 67.87461650866946,
    68.32434388671577, 68.71380121590303, 69.00247439204848, 69.23194656532482,
    69.43624694781325, 69.58108597167895, 69.69103260431739, 69.85452347167895,
    69.95188780397577, 69.98447087283263, 69.99934200273397
])

def NNoutput_to_DLR(Goutput, dlr_tau_mesh, sample_index=0):
    """Convert neural network output to DLR format.

    Args:
        Goutput (np.ndarray), ndim=(batchsize,ndlr_taus,2) or (batchsize, ndlr_taus*2): The output from the neural network.
        dlr_tau_mesh (triqs mesh): The mesh in imaginary time to use for the DLR representation, 
                     has to be the same as the network is trained on.
        sample_index (int): The index of the sample in the batch to convert, default is 0.

    Returns:
        BlockGf: The DLR representation of the input.
    """
    ndlr_taus = 0
    taus = []
    for tau in dlr_tau_mesh:
        taus.append(tau.value)
        ndlr_taus += 1
    assert(np.linalg.norm(np.array(taus) - mesh_saved) < 1e-6), "The dlr_tau_mesh provided does not match the mesh the network was trained on. Please provide the correct mesh or retrain the network on the desired mesh."
    
    if Goutput.shape[-1] != 2:
        assert(Goutput.shape[1] == ndlr_taus*2), "If the last dimension of Goutput is not 2, then the second dimension must be equal to ndlr_taus*2."
        Goutput = np.reshape(Goutput, (Goutput.shape[0], ndlr_taus, 2))
    Goutput = Goutput[sample_index,:,:]
    
    gf_struct = [('up', 1), ('down', 1)]
    Goutput_tau_dlr = BlockGf(mesh=dlr_tau_mesh, gf_struct=gf_struct)
    it = 0
    for tau in Goutput_tau_dlr.mesh:
        Goutput_tau_dlr['up'][tau][0,0]=Goutput[it,0] + 1j*Goutput[it,1]
        Goutput_tau_dlr['down'][tau][0,0]=Goutput[it,0] + 1j*Goutput[it,1]
        it += 1
    Goutput_dlr = make_gf_dlr(Goutput_tau_dlr)
    return Goutput_dlr

def DLR_to_NNinput(Delta_dlr, dlr_tau_mesh_loaded, U, mu, beta):
    """Convert DLR representation to neural network input format.

    Args:
        Delta_dlr (BlockGf): The DLR representation to convert.
        dlr_tau_mesh_loaded (triqs mesh): The mesh in imaginary time to use for the DLR representation.
        U (float): The interaction strength.
        mu (float): The chemical potential.
        beta (float): The inverse temperature.

    Returns:
        np.ndarray: The neural network input in the correct format.
    """
    taus = []
    for tau in dlr_tau_mesh_loaded:
        taus.append(tau.value)
    assert(np.linalg.norm(np.array(taus) - mesh_saved) < 1e-6)
    
    Delta_dlr = switch_mesh(Delta_dlr, 70.0, mesh_new=dlr_tau_mesh_loaded, beta_old=beta)
    Delta_tau_dlr = make_gf_dlr_imtime(Delta_dlr)
    Deltavals_NN = []
    for tau in Delta_tau_dlr['up'].mesh:
        Deltavals_NN.append(Delta_tau_dlr['up'][tau][0,0])
    Deltavals_NN = np.array(Deltavals_NN)
    Deltavals_NN = np.vstack((Deltavals_NN.real, Deltavals_NN.imag)).T 
    Deltavals_NN = np.expand_dims(Deltavals_NN, axis=0)
    Deltavals_NN = Deltavals_NN.reshape((Deltavals_NN.shape[0], Deltavals_NN.shape[1]*Deltavals_NN.shape[2]))
    nn_input = np.zeros((Deltavals_NN.shape[0], Deltavals_NN.shape[1]+3))
    nn_input[:,:-3] = Deltavals_NN[:,:] 
    nn_input[:,-3] = U
    nn_input[:,-2] = mu/U
    nn_input[:,-1] = beta

    return nn_input

def switch_mesh(gf_dlr, beta_new, mesh_new = None, beta_old=70.0, gf_struct=[('up', 1), ('down', 1)], wmax_val=10.0, eps_val=1e-13):
    """Switch the mesh of a Green's function from mesh_old to mesh_new, adjusting for different beta values.

    Args:
        gf (BlockGf): The Green's function to switch meshes for.
        mesh_old (triqs mesh): The original mesh of the Green's function.
        mesh_new (triqs mesh): The new mesh to switch to.
        beta_new (float): The beta value corresponding to the new mesh.
        beta_old (float): The beta value corresponding to the old mesh, default is 70.0.

    Returns:
        BlockGf: The Green's function with the new mesh.
    """
    if mesh_new is None:
       mesh_new = MeshDLRImTime(beta=beta_new, statistic='Fermion', w_max=wmax_val, eps=eps_val)
    gf_tau_dlr_new = BlockGf(mesh=mesh_new, gf_struct=gf_struct)
    for tau in mesh_new:
        gf_tau_dlr_new['up'][tau][0,0] = gf_dlr['up'](tau*beta_old/beta_new)[0,0]
        gf_tau_dlr_new['down'][tau][0,0] = gf_dlr['down'](tau*beta_old/beta_new)[0,0]
        
    gf_dlr_new = make_gf_dlr(gf_tau_dlr_new)
    return gf_dlr_new