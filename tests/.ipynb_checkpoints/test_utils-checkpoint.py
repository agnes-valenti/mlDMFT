import numpy as np
import pytest
from triqs.gf import *
import mldmft
from mldmft.utils import NNoutput_to_DLR, DLR_to_NNinput, switch_mesh #make_1d_to_2d, make_2d_to_1d

def test_dlr_NN():
    h5_mesh = '/mnt/home/avalenti/softw/mlDMFT/mlDMFT/mldmft/models/orb1/mesh_beta70.h5'
    with HDFArchive(h5_mesh ,'r') as h5:
        dlr_tau_mesh_loaded = h5['mesh_dlr_imtime']
    wmax_val = 10
    eps_val = 1e-13
    gf_struct_dummy = [('up', 1), ('down', 1)]
    beta = np.random.rand()*70+10 
    U = np.random.rand()*7.0
    mu = np.random.rand()*U
    
    dlr_mesh_new = MeshDLRImTime(beta=beta, statistic='Fermion', w_max=wmax_val, eps=eps_val) 
    
    Deltainput = "/mnt/home/avalenti/softw/mlDMFT/mlDMFT/mldmft/examples/1orbital/NN/example_inputs/inputs.npy" 
    Deltainput = np.load(Deltainput)
    if Deltainput.ndim == 2:
        Deltainput = np.expand_dims(Deltainput, axis = 0)
    elif Deltainput.ndim == 3 and Deltainput.shape[0] > 1:
        Deltainput = Deltainput[0,:,:] # only first element used in case Deltainput is of larger size
        Deltainput = np.expand_dims(Deltainput, axis = 0)
    elif Deltainput.ndim != 3:
        raise TypeError(f"Deltainput of invalid shape {Deltainput.shape}. Expected shape: $(n_tau, 2)$ or $(m, n_tau, 2)$")

    DeltainputNN = np.reshape(Deltainput, (Deltainput.shape[0], Deltainput.shape[1]*Deltainput.shape[2]))
    Delta_dlr = NNoutput_to_DLR(DeltainputNN, dlr_tau_mesh_loaded)
    Delta_dlr = switch_mesh(Delta_dlr, dlr_mesh_new, beta)
    DeltainputNN_compare = DLR_to_NNinput(Delta_dlr, dlr_tau_mesh_loaded, U, mu, beta)[:,-3]
    assert np.allclose(DeltainputNN, DeltainputNN_compare)
    
    
#def test_1d():
#    n_orb = 2
#    taus = np.linspace(0, 10, 5)
#    data_2d = np.random.rand(len(taus), n_orb, n_orb) + 1j * np.random.rand(len(taus), n_orb, n_orb)
#    data_1d = make_2d_to_1d(data_2d)
#    data_1d_test = np.zeros_like(data_1d)
#    data_1d_test[0, :, 0] = data_2d[:,0,0].real
#    data_1d_test[0, :, 1] = data_2d[:,0,1].real
#    data_1d_test[0, :, 2] = data_2d[:,1,1].real
#    data_1d_test[0,:,3] = data_2d[:,0,0].imag
#    data_1d_test[0,:,4] = data_2d[:,0,1].imag
#    data_1d_test[0,:,5] = data_2d[:,1,1].imag
#    assert np.allclose(data_1d, data_1d_test)

#@pytest.mark.parametrize("n_orb", [2, 3])
#def test_make_1d_to_2d_and_back(n_orb):
#    taus = np.linspace(0, 10, 5)
#    data_2d = np.random.rand(len(taus), n_orb, n_orb) + 1j * np.random.rand(len(taus), n_orb, n_orb)
#    data_2d = data_2d + np.transpose(data_2d, (0,2,1))  # Make it symmetric
#    data_1d = make_2d_to_1d(data_2d)
#    test_2d = make_1d_to_2d(data_1d)
#    assert np.allclose(data_2d, test_2d)






