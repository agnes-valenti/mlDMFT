import numpy as np
import pytest
from h5 import HDFArchive
from triqs.gf import MeshDLRImTime
from mldmft.utils import NNoutput_to_DLR, DLR_to_NNinput, switch_mesh #make_1d_to_2d, make_2d_to_1d

def test_dlr_nn():
    """Test the conversion between DLR and NN representations, by converting back and forth.

    Raises:
        TypeError: If the original and back-converted NN inputs do not match within a specified tolerance.
    """
    h5_mesh = '/mnt/home/avalenti/softw/mlDMFT/mlDMFT/mldmft/models/orb1/mesh_beta70.h5'
    with HDFArchive(h5_mesh ,'r') as h5:
        dlr_tau_mesh_loaded = h5['mesh_dlr_imtime']
    wmax_val = 10
    eps_val = 1e-13
    beta = np.random.rand()*70+10 
    u = np.random.rand()*7.0
    mu = np.random.rand()*u
    
    dlr_mesh_new = MeshDLRImTime(beta=beta, statistic='Fermion', w_max=wmax_val, eps=eps_val) 
    
    deltainput = "/mnt/home/avalenti/softw/mlDMFT/mlDMFT/mldmft/examples/1orbital/NN/example_inputs/inputs.npy" 
    deltainput = np.load(deltainput)
    if deltainput.ndim == 2:
        deltainput = np.expand_dims(deltainput, axis = 0)
    elif deltainput.ndim == 3 and deltainput.shape[0] > 1:
        deltainput = deltainput[0,:,:] # only first element used in case Deltainput is of larger size
        deltainput = np.expand_dims(deltainput, axis = 0)
    elif deltainput.ndim != 3:
        raise TypeError(f"Deltainput of invalid shape {deltainput.shape}. Expected shape: $(n_tau, 2)$ or $(m, n_tau, 2)$")

    deltainput_nn = np.reshape(deltainput, (deltainput.shape[0], deltainput.shape[1]*deltainput.shape[2]))
    delta_dlr = NNoutput_to_DLR(deltainput_nn, dlr_tau_mesh_loaded)
    delta_dlr = switch_mesh(delta_dlr, beta, mesh_new=dlr_mesh_new)
    deltainput_nn_compare = DLR_to_NNinput(delta_dlr, dlr_tau_mesh_loaded, u, mu, beta)[:,:-3]
    print("DeltainputNN_compare:", deltainput_nn_compare)
    print("DeltainputNN:", deltainput_nn)
    assert np.allclose(deltainput_nn, deltainput_nn_compare)

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

