import numpy as np
import pytest
from h5 import HDFArchive
from triqs.gf import MeshDLRImTime, BlockGf, SemiCircular, make_gf_dlr, make_gf_dlr_imfreq, make_gf_dlr_imtime
from mldmft.utils_2orbs import NNoutput_to_DLR, DLR_to_NNinput, _make_1d_to_2d, _make_2d_to_1d

with HDFArchive(f'mldmft/models/orb2/mesh_beta50.0.h5' ,'r') as h5:    
    DLR_MESH_2ORBS = h5['mesh_dlr_imtime']

def test_dlr_nn():
    """Test the conversion between DLR and NN representations, by converting back and forth.

    Raises:
        TypeError: If the original and back-converted NN inputs do not match within a specified tolerance.
    """
    np.random.seed(0) # for reproducibility
    u = np.random.rand()*7.0
    mu = np.random.rand()*u
    jval = np.random.rand()*u*0.3
    cfield = np.random.rand()*2.0 
    gf_struct = [('up', 2), ('down', 2)]
    delta_imtime = BlockGf(mesh=DLR_MESH_2ORBS, gf_struct=gf_struct)
    delta_dlr = make_gf_dlr(delta_imtime)
    delta_imfreq = make_gf_dlr_imfreq(delta_dlr)
    D = 1
    delta_imfreq["up"] << SemiCircular(D)
    delta_imfreq["down"] << SemiCircular(D)

    delta_dlr = make_gf_dlr(delta_imfreq)
    NNinput = DLR_to_NNinput(delta_dlr, DLR_MESH_2ORBS, u, mu, jval, cfield)
    NN_test = NNinput[:,:-4]
    NN_test = np.reshape(NN_test, (1,-1, 6))
    delta_dlr_compare = NNoutput_to_DLR(NN_test, DLR_MESH_2ORBS)
    
    delta_dlr_list = []
    delta_dlr_compare_list = []
    delta_tau_dlr = make_gf_dlr_imtime(delta_dlr)
    delta_tau_dlr_compare = make_gf_dlr_imtime(delta_dlr_compare)
    for tau in DLR_MESH_2ORBS:
        for i in range(2):
            for j in range(2):
                delta_dlr_list.append(delta_tau_dlr["up"][tau][i,j])
                delta_dlr_compare_list.append(delta_tau_dlr_compare["up"][tau][i,j])
    delta_dlr_array = np.array(delta_dlr_list)
    delta_dlr_compare_array = np.array(delta_dlr_compare_list)

    assert np.allclose(delta_dlr_array, delta_dlr_compare_array, atol=1e-4)

def test_1d():
    n_orb = 2
    taus = np.linspace(0, 10, 5)
    data_2d = np.random.rand(len(taus), n_orb, n_orb) + 1j * np.random.rand(len(taus), n_orb, n_orb)
    data_1d = _make_2d_to_1d(data_2d)
    data_1d_test = np.zeros_like(data_1d)
    data_1d_test[0, :, 0] = data_2d[:,0,0].real
    data_1d_test[0, :, 1] = data_2d[:,0,1].real
    data_1d_test[0, :, 2] = data_2d[:,1,1].real
    data_1d_test[0,:,3] = data_2d[:,0,0].imag
    data_1d_test[0,:,4] = data_2d[:,0,1].imag
    data_1d_test[0,:,5] = data_2d[:,1,1].imag
    assert np.allclose(data_1d, data_1d_test)

@pytest.mark.parametrize("n_orb", [2,3])
def test_make_1d_to_2d_and_back(n_orb):
    taus = np.linspace(0, 10, 5)
    data_2d = np.random.rand(len(taus), n_orb, n_orb) + 1j * np.random.rand(len(taus), n_orb, n_orb)
    data_2d = data_2d + np.transpose(data_2d, (0,2,1))  # Make it symmetric
    data_1d = _make_2d_to_1d(data_2d, n_orb=n_orb)
    test_2d = _make_1d_to_2d(data_1d, n_orb=n_orb)
    assert np.allclose(data_2d, test_2d)

