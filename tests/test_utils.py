import numpy as np
import pytest
import mldmft
from mldmft.utils import make_1d_to_2d, make_2d_to_1d

def test_1d():
    n_orb = 2
    taus = np.linspace(0, 10, 5)
    data_2d = np.random.rand(len(taus), n_orb, n_orb) + 1j * np.random.rand(len(taus), n_orb, n_orb)
    data_1d = make_2d_to_1d(data_2d)
    data_1d_test = np.zeros_like(data_1d)
    data_1d_test[0, :, 0] = data_2d[:,0,0].real
    data_1d_test[0, :, 1] = data_2d[:,0,1].real
    data_1d_test[0, :, 2] = data_2d[:,1,1].real
    data_1d_test[0,:,3] = data_2d[:,0,0].imag
    data_1d_test[0,:,4] = data_2d[:,0,1].imag
    data_1d_test[0,:,5] = data_2d[:,1,1].imag
    assert np.allclose(data_1d, data_1d_test)

@pytest.mark.parametrize("n_orb", [2, 3])
def test_make_1d_to_2d_and_back(n_orb):
    taus = np.linspace(0, 10, 5)
    data_2d = np.random.rand(len(taus), n_orb, n_orb) + 1j * np.random.rand(len(taus), n_orb, n_orb)
    data_2d = data_2d + np.transpose(data_2d, (0,2,1))  # Make it symmetric
    data_1d = make_2d_to_1d(data_2d)
    test_2d = make_1d_to_2d(data_1d)
    assert np.allclose(data_2d, test_2d)




