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

def make_1d_to_2d(data_1d):
    lentaus = np.shape(data_1d)[1]
    n_orb = np.sqrt(0.25+np.shape(data_1d)[2])-0.5
    assert abs(n_orb - round(n_orb)) < 1e-15, "n_orb is not close to an integer within the tolerance"
    n_orb = int(round(n_orb))
    data_2d =np.zeros((lentaus, n_orb, n_orb), dtype=np.complex128)
    it = 0
    for i in range(n_orb):
        for j in range(i+1):
            data_2d[:,j,i] = data_1d[0, :, it] + 1j *data_1d[0, :, it+n_orb*(n_orb+1)//2]
            it += 1
            if i != j:
                data_2d[:,i,j] = data_2d[:,j,i]
    return data_2d

def make_2d_to_1d(data_2d):
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

def loaded_to_DLR(Goutput_real, Goutput_imag, lentaus, gf_struct, beta=10, wmax_val=10, eps_val=1e-13):
    n_orb = gf_struct[0][1]
    Goutput = np.concatenate((Goutput_real, Goutput_imag), axis=2)
    Goutput = np.reshape(Goutput, (Goutput.shape[0], lentaus, n_orb*n_orb + n_orb))
    Goutput = make_1d_to_2d(Goutput) 
    dlr_iw_mesh = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=wmax_val, eps=eps_val)
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


def Delta_to_G0(Goutput, Deltatau_initshape, taus, muval, cfield_splitting, gf_struct, n_orb, wmax_val=10, eps_val=1e-13, beta=10, n_iw=1025, n_tau=10001):
    G0_vals = np.zeros_like(Deltatau_initshape)
    Goutput = Goutput[:,:-4]
    Goutput = np.reshape(Goutput, (Goutput.shape[0], len(taus), 3*2))
    Deltatau_up_vals = make_1d_to_2d(Goutput)
    
    dlr_iw_mesh = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=wmax_val, eps=eps_val)
    test_iw_dlr = BlockGf(mesh=dlr_iw_mesh, gf_struct=gf_struct)
    test_dlr = make_gf_dlr(test_iw_dlr)
    Deltatau_dlr = make_gf_dlr_imtime(test_dlr)
    it = 0
    for tau in Deltatau_dlr.mesh:
        Deltatau_dlr['up'][tau][:,:]=Deltatau_up_vals[it,:,:]
        Deltatau_dlr['down'][tau][:,:]=Deltatau_up_vals[it,:,:]
        it += 1
    #plot_from_triqs(Deltatau_dlr, "Delta_")
  
    Delta_dlr = make_gf_dlr(Deltatau_dlr)
    Delta_omega = make_gf_imfreq(Delta_dlr, n_iw = n_iw)
    #test_delta = make_gf_from_fourier(Delta_omega, n_tau = n_tau)
    
    G0_iw = BlockGf(mesh=Delta_omega.mesh, gf_struct=gf_struct)
    for name, g0 in G0_iw: 
        h0 = -muval*np.eye(2) + cfield_splitting*np.array([[1,0],[0,-1]]) 
        g0 << inverse(iOmega_n -h0 - Delta_omega[name]) #inverse(iOmega_n +mu - Delta_omega[name])
        
    G0_tau = make_gf_from_fourier(G0_iw, n_tau=n_tau)
    #plt.figure()
    #for name, g in G0_tau:
    #    oplot(g, '-', label=name)
    #plt.savefig("Delta_dlr.png") 

    G0_dlr = fit_gf_dlr(G0_tau, w_max = 10, eps=1e-13)
    Gtau_dlr = make_gf_dlr_imtime(G0_dlr)

    Deltatau_up_vals_output = np.zeros((len(taus), n_orb, n_orb), dtype=np.complex128)
    it = 0
    for tau in Gtau_dlr.mesh:
        Deltatau_up_vals_output[it,:,:]=Gtau_dlr['up'][tau][:,:]
        it += 1

    Deltatau_vals_output = np.zeros((len(taus), 3*2))
    Deltatau_vals_output[:,0] = Deltatau_up_vals_output[:,0,0].real
    Deltatau_vals_output[:,1] = Deltatau_up_vals_output[:,0,1].real
    Deltatau_vals_output[:,2] = Deltatau_up_vals_output[:,1,1].real
    Deltatau_vals_output[:,3] = Deltatau_up_vals_output[:,0,0].imag
    Deltatau_vals_output[:,4] = Deltatau_up_vals_output[:,0,1].imag
    Deltatau_vals_output[:,5] = Deltatau_up_vals_output[:,1,1].imag

    Deltatau_vals_output = np.reshape(Deltatau_vals_output, np.shape(Deltatau_vals_output)[0]*np.shape(Deltatau_vals_output)[1])

    G0_vals[0, :-4] = Deltatau_vals_output[:]
    G0_vals[0, -4] = Deltatau_initshape[0, -4]
    G0_vals[0, -3] = Deltatau_initshape[0, -3]
    G0_vals[0, -2] = Deltatau_initshape[0, -2]
    G0_vals[0, -1] = Deltatau_initshape[0, -1] 
    return G0_vals

def get_Delta(folder, filename_real, filename_imag, U_, mu_, J_, cfield_):
    train_data_real = np.load(os.path.join(folder, "tau", filename_real)) #"G_tau_real_test.npy"))
    train_data_imag = np.load(os.path.join(folder, "tau", filename_imag)) #G_tau_imag_test.npy"))
    train_data = make_2d_to_1d(train_data_real[nv,:,0,:,:] + 1j*train_data_imag[nv,:,0,:,:])
    test = make_1d_to_2d(train_data)
    train_data_test = make_2d_to_1d(test)
    print("Check: ", np.allclose(train_data, train_data_test))
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
    train_data_1 = np.zeros((np.shape(train_data)[0], np.shape(train_data)[1]+4))
    train_data_1[:, :-4] = train_data
    Umuvals = None
        
    #Umuvals = np.load(os.path.join(folder, "Umu_test.npy"))[nv]
    Umuvals = np.load(os.path.join(folder, "Umu_train.npy"))[nv]
    #print("Umuvals shape: ", Umuvals.shape)
    #print(Umuvals)
    #U_ = Umuvals[0]
    #mu_ = Umuvals[1]
    #J_ = Umuvals[2]
    #cfield_ = Umuvals[3]

    train_data_1[:, -4] = U_

    train_data_1[:, -3] = mu_/U_
    train_data_1[:, -2] = J_ #Umuvals[:,0]
    train_data_1[:, -1] = cfield_

    Deltatau = np.zeros((1, train_data_1.shape[1]))
    Deltatau[0,:] = train_data_1[0,:]
    return Deltatau, U_, mu_, J_, cfield_

def save_data(data_, filename_):
    data_ = data_[:,:-4]
    data_ = np.reshape(data_, (data_.shape[0], -1, 3*2))
    np.save(filename_, data_)

def plot_from_triqs(triqs_gf, filename_):
    fig, axs = plt.subplots(2, 2, figsize=(5, 5), dpi=300)
    taus_ = triqs_gf.mesh
    it = 0
    for name, g in triqs_gf:
      if 'up' in name:
        tauvals_ = []
        for tau in taus_:
            #print("g: " ,g[tau][0,0])
            tauvals_.append(tau.value)
        for a in range(2):
            for b in range(2):
                gvals_ = []
                for tau in taus_:
                    gvals_.append(g[tau][a,b])
                gvals_ = np.array(gvals_)
                alphavals = 1.0 #1.0*(it+1)/(maxit+1)
                #axs[a,b].plot(tauvfullvals, gfullvals.real, label=name + '_full')
                axs[a,b].plot(tauvals_, gvals_.real, alpha = alphavals, color='black',label=name + '_dlr')
    plt.savefig(filename_ + "_plotfromtriqs.png")
