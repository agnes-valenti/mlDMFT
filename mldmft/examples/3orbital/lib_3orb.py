import numpy as np
from triqs.gf import *
from triqs.gf.tools import *
from triqs.gf.meshes import MeshImFreq
from triqs.gf import SemiCircular


def build_initial_delta(beta, half_bandwidth, gf_struct, n_iw=1025):
    """Construct semicircular hybridization Δ(iω) and Δ(τ)."""

    n_tau = 10 * n_iw

    mesh_imfreq = MeshImFreq(beta=beta, S="Fermion", n_iw=n_iw)

    Delta_iw = BlockGf(mesh=mesh_imfreq, gf_struct=gf_struct)
    for bl, g in Delta_iw:
        Delta_iw[bl] << (half_bandwidth / 2.0) ** 2 * SemiCircular(half_bandwidth)

    Delta_tau = BlockGf(
        mesh=MeshImTime(beta=beta, S="Fermion", n_tau=n_tau),
        gf_struct=gf_struct
    )

    for block, delta in Delta_tau:
        Delta_tau[block] << make_gf_from_fourier(
            Delta_iw[block],
            Delta_tau[block].mesh,
            fit_hermitian_tail(
                Delta_iw[block],
                make_zero_tail(Delta_iw[block], 1)
            )[0]
        )

    return Delta_tau


def project_to_nn_mesh(Delta_tau, mesh, gf_struct, wmax, eps, pairs):
    """
    Project Δ(τ) onto the fixed DLR mesh used during NN training.
    """

    Delta_dlr = fit_gf_dlr(Delta_tau, w_max=wmax, eps=eps)
    Delta_dlr_imtime = make_gf_dlr_imtime(Delta_dlr)

    dlr_mesh = Delta_dlr_imtime.mesh

    delta_nn_mesh = BlockGf(mesh=mesh, gf_struct=gf_struct)

    for tau_new, tau in zip(mesh, dlr_mesh):
        for (i, j) in pairs:
            delta_nn_mesh['up'][tau_new][i, j] = Delta_dlr['up'](tau)[i, j]
            delta_nn_mesh['down'][tau_new][i, j] = Delta_dlr['down'](tau)[i, j]

    return delta_nn_mesh, Delta_dlr_imtime


def blockgf_to_nn_input(Delta_dlr_imtime, mesh, pairs):
    """
    Flatten Δ(τ) upper triangle into NN input format.
    """

    ntau = len(mesh)
    ntri = len(pairs)

    train_data = np.zeros((1, ntau, 2 * ntri))

    for s, (i, j) in enumerate(pairs):

        arr = np.zeros((1, ntau), dtype=complex)
        data = Delta_dlr_imtime['up'].data[:, i, j]

        for it in range(ntau):
            arr[0, it] = data[it]

        train_data[:, :, s] = np.real(arr)
        train_data[:, :, s + ntri] = np.imag(arr)

    train_data = np.reshape(train_data, (1, ntau * 2 * ntri))

    return train_data


def build_nn_feature_vector(train_data, U, mu, J):
    """
    Append interaction parameters to NN feature vector.
    """

    batchsize, n = train_data.shape

    NNinput = np.zeros((batchsize, n + 3))
    NNinput[:, :-3] = train_data
    NNinput[:, -3] = U
    NNinput[:, -2] = mu / U
    NNinput[:, -1] = J

    return NNinput


def nn_output_to_channel_array(pred_flat, ntau, ntri):
    """
    Convert flat NN output back to shape (ntau, 2*ntri).

    Per tau ordering is:
    [Re(pair0..pair_{ntri-1}), Im(pair0..pair_{ntri-1})]
    """
    pred_flat = np.asarray(pred_flat)

    if pred_flat.ndim == 2 and pred_flat.shape[0] == 1:
        pred_flat = pred_flat[0]

    if pred_flat.ndim != 1:
        raise ValueError(f"Expected flat array of shape (ntau*2*ntri,), got {pred_flat.shape}")

    expected = ntau * 2 * ntri
    if pred_flat.size != expected:
        raise ValueError(f"Expected size {expected}, got {pred_flat.size}")

    return pred_flat.reshape(ntau, 2 * ntri)


def nn_channels_to_blockgf(channel_array, template_bgf, pairs, fill_lower_triangle=True):
    """
    Convert channel array of shape (ntau, 2*ntri) back to BlockGf.

    Per tau ordering is:
    [Re(pair0..pair_{ntri-1}), Im(pair0..pair_{ntri-1})]
    """
    channel_array = np.asarray(channel_array)

    if channel_array.ndim != 2:
        raise ValueError(f"channel_array must have shape (ntau, 2*ntri), got {channel_array.shape}")

    ntau, nchan = channel_array.shape
    ntri = len(pairs)

    if nchan != 2 * ntri:
        raise ValueError(f"Expected second dimension {2*ntri}, got {nchan}")

    real_part = channel_array[:, :ntri]
    imag_part = channel_array[:, ntri:]

    for spin in ['up', 'down']:
        g = template_bgf[spin]
        norb = g.target_shape[0]

        if len(g.mesh) != ntau:
            raise ValueError(f"len(mesh)={len(g.mesh)} != ntau={ntau}")

        for t_idx, tau in enumerate(g.mesh):
            for i in range(norb):
                for j in range(norb):
                    g[tau][i, j] = 0.0 + 0.0j

            for ch, (i, j) in enumerate(pairs):
                val = real_part[t_idx, ch] + 1j * imag_part[t_idx, ch]
                g[tau][i, j] = val

                if fill_lower_triangle and i != j:
                    g[tau][j, i] = np.conjugate(val)

    return template_bgf

