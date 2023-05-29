import numpy as np
import torch, os
import numpy.linalg as lg
import random
from scipy.linalg import dft
import scipy.io as scio
# import matplotlib.pyplot as plt
from generate_channel import generate_channel, channel_complex2real, generate_channel_with_array_response

def batch_combine_channel(channel_bs_user_k, channel_irs_user_k, channel_bs_irs, phase_shifts):
    (num_sample, num_antenna_bs, num_elements_ir) = channel_bs_irs.shape
    len_pilots = phase_shifts.shape[1]

    channel_combine_irs = channel_bs_irs * channel_irs_user_k.reshape((num_sample, 1, num_elements_ir))
    channel_bs_user_k = np.repeat(channel_bs_user_k, len_pilots, axis=1)
    channel_combine = channel_bs_user_k.reshape((num_sample, num_antenna_bs, len_pilots)) \
                      + channel_combine_irs @ phase_shifts

    return channel_combine

def generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db, scale_factor=0, Pt=15):
    (channel_bs_user, channel_irs_user, channel_bs_irs) = channels
    (num_samples, num_antenna_bs, num_elements_irs) = channel_bs_irs.shape
    N_devices = channel_irs_user.shape[2]
    len_pilots = phase_shifts.shape[1]

    noise_sqrt = np.sqrt(10 ** ((noise_power_db - Pt + scale_factor) / 10))

    y = np.zeros((num_samples, num_antenna_bs, len_pilots), dtype=complex)
    for kk in range(N_devices):
        channel_bs_user_k = channel_bs_user[:, :, kk]
        channel_irs_user_k = channel_irs_user[:, :, kk]
        channel_combine = batch_combine_channel(channel_bs_user_k, channel_irs_user_k,
                                                channel_bs_irs, phase_shifts)
        pilots_k = pilots[:, kk]
        pilots_k = np.array([pilots_k] * num_samples)
        pilots_k = pilots_k.reshape((num_samples, 1, len_pilots))
        y = y + channel_combine * pilots_k

    noise = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples, num_antenna_bs, len_pilots]) \
            + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples, num_antenna_bs, len_pilots])
    y = y + noise_sqrt * noise

    y_real = np.concatenate([y.real, y.imag], axis=1)

    return np.array(y), np.array(y_real)

def channel_complex2real(channels):
    channel_bs_user, channel_irs_user, channel_bs_irs = channels
    (num_sample, num_antenna_bs, num_elements_irs) = channel_bs_irs.shape
    N_devices = channel_irs_user.shape[2]

    A_T_real = np.zeros([num_sample, 2 * num_elements_irs, 2 * num_antenna_bs, N_devices])
    # Hd_real = np.zeros([num_sample, 2 * num_antenna_bs, N_devices])
    set_channel_combine_irs = np.zeros([num_sample, num_antenna_bs, num_elements_irs, N_devices], dtype=complex)

    for kk in range(N_devices):
        channel_irs_user_k = channel_irs_user[:, :, kk]
        channel_combine_irs = channel_bs_irs * channel_irs_user_k.reshape(num_sample, 1, num_elements_irs)
        set_channel_combine_irs[:, :, :, kk] = channel_combine_irs
        A_tmp_tran = np.transpose(channel_combine_irs, (0, 2, 1))
        A_tmp_real1 = np.concatenate([A_tmp_tran.real, A_tmp_tran.imag], axis=2)
        A_tmp_real2 = np.concatenate([-A_tmp_tran.imag, A_tmp_tran.real], axis=2)
        A_tmp_real = np.concatenate([A_tmp_real1, A_tmp_real2], axis=1)
        A_T_real[:, :, :, kk] = A_tmp_real

    Hd_real = np.concatenate([channel_bs_user.real, channel_bs_user.imag], axis=1)

    return A_T_real, Hd_real, np.array(set_channel_combine_irs)

def compute_stat_info(params_system, noise_power_db, location_user, Rician_factor, num_samples=10000, pilot_power = 15):
    (num_antenna_bs, num_elements_irs, N_devices) = params_system
    len_pilot = N_devices * 1
    len_frame = N_devices
    phase_shifts, pilots = generate_pilots_bl(len_pilot, num_elements_irs, N_devices)
    channels, set_location_user = generate_channel(params_system,location_user_initial=location_user,
                                                   Rician_factor=Rician_factor, num_samples=num_samples)
    (channel_bs_user, channel_irs_user, channel_bs_irs) = channels
    _, _, channel_bs_irs_user = channel_complex2real(channels)
    y, _ = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db, Pt = pilot_power)
    Y = decorrelation(y, pilots)
    A, Hd, = channel_bs_irs_user, channel_bs_user

    ones = np.ones((1, len_pilot))
    phaseshifts_new = np.concatenate([ones, phase_shifts], axis=0)
    Q = phaseshifts_new[:, 0:len_pilot:len_frame]

    A, Hd, Y = A[:, :, :, 0], Hd[:, :, 0], Y[:, :, 0, :]
    A_h = np.concatenate((Hd.reshape(-1, num_antenna_bs, 1), A), axis=2)
    A = A_h

    mean_A, mean_Y = np.mean(A, axis=0, keepdims=True), np.mean(Y, axis=0, keepdims=True)
    # print(mean_Y - mean_A @ Q)
    A = A - mean_A
    C_A = np.sum(np.matmul(np.transpose(A.conjugate(), (0, 2, 1)), A), axis=0) / num_samples
    Y = Y - mean_Y
    # print(Y-A@Q)
    C_Y = np.sum(np.matmul(np.transpose(Y.conjugate(), (0, 2, 1)), Y), axis=0) / num_samples
    Q_H = np.transpose(Q.conjugate())
    C_N = C_Y - np.matmul(Q_H, np.matmul(C_A, Q))
    gamma_n = np.real(np.mean(np.diagonal(C_N)))
    stat_info = (gamma_n, C_A, mean_A)
    return stat_info

def decorrelation(received_pilots, pilots):
    (len_pilots, N_devices) = pilots.shape
    (num_samples, num_antenna_bs, _) = received_pilots.shape
    pilots = np.array([pilots] * num_samples)
    pilots = pilots.reshape((num_samples, len_pilots, N_devices))

    len_frame = N_devices
    num_frame = len_pilots // len_frame

    x_tmp = np.conjugate(pilots[:, 0:len_frame, :])
    y_decode = np.zeros([num_samples, num_antenna_bs, N_devices, num_frame], dtype=complex)
    for jj in range(num_frame):
        y_k = received_pilots[:, :, jj * len_frame:(jj + 1) * len_frame]
        y_decode_tmp = y_k @ x_tmp / len_frame
        y_decode[:, :, :, jj] = y_decode_tmp
    return y_decode

def channel_estimation_lmmse(params_system, y, pilots, phase_shifts, stat_info):
    (num_antenna_bs, num_elements_irs, N_devices) = params_system
    len_pilot = pilots.shape[0]
    num_sample = y.shape[0]

    len_frame = N_devices
    ones = np.ones((1, len_pilot))
    phaseshifts_new = np.concatenate([ones, phase_shifts], axis=0)
    Q = phaseshifts_new[:, 0:len_pilot:len_frame]

    (gamma_n, C_A, mean_A) = stat_info
    C_Y = np.matmul(np.matmul(np.transpose(Q.conjugate()), C_A), Q) + gamma_n * np.eye(Q.shape[1])
    mean_Y = np.matmul(mean_A, Q)

    y_d = decorrelation(y, pilots)
    channel_bs_user_est = np.zeros((num_sample, num_antenna_bs, N_devices), dtype=complex)
    channel_bs_irs_user_est = np.zeros((num_sample, num_antenna_bs, num_elements_irs, N_devices), dtype=complex)
    for kk in range(N_devices):
        y_k = y_d[:, :, kk, :]

        channel_est = lmmse_estimator(y_k, Q, C_A, C_Y, mean_A, mean_Y)
        channel_bs_user_est[:, :, kk] = channel_est[:, :, 0]
        channel_bs_irs_user_est[:, :, :, kk] = channel_est[:, :, 1:num_elements_irs + 1]
    return channel_bs_user_est, channel_bs_irs_user_est, y_d

def generate_pilots_bl(len_pilot, num_elements_irs, N_devices):
    len_frame = N_devices
    num_frame = len_pilot // len_frame
    if num_frame > num_elements_irs + 1:
        phase_shifts = dft(num_frame)
        phase_shifts = phase_shifts[0:num_elements_irs + 1, 0:num_frame]
    else:
        phase_shifts = dft(num_elements_irs + 1)
        phase_shifts = phase_shifts[0:num_elements_irs + 1, 0:num_frame]

    phase_shifts = np.repeat(phase_shifts, len_frame, axis=1)
    phase_shifts = np.delete(phase_shifts, 0, axis=0)

    pilots_subframe = dft(len_frame)
    pilots_subframe = pilots_subframe[:, 0:N_devices]
    pilots = np.array([pilots_subframe] * num_frame)
    pilots = np.reshape(pilots, [len_pilot, N_devices])
    # print('X^H * X:\n ', np.diagonal(np.matmul(np.conjugate(np.transpose(X)), X)), '\n')
    return phase_shifts, pilots

def test_channel_estimation_lmmse(params_system, len_pilot, noise_power_db, location_user, Rician_factor, num_sample, pilot_power = 15):
    (num_antenna_bs, num_elements_irs, N_devices) = params_system
    phase_shifts, pilots = generate_pilots_bl(len_pilot, num_elements_irs, N_devices)
    # phase_shifts, pilots = generate_pilots_bl_v2(len_pilot, num_elements_irs, N_devices)

    # print(phase_shifts, np.abs(phase_shifts))
    # print(pilots, '\n\n', np.diag(pilots @ np.transpose(pilots.conjugate())))
    channels, set_location_user = generate_channel(params_system,
                                                   num_samples=num_sample, location_user_initial=location_user,
                                                   Rician_factor=Rician_factor)
    (channel_bs_user, channel_irs_user, channel_bs_irs) = channels
    _, _, channel_bs_irs_user = channel_complex2real(channels)
    # y1, y1_r = generate_received_pilots(channels, phase_shifts, pilots, noise_power_db)
    y, y_real = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db, Pt = pilot_power)
    stat_info = compute_stat_info(params_system, noise_power_db, location_user, Rician_factor)

    # ===channel estimation===
    channel_bs_user_est, channel_bs_irs_user_est, y_decode = channel_estimation_lmmse(params_system, y, pilots, phase_shifts,stat_info)
    err_bs_user = np.linalg.norm(channel_bs_user_est - channel_bs_user, axis=(1))**2
    err_bs_irs_user = np.linalg.norm(channel_bs_irs_user_est - channel_bs_irs_user, axis=(1, 2))**2
    return np.mean(err_bs_user), np.mean(err_bs_irs_user)

def channel_estimation(params_system, len_pilot, noise_power_db, location_user, Rician_factor, num_sample, pilot_power = 15,max_iter = 20, dir = './DC/plot_data', location_bs=np.array([-80, 0, 30]), location_irs=np.array([10,0,10]), L_0 = -30, alpha = [3.6,2.5,2.2], Estimated = True):
    (num_antenna_bs, num_elements_irs, N_devices) = params_system
    Channel_data = np.zeros((max_iter, num_sample, N_devices, num_elements_irs+1),dtype = np.complex128)
    Channel_data_real = np.zeros((max_iter, num_sample, N_devices, num_elements_irs+1), dtype = np.complex128)
    num_frame = len_pilot//N_devices
    Received_pilots = np.zeros((max_iter, num_sample, N_devices*num_frame), dtype = np.complex128)
    for ii in range(max_iter):
        phase_shifts, pilots = generate_pilots_bl(len_pilot, num_elements_irs, N_devices)

        channels, set_location_user = generate_channel(params_system, location_bs=location_bs, location_irs=location_irs, num_samples=num_sample,location_user_initial=location_user,Rician_factor=Rician_factor, L_0 = L_0, alpha = alpha)
        
        (channel_bs_user, channel_irs_user, channel_bs_irs) = channels
        _, _, channel_bs_irs_user = channel_complex2real(channels)
        y, y_real = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db, Pt = pilot_power)
        y_decode = decorrelation(y, pilots)
        channel_bs_user = channel_bs_user.reshape(num_sample,N_devices,num_antenna_bs)
        channel_bs_irs_user = channel_bs_irs_user.reshape(num_sample,N_devices,num_elements_irs)
        combined_channel = np.concatenate((channel_bs_user, channel_bs_irs_user), axis=2)
        # y_decode = torch.tensor(y_decode,dtype = torch.complex128, device =0).reshape(num_sample, len_pilot)
        # combined_channel = torch.tensor(combined_channel,dtype = torch.complex128, device =0)
        real_channel = np.concatenate((channel_bs_user, channel_bs_irs_user), axis=2)
        Channel_data_real[ii, :, :, :] = real_channel
        Received_pilots[ii, :, :] = y_decode.reshape(num_sample,N_devices*num_frame)
        if Estimated:
            stat_info = compute_stat_info(params_system, noise_power_db, location_user, Rician_factor, num_samples = num_sample)
            # === channel estimation ===
            channel_bs_user_est, channel_bs_irs_user_est, y_decode = channel_estimation_lmmse(params_system, y, pilots, phase_shifts,stat_info)
            
            # === Reshape and Combine ===
            channel_bs_user_est = channel_bs_user_est.reshape(num_sample,N_devices,num_antenna_bs)
            channel_bs_irs_user_est = channel_bs_irs_user_est.reshape(num_sample,N_devices,num_elements_irs)
            estimated_channel = np.concatenate((channel_bs_user_est, channel_bs_irs_user_est), axis=2)
            Channel_data[ii,:,:,:] = estimated_channel      
    scio.savemat(dir + '/Channel_estimated_LMMSE_data_%d_%d_%d_%d_equalSNR_0.mat'%(N_devices,len_pilot,pilot_power,num_elements_irs),{'Channel_estimated_data_LMMSE':Channel_data})
    scio.savemat(dir + '/Channel_real_LMMSE_data_%d_%d_%d_%d_equalSNR_0.mat'%(N_devices,len_pilot,pilot_power,num_elements_irs),{'Channel_real_data_LMMSE':Channel_data_real})
    scio.savemat(dir + '/Received_pilots_%d_%d_%d_%d_equalSNR_0.mat'%(N_devices,len_pilot,pilot_power,num_elements_irs),{'Received_pilots':Received_pilots})

def channel_generation(params_system, len_pilot, noise_power_db, location_user, Rician_factor, num_sample, pilot_power = 15, location_bs=np.array([-80, 0, 30]), location_irs=np.array([10,0,10]), L_0 = -30, alpha = [3.6,2.5,2.2]):
    (num_antenna_bs, num_elements_irs, N_devices) = params_system
    phase_shifts, pilots = generate_pilots_bl(len_pilot, num_elements_irs, N_devices)

    channels, set_location_user = generate_channel(params_system,
                                                num_samples=num_sample, location_user_initial=location_user,location_irs=location_irs,
                                                Rician_factor=Rician_factor, L_0 = L_0, alpha = alpha)
    (channel_bs_user, channel_irs_user, channel_bs_irs) = channels
    channel_bs_user_, _, channel_bs_irs_user = channel_complex2real(channels)
    y, y_real = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db, Pt = pilot_power)
    y_decode = decorrelation(y, pilots)
    channel_bs_user = channel_bs_user.reshape(num_sample,N_devices,num_antenna_bs)
    channel_bs_irs_user = channel_bs_irs_user.reshape(num_sample,N_devices,num_elements_irs)
    combined_channel = np.concatenate((channel_bs_user, channel_bs_irs_user), axis=2)
    y_decode = torch.tensor(y_decode,dtype = torch.complex128, device =0).reshape(num_sample, len_pilot)
    combined_channel = torch.tensor(combined_channel,dtype = torch.complex128, device =0)
    return combined_channel, y_decode

def channel_generation_response(params_system, len_pilot, noise_power_db, location_user, Rician_factor, num_sample, pilot_power = 15, location_bs=np.array([-80, 0, 30]), location_irs=np.array([10,0,10]), L_0 = -30, alpha = [3.6,2.5,2.2]):
    (num_antenna_bs, num_elements_irs, N_devices) = params_system
    phase_shifts, pilots = generate_pilots_bl(len_pilot, num_elements_irs, N_devices)

    channels, set_location_user, array_response = generate_channel_with_array_response(params_system,
                                                num_samples=num_sample, location_user_initial=location_user,location_irs=location_irs,
                                                Rician_factor=Rician_factor, L_0 = L_0, alpha = alpha)
    (channel_bs_user, channel_irs_user, channel_bs_irs) = channels
    channel_bs_user_, _, channel_bs_irs_user = channel_complex2real(channels)
    y, y_real = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db, Pt = pilot_power)
    y_decode = decorrelation(y, pilots)
    channel_bs_user = channel_bs_user.reshape(num_sample,N_devices,num_antenna_bs)
    channel_bs_irs_user = channel_bs_irs_user.reshape(num_sample,N_devices,num_elements_irs)
    combined_channel = np.concatenate((channel_bs_user, channel_bs_irs_user), axis=2)
    y_decode = torch.tensor(y_decode,dtype = torch.complex128, device =0).reshape(num_sample, len_pilot)
    combined_channel = torch.tensor(combined_channel,dtype = torch.complex128, device =0)
    return combined_channel, y_decode, set_location_user, array_response

def lmmse_estimator(Y, Q, C_A, C_Y, mean_A, mean_Y):
    # # Y = AQ+N

    # ================================================
    # A = np.matmul(Y,np.linalg.inv(C_Y))
    # A = np.matmul(A,np.transpose(Q.conjugate()))
    # A = np.matmul(A,C_A)

    # ===============for numerical stability===========
    Y = Y - mean_Y
    Q_H = np.transpose(Q.conjugate())
    C_N = C_Y - np.matmul(Q_H, np.matmul(C_A, Q))
    gamma_n = np.real(np.mean(np.diagonal(C_N)))
    n, ell = Q.shape[0], Q.shape[1]
    if ell > n:
        QQ_H = np.matmul(Q, Q_H)
        C_A_inv = np.linalg.inv(C_A)
        tmp = np.linalg.inv(gamma_n * C_A_inv + QQ_H)
        tmp = np.matmul(tmp, QQ_H)
        tmp = np.matmul(C_A_inv, tmp)
        tmp = np.matmul(tmp, C_A)
        A = ls_estimator(Y, Q)
        A = np.matmul(A, tmp)
    else:
        tmp = np.matmul(Q_H, C_A)
        tmp = np.matmul(tmp, Q)
        tmp = tmp + gamma_n * np.eye(ell)
        tmp = np.linalg.inv(tmp)
        A = np.matmul(Y, tmp)
        A = np.matmul(A, Q_H)
        A = np.matmul(A, C_A)

    return A + mean_A

def ls_estimator(y, x):
    """
    y = h *x + n
    y: batch_size*m*l
    h: batch_size*m*n
    x: batch_size*n*l

    Output: h = y*x^H*(x*x^H)^-1
    """
    n, ell = x.shape[0], x.shape[1]
    x_H = np.transpose(x.conjugate())
    if ell < n:
        x_Hx = np.matmul(x_H, x)
        # print('Cond number:',np.linalg.cond(x_Hx))
        x_Hx_inv = np.linalg.inv(x_Hx)
        h = np.matmul(y, x_Hx_inv)
        h = np.matmul(h, x_H)
    elif ell == n:
        # print('Cond number:',np.linalg.cond(x))
        h = np.linalg.inv(x)
        h = np.matmul(y, h)
    else:
        xx_H = np.matmul(x, x_H)
        # print('Cond number:',np.linalg.cond(xx_H))
        xx_H_inv = np.linalg.inv(xx_H)
        h = np.matmul(y, x_H)
        h = np.matmul(h, xx_H_inv)
    return h
