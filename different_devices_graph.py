import numpy as np
import time, random, os
import torch
import scipy.io as scio
from GraphNet_share import Graph_net
from E2E_model_graph import update, Test
from utils import *
from LMMSE_estimator import *
from datetime import date
random.seed(1)

def various_devices():
    Results = {}
    NN = len(set_num_device)
    Results['mse_our'] = np.zeros(NN, )
    Results['mse_our_dual'] = np.zeros(NN, )
    Time_Cache = np.zeros((NN, 1))
    for N_devices in set_num_device:
        torch.cuda.empty_cache()
        params_system = (num_antenna_bs, N_IRS, N_devices)
        idx = set_num_device.index(N_devices)
        Pilots_Length = num_frame*N_devices
        print('Number of Devices :%d' % N_devices)
        print('---------')
        P_bar_t = [dbm2pw(27) for _ in range(N_devices)]
        # Generate channel and pilots
        skip_learning = False
        DNN = Graph_net(2, N_devices, N_IRS)
        if os.path.isfile(dir + '/DNN_dual_unequal_N_devices_%d_Len_pilots_%d_NIRS_%d_transmit_p_%d.pth' % (N_devices, Pilots_Length,N_IRS, pilot_power)):
            skip_learning = True
            DNN.load_state_dict(torch.load(dir + '/DNN_dual_unequal_N_devices_%d_Len_pilots_%d_NIRS_%d_transmit_p_%d.pth' % (N_devices,Pilots_Length, N_IRS, pilot_power)))
        else:
            torch.cuda.empty_cache()
            # Generate channel and pilots
            combined_channel, y_decode = channel_generation(params_system, num_frame*N_devices, noise_power_db, location_user, Rician_factor, num_sample = 1000, pilot_power = pilot_power, location_bs=location_bs, location_irs=location_irs, L_0 = L_0, alpha = alpha)
            for i in range(Max_iter-1):
                combined_channel_, y_decode_ = channel_generation(params_system, num_frame*N_devices, noise_power_db, location_user, Rician_factor, num_sample = 1000, pilot_power = pilot_power, location_bs=location_bs, location_irs=location_irs, L_0 = L_0, alpha = alpha)
                combined_channel = torch.cat((combined_channel, combined_channel_),0)
                y_decode = torch.cat((y_decode, y_decode_),0)
                combined_channel_, y_decode_ = combined_channel_.cpu(), y_decode_.cpu()
            # # Training
            torch.cuda.empty_cache()
            loss_r = update(DNN = DNN, training_data= [combined_channel, y_decode], training_loop = 1000, P_bar= P_bar_t, N_devices = N_devices, batch_size = 64, Training_size = Training_size, noise = noise_power_db, dir = dir, transmit_p = pilot_power)        
            combined_channel, y_decode = combined_channel.cpu(), y_decode.cpu()
        torch.cuda.empty_cache()
        # Testing    
        fq_E2E = np.zeros((Max_iter, Test_size, N_devices), dtype= np.complex)
        p_E2E = np.zeros((Max_iter, Test_size, N_devices))
        eta_E2E = np.zeros((Max_iter, Test_size))
        for iter in range(Max_iter):
            # Generate test channel and pilots
            combined_channel, y_decode = channel_generation(params_system, num_frame*N_devices, noise_power_db, location_user, Rician_factor, num_sample = Test_size, pilot_power = pilot_power,location_bs=location_bs, location_irs=location_irs, L_0 = L_0, alpha = alpha)
            t1 = time.time()
            loss_test_dual, [p, eta, fq, phase_shift] = Test(DNN=DNN, test_data= [combined_channel, y_decode], test_size = Test_size, P_bar = P_bar_t, N_devices = N_devices, test_loop = 1,  batch_size=128, noise = noise_power_db, dir = dir, save = True)
            t2 = time.time()
            fq_E2E[iter, :, :] = fq.squeeze()
            eta_E2E[iter, :] = eta
            p_E2E[iter, :, :] = p
            Results['mse_our'][idx] = Results['mse_our'][idx] + np.mean(loss_test_dual)
            print('Our:%e'%(np.mean(loss_test_dual)))
            combined_channel, y_decode = combined_channel.cpu(), y_decode.cpu()
            torch.cuda.empty_cache()
            Time_Cache[idx] += t2 - t1
        print('Our:%e'%(Results['mse_our'][idx]/Max_iter))
        scio.savemat(dir + '/E2E_fq_%d.mat'%N_devices,{'E2E_fq':fq_E2E})
        scio.savemat(dir + '/E2E_p_%d.mat'%N_devices,{'E2E_p':p_E2E})
        scio.savemat(dir + '/E2E_eta_%d.mat'%N_devices,{'E2E_eta':eta_E2E})
        scio.savemat(dir + '/mse_our_time_%d_devices.mat'%N_devices,{'mse_our_time':Time_Cache/(Max_iter*Test_size)})
    # Save results
    np.save(dir+'/Result_equal%d_devices.npy', Results)
    scio.savemat(dir + '/mse_our_devices.mat',{'mse_our':Results['mse_our']/Max_iter})
    return Results

num_antenna_bs, N_IRS, N_devices, num_sample, pilot_power = 1, 40, 10, 100, 10
noise_power_db, Rician_factor, L_0, num_frame, alpha = -70, 10, -25, 4, [4,1.8,2.1]
location_bs, location_irs, location_user = np.array([-200, 0, 30]), np.array([0,0,10]), None

Max_iter = 10
Training_size, Test_size = Max_iter*500, 100
set_num_device = [10]

t1 = time.time()
dir = './data/%s/Devices_%s_noise-%d'%(date.today().strftime('%Y-%m-%d'), str(time.strftime('%H_%M_%S')),noise_power_db)
dir = r'./data/2022-03-08/Devices_20_30_46_noise--70'
# dir = './test2'
if not os.path.exists(dir):
    os.makedirs(dir)
Result = various_devices()
print('Total Time :%.f'%(time.time() - t1))
    
