import torch,random
import torch.nn as nn
import numpy as np
import scipy.io as scio
from utils import *
import sys,time,os
from copy import deepcopy

torch.manual_seed(1)
random.seed(1)


def update(DNN, training_data, Training_size, P_bar, N_devices, training_loop=10000, batch_size=32, noise = -40, nu = 1, dir = '.',transmit_p = 1):
    """
    training_data is a tensor with the size of L\times N, each column is a channel vector
    N: the number of devices
    """
    t1 = time.time()
    noise_in_pw = dbm2pw(noise)
    combined_channel,received_pilots = training_data    
    len_pilots = received_pilots.shape[1]
    Training_size = Training_size
    loss_r = []
    mse_r = []
    loss_ave = []
    N_IRS = training_data[0][-1].size()[-1]-1
    Data_length, tot_frame = received_pilots[:Training_size,:].size()
    num_frame = tot_frame // N_devices
    Opt = torch.optim.Adam(DNN.parameters())
    for iter in range(training_loop):
        index = np.random.choice(max(Training_size,Data_length), size=(batch_size,))
        batch_combined_channel = combined_channel[:Training_size,:,:][index]
        batch_recived_pilots = received_pilots[:Training_size,:][index]
        input = (batch_combined_channel[:,:,0].view(batch_size, 1, N_devices) + batch_combined_channel[:,:,1:].sum(-1).view(batch_size, 1, N_devices))
        channel_compose = torch.cat((input.real, input.imag), 1).cuda()
        Net_output = DNN.forward(channel_compose).clone()
        
        phase_shift_real = torch.cos(Net_output[:,N_devices:N_devices+N_IRS]*np.pi*2)
        phase_shift_imag = torch.sin(Net_output[:,N_devices:N_devices+N_IRS]*np.pi*2)
        batch_channel_coefficient_real = batch_combined_channel[:,:,0].real.view(batch_size,N_devices, 1) + batch_combined_channel[:,:,1:].real@phase_shift_real.view(batch_size,N_IRS, 1) - batch_combined_channel[:,:,1:].imag@ phase_shift_imag.view(batch_size,N_IRS, 1)
        batch_channel_coefficient_imag = batch_combined_channel[:,:,0].imag.view(batch_size,N_devices, 1) + batch_combined_channel[:,:,1:].imag@phase_shift_real.view(batch_size, N_IRS, 1) + batch_combined_channel[:,:,1:].real@phase_shift_imag.view(batch_size, N_IRS, 1)
        
        batch_channel_coefficient = torch.sqrt(batch_channel_coefficient_real**2 + batch_channel_coefficient_imag**2).view(batch_size, N_devices)
        eta = Net_output[:,-1]*1e-9
        # Refresh Channel
        power = [Net_output[:,k].squeeze(-1) for k in range(N_devices)]
        regulazier = sum([nn.ReLU()(power[k].mean()-P_bar[k]) for k in range(N_devices)])

        signal_misalignment_error = sum([(torch.sqrt(power[j]) * batch_channel_coefficient[:,j]/ torch.sqrt(eta) - 1) ** 2 for j in range(N_devices)]) 
        noise_induced_error  = noise_in_pw/eta
        loss = 1 / (N_devices**2) * (noise_induced_error+signal_misalignment_error).mean() + nu*regulazier
        Opt.zero_grad()
        loss.backward()
        Opt.step()
        mse_r.append(1 / N_devices ** 2 * (noise_induced_error+signal_misalignment_error).mean().cpu().detach().numpy())
        loss_r.append(loss.cpu().detach().numpy())
        if len(loss_r)>200:
            loss_ave.append(np.mean(loss_r[-200:]))
        if (iter + 1) % 200 == 0:
            t2 = time.time()
            print("EP:%d|\t Loss:%e |\t time cost:%.2f|\t Loss Variances:%e" % (iter + 1, np.mean(loss_r[-200:]), t2-t1, torch.var(torch.log10(torch.tensor(np.asarray(mse_r[-200:]))))))
            sys.stdout.flush()
            t1 = deepcopy(t2)
        if len(loss_ave) > 200 and (iter + 1) % 50:
            if torch.var(torch.log10(torch.tensor(np.asarray(loss_ave[-200:]))))<1e-5:
                torch.save(DNN.state_dict(), dir + '/DNN_dual_unequal_N_devices_%d_Len_pilots_%d_NIRS_%d_transmit_p_%d.pth' % (N_devices,len_pilots,N_IRS,transmit_p))
                break
    torch.save(DNN.state_dict(), dir + '/DNN_dual_unequal_N_devices_%d_Len_pilots_%d_NIRS_%d_transmit_p_%d.pth' % (N_devices,len_pilots,N_IRS,transmit_p))
    print("Step:%d|\t Loss:%e |\t time cost:%.2f" % (iter + 1, np.mean(loss_r[-1000:]), t2-t1))
    torch.cuda.empty_cache()
    return loss_r

def Test(DNN, test_data, test_size, P_bar, N_devices, test_loop=10000, batch_size=32, noise = -80, save = True, dir = '.'):
    t1 = time.time()
    noise_in_pw = dbm2pw(noise)
    combined_channel,received_pilots = test_data    
    len_pilots = received_pilots.shape[1]
    loss_r = []
    mse_r = []
    loss_ave = []
    N_IRS = test_data[0][-1].shape[1]-1
    Data_length, tot_frame = received_pilots[:test_size,:].size()
    num_frame = tot_frame// N_devices
    prob = 0
    for iter in range(test_loop):
        index = np.random.choice(max(test_size,Data_length), size=(batch_size,))
        if save:
            Data_length = test_size
            index = np.asarray([i for i in range(Data_length)])
            batch_size = Data_length
        batch_combined_channel = combined_channel[:test_size,:,:][index]
        batch_recived_pilots = received_pilots[:test_size,:][index]
        input = (batch_combined_channel[:,:,0].view(batch_size, 1, N_devices) + batch_combined_channel[:,:,1:].sum(-1).view(batch_size, 1, N_devices))
        channel_compose = torch.cat((input.real, input.imag), 1).cuda()
        Net_output = DNN.forward(channel_compose).clone()

        phase_shift_real = torch.cos(Net_output[:,N_devices:N_devices+N_IRS]*np.pi*2)
        phase_shift_imag = torch.sin(Net_output[:,N_devices:N_devices+N_IRS]*np.pi*2)
        batch_channel_coefficient_real = batch_combined_channel[:,:,0].real.view(batch_size,N_devices, 1) + torch.bmm(batch_combined_channel[:,:,1:].real, phase_shift_real.view(batch_size,N_IRS, 1)) - torch.bmm(batch_combined_channel[:,:,1:].imag, phase_shift_imag.view(batch_size,N_IRS, 1))
        batch_channel_coefficient_imag = batch_combined_channel[:,:,0].imag.view(batch_size,N_devices, 1) + torch.bmm(batch_combined_channel[:,:,1:].imag, phase_shift_real.view(batch_size, N_IRS, 1)) + torch.bmm(batch_combined_channel[:,:,1:].real, phase_shift_imag.view(batch_size, N_IRS, 1))
        batch_channel_coefficient = torch.sqrt(batch_channel_coefficient_real**2 + batch_channel_coefficient_imag**2).view(batch_size, N_devices)
        eta = Net_output[:,-1]*1e-9
        # Refresh Channel
        power = power = [Net_output[:,k].squeeze(-1) for k in range(N_devices)]
        signal_misalignment_error = sum([(torch.sqrt(power[j]) * batch_channel_coefficient[:,j]/ torch.sqrt(eta) - 1) ** 2 for j in range(N_devices)]) 
        """
        eta = Net_output[:,-1]*1e-9
        # Refresh Channel
        power = [Net_output[:,k].squeeze(-1)**2*(torch.cos(Net_output[:,N_devices+k]*np.pi*2)**2 + torch.sin(Net_output[:,k]*np.pi*2)**2) for k in range(N_devices)]
        regulazier = sum([nn.ReLU()(power[k].mean()-P_bar[k]) for k in range(N_devices)])
        
        # signal_misalignment_error = sum([(torch.sqrt(power[j]) * batch_channel_coefficient[:,j]/ torch.sqrt(eta) - 1) ** 2 for j in range(N_devices)])
        signal_misalignment_error = sum([((Net_output[:,k].squeeze(-1)*torch.cos(Net_output[:,N_devices+k]*np.pi*2)*batch_channel_coefficient_real[:,k] - Net_output[:,k].squeeze(-1)*torch.sin(Net_output[:,k]*np.pi*2)*batch_channel_coefficient_imag[:,k])/torch.sqrt(eta) - 1)**2 - (Net_output[:,k].squeeze(-1)*torch.cos(Net_output[:,N_devices+k]*np.pi*2)*batch_channel_coefficient_imag[:,k] + Net_output[:,k].squeeze(-1)*torch.sin(Net_output[:,k]*np.pi*2)*batch_channel_coefficient_real[:,k]) for k in range(N_devices)])
        """ 
        noise_induced_error  = noise_in_pw/eta
        prob = sum([((nn.ReLU(inplace=True)(power[k].mean()-P_bar[k])>5e-3)*1) for k in range(N_devices)])+prob
        loss = 1 / N_devices ** 2 * (noise_induced_error+signal_misalignment_error).mean()
        loss_r.append(loss.cpu().detach().numpy())
    scio.savemat('./Prob_%d.mat'%N_devices, {'prob':prob.detach().cpu().numpy()})
    p = np.zeros((Data_length, N_devices))
    phase_shift = phase_shift_real + 1j*phase_shift_imag
    print(prob)
    for nd in range(N_devices):
        p[:,nd] = np.asarray(power[nd].detach().cpu().numpy())
    torch.cuda.empty_cache()
    return loss_r, [p, eta.detach().cpu().numpy(), (batch_channel_coefficient_real+batch_channel_coefficient_imag*1j).detach().cpu().numpy(), phase_shift]
