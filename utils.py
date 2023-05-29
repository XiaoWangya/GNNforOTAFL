import numpy as np
import torch
import numpy.linalg as lg
import random
from scipy.linalg import dft
# import matplotlib.pyplot as plt

torch.manual_seed(1)
random.seed(12)

def dbm2pw(x):
    return 10**((x-30)/10)

def pw2dbm(p):
    return 30+10*np.log10(p)

def db2pow(ydb):
    return 10**(ydb/10)

def Cal_Line_of_sight(Device_set,RIS_loc,FC_loc,N_IRS):
    num_user = len(Device_set)
    H_irs_user = np.zeros((num_user,N_IRS),dtype= np.complex128)
    for i in range(num_user):
        dis = lg.norm(np.asarray(Device_set[i])- np.asarray(RIS_loc))
        for jj in range(N_IRS):
            i1 = jj//10
            i2 = jj%10
            ag1 = (Device_set[i][1] - RIS_loc[1])/dis
            ag2 = (Device_set[i][2] - RIS_loc[2])/dis
            H_irs_user[i, jj] = np.exp((i1*ag1+i2*ag2)*1j)
    dis2 = lg.norm(np.asarray(FC_loc)- np.asarray(RIS_loc))
    ag3= (FC_loc[1] - RIS_loc[1])/dis2
    ag4 = (FC_loc[2] - RIS_loc[2])/dis2
    H_fc_irs = np.zeros((N_IRS,),dtype= np.complex128)
    for jj in range(N_IRS):
        i3 = jj//10
        i4 = jj%10
        H_fc_irs[jj] = np.exp((i3*ag3+i4*ag4)*1j)
    return H_irs_user, H_fc_irs

def Channel_generation(N_devices, N_IRS, RIS_loc, FC_loc, FC_RIS_dis, L_0, alpha,len_sample,K = 8):
    Device_set = []
    # Determining Position
    dis_matrix = torch.zeros(size = (N_devices,2),device =0)
    # Generate fading channel
    for _ in range(N_devices):
        angle = np.random.rand()*2*np.pi
        Device_set.append([np.cos(angle)*5+20, np.sin(angle)*5-20, 0])
        dis_matrix[_,0] = lg.norm(np.asarray(Device_set[-1])- np.asarray(FC_loc))#distance between devices and FC
        dis_matrix[_,1] = lg.norm(np.asarray(Device_set[-1])- np.asarray(RIS_loc))#distance between devices and RIS
    H_irs_user, H_fc_irs = Cal_Line_of_sight(Device_set,RIS_loc,FC_loc,N_IRS)

    Channel_sequence = (torch.normal(mean=0., std=torch.tensor(np.sqrt(2)/2)*torch.ones(len_sample, (N_devices+1)*N_IRS+N_devices))*1j+torch.normal(mean=0., std=torch.tensor(np.sqrt(2)/2)*torch.ones(len_sample, (N_devices+1)*N_IRS+N_devices))).cuda()# N_devices+N_device*N_IRS+N_IRS
    # Channel_sequence[:,N_devices:] = Channel_sequence[:,N_devices:]*np.sqrt(1/(1+K))+np.sqrt(K/(1+K))*1
    for j in range(N_devices):
        Channel_sequence[:, j] =  Channel_sequence[:, j]*db2pow(L_0)*torch.sqrt((dis_matrix[j, 0])**(-alpha[0]))# FC-User
        Channel_sequence[:, N_devices+j*N_IRS:N_devices+(j+1)*N_IRS] = Channel_sequence[:, N_devices+j*N_IRS:N_devices+(j+1)*N_IRS]*np.sqrt(1/(1+K)) + torch.tensor(np.sqrt(K/(1+K))*H_irs_user[j,:],dtype = torch.complex128, device =0)
        Channel_sequence[:, N_devices+j*N_IRS:N_devices+(j+1)*N_IRS] =  Channel_sequence[:, N_devices+j*N_IRS:N_devices+(j+1)*N_IRS]*db2pow(L_0)*torch.sqrt((dis_matrix[j, 1])**(-alpha[1]))# FC-IRS
    for jj in range(N_IRS):
        if -N_IRS+jj+1 == 0:
            Channel_sequence[:, -N_IRS+jj:] = Channel_sequence[:, -N_IRS+jj:]*np.sqrt(1/(1+K)) + torch.tensor(np.sqrt(K/(1+K))*H_fc_irs[-1],dtype = torch.complex128, device =0)
            Channel_sequence[:, -N_IRS+jj:] =  Channel_sequence[:, -N_IRS+jj:]*db2pow(L_0)*np.sqrt((FC_RIS_dis)**(-alpha[2]))
        else:
            Channel_sequence[:, -N_IRS+jj:] = Channel_sequence[:, -N_IRS+jj:]*np.sqrt(1/(1+K)) + torch.tensor(np.sqrt(K/(1+K))*H_fc_irs[-N_IRS+jj:],dtype = torch.complex128, device =0)
            Channel_sequence[:, -N_IRS+jj:-N_IRS+jj+1] =  Channel_sequence[:, -N_IRS+jj:-N_IRS+jj+1]*db2pow(L_0)*np.sqrt((FC_RIS_dis)**(-alpha[2]))
    Channel_FC_UE = Channel_sequence[:,:N_devices]
    Channel_IRS_UE = Channel_sequence[:, N_devices:N_devices+N_devices*N_IRS]
    Channel_FC_IRS = Channel_sequence[:,-N_IRS:]
    cascaded_channel = torch.zeros(size = (len_sample, N_devices, N_IRS),dtype = torch.complex128, device =0)
    full_channel = torch.zeros(size = (len_sample, N_devices, N_IRS+1),dtype = torch.complex128, device =0)
    for jjj in range(N_devices):
        cascaded_channel[:,jjj,:] = Channel_sequence[:, N_devices+jjj*N_IRS:N_devices+(jjj+1)*N_IRS]*Channel_sequence[:, -N_IRS:]
        full_channel[:,jjj,:] = torch.cat((Channel_FC_UE[:,jjj].unsqueeze(-1),cascaded_channel[:,jjj,:]),1)
    return [Channel_FC_UE,Channel_FC_IRS,Channel_IRS_UE,cascaded_channel,full_channel]

def Generate_pilots(N_devices, pilots_length, len_sample, transmit_p = 1):
    len_frame = N_devices
    num_frame = pilots_length // len_frame
    power_in_watt = dbm2pw(transmit_p)
    # Generate orthogonal pilots
    pilots_subframe = dft(len_frame)*np.sqrt(power_in_watt)
    pilots_subframe = pilots_subframe[:, 0:N_devices]
    pilots = np.array([pilots_subframe] * num_frame)
    pilots = torch.tensor(np.reshape(pilots, [pilots_length, N_devices]),dtype = torch.complex128, device =0)
    return pilots

def Phaseshift_Generation(N_devices,N_IRS,pilots_length):
    len_frame = N_devices
    num_frame = pilots_length // len_frame
    if num_frame > N_IRS + 1:
        phase_shifts = dft(num_frame)
        phase_shifts = phase_shifts[0:N_IRS + 1, 0:num_frame]
    else:
        phase_shifts = dft(N_IRS + 1)
        phase_shifts = phase_shifts[0:N_IRS + 1, 0:num_frame]

    return phase_shifts

def Received_pilots_generation(phase_shifts,pilots,len_sample,N_devices,N_IRS,pilots_length,Channel_sequence,noise = -85):
    len_frame = N_devices
    num_frame = pilots_length // len_frame
    # Recived Pilots
    # phase_shifts[1:N_IRS + 1, 0:num_frame] = 0
    phase_shifts = torch.tensor(phase_shifts,dtype = torch.complex128, device =0)
    Received_pilots = torch.zeros(len_sample, N_devices*num_frame, dtype = torch.complex128, device =0)
    combined_channel = torch.zeros(size = (len_sample, N_devices, num_frame),dtype = torch.complex128, device =0)
    for jjj in range(N_devices):
        combined_channel[:,jjj,:] = Channel_sequence[-1][:,jjj,:]@phase_shifts
    for frame in range(num_frame):
        Received_pilots[:,frame*N_devices:(frame+1)*N_devices] = combined_channel[:,:,frame].squeeze(-1)@pilots[frame*N_devices:(frame+1)*N_devices,:] + torch.tensor(np.sqrt(2)/2)*torch.normal(mean = 0, std = np.sqrt(dbm2pw(noise))*torch.ones(len_sample, N_devices)).cuda() + 1j*torch.tensor(np.sqrt(2)/2)*torch.normal(mean = 0, std = np.sqrt(dbm2pw(noise))*torch.ones(len_sample, N_devices)).cuda()
    return combined_channel, phase_shifts,Received_pilots

def Data_generation(N_devices, N_IRS, RIS_loc, FC_loc, FC_RIS_dis, L_0, alpha,len_sample,pilots_length,K = 8, noise = 110, transmit_p = 1):
    Channel_sequence = Channel_generation(N_devices, N_IRS, RIS_loc, FC_loc, FC_RIS_dis, L_0, alpha,len_sample,K = K)
    pilots = Generate_pilots(N_devices, pilots_length, len_sample,transmit_p = transmit_p)
    phase_shifts = Phaseshift_Generation(N_devices,N_IRS,pilots_length)
    combined_channel, phase_shifts,Received_pilots = Received_pilots_generation(phase_shifts,pilots,len_sample,N_devices,N_IRS,pilots_length,Channel_sequence, noise = noise)
    return phase_shifts,Received_pilots, Channel_sequence, pilots

def plot(Results, x_ticks, dir):
    #plot
    x_axis = np.asarray(x_ticks)
    plt.semilogy(x_axis, Results['mse_our'], marker='v', label='DL power control')
    plt.xlim(x_ticks[0], x_ticks[-1])
    plt.grid(True)
    plt.legend()
    plt.savefig(dir + '/test.png')
    print('end')
