function [fq_real,p, eta, result] = AirComp(param, Ac, ii)
%% system setting
% rng('default')
% rng(856)
%% Wireless setting
if  strcmp(param.variable, 'IRS') && Ac.N
        N=Ac.N;
else
    N = 100;
end
M=1;
K = param.num_dev;
N_ite = 4;
Ite_DC=100;
eta_h = 10^5;
eta_l = Ac.eta_l;
num_frame = Ac.frame;
pilot_power = Ac.p;%dbm
sigma = Ac.sigma;
mode = Ac.mode;
%% solve
file2 = append('Channel_real_LMMSE_data_',num2str(K,'%d'),'_',num2str(K*num_frame,'%d'),'_',num2str(pilot_power,'%d'),'_',num2str(N,'%d'),'_equalSNR_0.mat');
channel_real = load(file2).Channel_real_data_LMMSE;
channel_est = load(file2).Channel_real_data_LMMSE;
if strcmp(mode, 'DL')
    file1 = append('Channel_estimated_DL_data_',num2str(K,'%d'),'_',num2str(K*num_frame,'%d'),'_',num2str(pilot_power,'%d'),'_',num2str(N,'%d'),'.mat');
    channel_est = load(file1).Channel_estimated_data_DL;
    file2 = append('Channel_real_DL_data_',num2str(K,'%d'),'_',num2str(K*num_frame,'%d'),'_',num2str(pilot_power,'%d'),'_',num2str(N,'%d'),'.mat');
    channel_real = load(file2).Channel_real_data_DL;
elseif strcmp(mode, 'LMMSE')
    file1 = append('Channel_estimated_LMMSE_data_',num2str(K,'%d'),'_',num2str(K*num_frame,'%d'),'_',num2str(pilot_power,'%d'),'_',num2str(N,'%d'),'_equalSNR_0.mat');
    channel_est = load(file1).Channel_estimated_data_LMMSE;
end
N = Ac.N;
T= param.comm;
tic;
result = 0;
P_bar=ones(1,K)*10^(27/10 -3);
eta=ones(T,1)*1e-11;
v=zeros(N,T);
p=ones(T,K).*P_bar;
hd_est = reshape(channel_est(ii, 1: T, :, 1), [1,K,T]);
alpha_est = reshape(channel_est(ii,1: T,:,2:N+1),[K,N,T]);
hd_real = reshape(channel_real(ii, 1: T, :, 1), [1,K,T]);
alpha_real = reshape(channel_real(ii,1: T,:,2:N+1),[K,N,T]);
MSE=zeros(N_ite,1);
R=zeros(N+1,N+1,K);
fq_est=zeros(T,K);
fq_real=zeros(T,K);
feasible = ones(T,1);
if N == 0
    N_ite = 1;
end
for ite1=1:N_ite
    %% refreash channel
    for t=1:T
        for k=1:K
            fq_est(t,k)=[hd_est(1,k,t),alpha_est(k,:,t)]*[1;v(:,t)];
        end
    end
    %% solve p & eta
    [eta,p] = cal_pow_and_eta(fq_est,eta_l,P_bar,sigma,eta_h,eta_l,p);
    p_mean=mean(p,1);
    %% solve v
    if N
        for t = 1:T
            [v(:,t), feasible(t)] = DC_V(alpha_est(:,:,t),hd_est(:,:,t),Ite_DC,eta(t),p(t,:));
        end
    end
    %% Check MSE
    for t=1:T
        for k=1:K
            fq_real(t,k)=[hd_real(1,k,t),alpha_real(k,:,t)]*[1;v(:,t)];
        end
    end
    [Mse, signal_misalignment_error,noise_induced_error] = mse(fq_real,eta,p,K,T,sigma);
    Mse = Mse/K^2;
    MSE(ite1)=Mse;
    if sum(feasible)<T
        break
    end
    if ite1>=3
        if abs(MSE(ite1)-MSE(ite1-1))/MSE(ite1-1)<5e-3
            break
        end
    end
end
result = min(MSE(MSE>0))
toc;
end


