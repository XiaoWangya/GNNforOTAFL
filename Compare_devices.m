clc, clear;
% if isempty(gcp('nocreate'))
%     parpool('local', 20);
% end
%% IRS-assisted Over-the-air Computation FL
%                                               %
%   Author: Zixin Wang                         %
%   Email: wangzx2 @ shanghaitech.edu.cn        %
%                                               %
%% ========================================
addpath('./func_fed');
% addpath('./test');
addpath(genpath('./plot_data/different_N_devices/'));
%%
param.variable = 'devices';
data_type = 'mnist';
dist_type = 'iid';

param.agg_type = 'grad'; % 'model', 'diff', or 'grad'
param.rep = 5;
param.comm = 40;
param.verb = 1; % print testing info
max_iter = 10;
%% fig settings
line_width = 2;
%% AirComP
noise_in_dbm = -70;%dbm
Ac.eta_l = 1e-15;
Ac.frame = 4;
Ac.sigma = db2pow(noise_in_dbm)*1e-3;
Ac.p = 10;% transmit power
Ac.mode = 'no_cm_error';% [DL, LMMSE, no_cm_error]
Acc_perfect = zeros(5, param.comm+1);
loss_perfect = zeros(5, param.comm+1);
Acc_noirs = zeros(5, param.comm+1);
loss_noirs = zeros(5, param.comm+1);
Acc_noirs_lmmse = zeros(5, param.comm+1);
loss_noirs_lmmse = zeros(5, param.comm+1);
Acc_proposed = zeros(5, param.comm+1);
loss_proposed = zeros(5, param.comm+1);
Acc_proposed_lmmse = zeros(5, param.comm+1);
loss_proposed_lmmse = zeros(5, param.comm+1);
Acc_E2E = zeros(5, param.comm+1);
loss_E2E = zeros(5, param.comm+1);
Number_devices = [5, 10, 15, 20, 25];
for jj = 1:5
    %% refresh number of devices
    param.num_dev = Number_devices(jj);
    [param] = fed_initial(data_type, dist_type, param);
    param.num_dev
    %% import E2E data
    file1 = append('E2E_fq_',num2str(Number_devices(jj),'%d'),'.mat');
    file2 = append('E2E_p_',num2str(Number_devices(jj),'%d'),'.mat');
    file3 = append('E2E_eta_',num2str(Number_devices(jj),'%d'),'.mat');
    E2E_fq = load(file1).E2E_fq;
    E2E_p = load(file2).E2E_p;
    E2E_eta = load(file3).E2E_eta;
    %% Ours
    tic;
    temp1 = 0;
    temp2 = 0;
    parfor ii = 1:max_iter
        [acc_EE, loss_EE]= train_logr_aircomp(param, squeeze(E2E_fq(ii,:,:)), squeeze(E2E_p(ii,:,:)), E2E_eta(ii,:), Ac.sigma);
        temp1 = temp1 + loss_EE;
        temp2 = temp2 + acc_EE;
    end
    loss_E2E(jj,:) = loss_E2E(jj,:) + temp1;
    Acc_E2E(jj,:) = Acc_E2E(jj,:) + temp2;
    toc;
    %% DC with perfect CSI
    fprintf('no_cm_error')
    Ac.mode = 'no_cm_error';
    Ac.N = 100;
    tic;
    temp1 = 0;
    temp2 = 0;
    parfor ii = 1:max_iter
        [fq_irs, p_irs, eta_irs] = AirComp(param, Ac, ii);
        [acc_irs, loss_irs]= train_logr_aircomp(param, fq_irs, p_irs, eta_irs, Ac.sigma);
        temp1 = temp1 + loss_irs;
        temp2 = temp2 + acc_irs;
    end
    loss_proposed(jj,:) = loss_proposed(jj,:) + temp1;
    Acc_proposed(jj,:) = Acc_proposed(jj,:) + temp2;
    toc;
    %% No IRS with perfect CSI
    tic;
    Ac.N = 0;
    temp1 = 0;
    temp2 = 0;
    for ii = 1:max_iter
        [fq, p, eta] = AirComp(param, Ac, ii);
        [acc, loss] = train_logr_aircomp(param, fq, p, eta, Ac.sigma);
        temp1= temp1 + loss;
        temp2 = temp2 + acc;
    end
    loss_noirs(jj,:) = loss_noirs(jj,:) + temp1;
    Acc_noirs(jj,:)  = Acc_noirs(jj,:)  + temp2;
    toc;
    %% Perfect Transmission
    temp1 = 0;
    temp2 = 0;
    parfor ii = 1:max_iter
        [acc_n, loss_n] = train_logr(param);
        temp1	= temp1  + loss_n;
        temp2 = temp2 + acc_n;
    end
    loss_perfect(jj,:) = loss_perfect(jj,:)+ temp1;
    Acc_perfect(jj,:) = Acc_perfect(jj,:) + temp2;
end
Acc_proposed = Acc_proposed/max_iter;
Acc_proposed_lmmse = Acc_proposed_lmmse/max_iter;
Acc_noirs = Acc_noirs/max_iter;
Acc_noirs_lmmse = Acc_noirs_lmmse/max_iter;
Acc_E2E = Acc_E2E/max_iter;
Acc_perfect = Acc_perfect/max_iter;
loss_perfect = loss_perfect/max_iter;
loss_proposed = loss_proposed/max_iter;
loss_proposed_lmmse = loss_proposed_lmmse/max_iter;
loss_noirs = loss_noirs/max_iter;
loss_noirs_lmmse = loss_noirs_lmmse/max_iter;
loss_E2E = loss_E2E/max_iter;
%% Save
save num_dev.mat
%% plot convergence
figure;
hold on;
box on;
xx = Number_devices(1:5);
plot(xx, loss_noirs(:,end), '->', 'Color',[0.4,0.2,0.3],'LineWidth', line_width, 'DisplayName','No RIS');
hold on
plot(xx, loss_proposed(:,end), 'g-o', 'LineWidth', line_width,'DisplayName','DC');
hold on
plot(xx, loss_E2E(:,end), 'b-s', 'LineWidth', line_width, 'DisplayName','RISFEEL');
hold on
plot(xx, loss_perfect(:,end), 'r-^', 'LineWidth', line_width, 'DisplayName','Error free');
xlabel('Number of edge devices', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Training loss', 'Interpreter', 'latex', 'FontSize', 14);
grid on
xticks(xx)
legend('location','best')
%% plot test accuracy
figure;
hold on;
box on;
plot(xx, Acc_perfect(:,end), 'r-^', 'LineWidth', line_width, 'DisplayName','Error Free');
hold on
plot(xx, Acc_E2E(:,end), 'b-s', 'LineWidth', line_width, 'DisplayName','GNN-based learning');
hold on
plot(xx, Acc_proposed(:,end), 'g-o', 'LineWidth', line_width,'DisplayName','Optimization-based');
hold on
plot(xx, Acc_noirs(:,end), '->', 'Color',[0.4,0.2,0.3],'LineWidth', line_width, 'DisplayName','No RIS');
xlabel('Number of edge devices', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Test accuracy', 'Interpreter', 'latex', 'FontSize', 14);
xticks(xx)
grid on
legend('location','southeast')
