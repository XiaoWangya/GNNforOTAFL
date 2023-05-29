clc, clear;
% if isempty(gcp('nocreate'))
%     parpool('local', 30);
% end
%% IRS-assisted Over-the-air Computation FL
%                                               %
%   Author: Zixin Wang                         %
%   Email: wangzx2 @ shanghaitech.edu.cn        %
%                                               %
%% ========================================
addpath('./func_fed');
addpath('./plot_data/different_N_devices/');
%%
param.num_dev = 10;
%% import E2E data
file1 = append('E2E_fq_',num2str(param.num_dev,'%d'),'.mat');
file2 = append('E2E_p_',num2str(param.num_dev,'%d'),'.mat');
file3 = append('E2E_eta_',num2str(param.num_dev,'%d'),'.mat');
E2E_fq = load(file1).E2E_fq;
E2E_p = load(file2).E2E_p;
E2E_eta = load(file3).E2E_eta;
data_type = 'mnist';
dist_type = 'noniid';
[param] = fed_initial(data_type, dist_type, param);
param.agg_type = 'grad'; % 'model', 'diff', or 'grad'
param.variable = 'basic';
param.comm = 40;
param.verb = 1; % print testing info
param.rep = 5;
max_iter = 10;
%% fig settings
line_width = 2;
marker_indices = 1:5:(param.comm + 1);
%% AirComP
noise_in_dbm = -70;%dbm
Ac.eta_l = 1e-15;
Ac.frame = 4;
Ac.sigma = db2pow(noise_in_dbm)*1e-3;
Ac.p = 10;% transmit power
Ac.mode = 'no_cm_error';% [DL, LMMSE, no_cm_error]
Acc_proposed = zeros(1, param.comm+1);
loss_proposed = zeros(1, param.comm+1);
Acc_proposed_lmmse = zeros(1, param.comm+1);
loss_proposed_lmmse = zeros(1, param.comm+1);
Acc_noirs = zeros(1, param.comm+1);
loss_noirs = zeros(1, param.comm+1);
Acc_noirs_lmmse = zeros(1, param.comm+1);
loss_noirs_lmmse = zeros(1, param.comm+1);
Acc_perfect = zeros(1, param.comm+1);
loss_perfect = zeros(1, param.comm+1);
Acc_EE = zeros(1, param.comm+1);
loss_E2E = zeros(1, param.comm+1);
%% No IRS with perfect CSI
tic;
Ac.N = 0;
parfor ii = 1:max_iter
    [fq, p, eta] = AirComp(param, Ac, ii);
    [acc, loss] = train_logr_aircomp(param, fq, p, eta, Ac.sigma);
    loss_noirs = loss_noirs + loss;
    Acc_noirs = Acc_noirs + acc;
end
toc;
%% Ours
tic;
parfor ii = 1:max_iter
    [acc_EE, loss_EE]= train_logr_aircomp(param, squeeze(E2E_fq(ii,:,:)), squeeze(E2E_p(ii,:,:)), E2E_eta(ii,:), Ac.sigma);
    loss_E2E = loss_E2E + loss_EE;
    Acc_EE = Acc_EE + acc_EE;
end
toc;
%% DC with perfect CSI
Ac.N = 100;
tic;
parfor ii = 1:max_iter
    [fq_irs, p_irs, eta_irs] = AirComp(param, Ac, ii);
    [acc_irs, loss_irs]= train_logr_aircomp(param, fq_irs, p_irs, eta_irs, Ac.sigma);
    loss_proposed = loss_proposed + loss_irs;
    Acc_proposed = Acc_proposed + acc_irs;
end
toc;
%% Perfect Transmission
parfor ii = 1:max_iter
    [acc_n, loss_n] = train_logr(param);
    loss_perfect = loss_perfect + loss_n;
    Acc_perfect = Acc_perfect + acc_n;
end
%% Data averaging
Acc_proposed = Acc_proposed/max_iter;
Acc_proposed_lmmse = Acc_proposed_lmmse/max_iter;
Acc_noirs_lmmse = Acc_noirs_lmmse/max_iter;
Acc_noirs = Acc_noirs/max_iter;
Acc_EE = Acc_EE/max_iter;
loss_proposed = loss_proposed/max_iter;
loss_proposed_lmmse = loss_proposed_lmmse/max_iter;
loss_noirs_lmmse = loss_noirs_lmmse/max_iter;
loss_noirs = loss_noirs/max_iter;
loss_E2E = loss_E2E/max_iter;
Acc_perfect = Acc_perfect/max_iter;
loss_perfect = loss_perfect/max_iter;
%% save data
save Feel_noniid.mat
%% plot convergence
figure;
hold on;
box on;
plot(0:param.comm, loss_noirs, '->', 'Color',[0.4,0.2,0.3],'LineWidth', line_width, 'DisplayName','No RIS', 'MarkerIndices', marker_indices);
hold on
plot(0:param.comm, loss_proposed, 'g-o', 'LineWidth', line_width,'DisplayName', 'Optimization-based', 'MarkerIndices', marker_indices);
hold on
plot(0:param.comm, loss_E2E, 'b-s', 'LineWidth', line_width, 'DisplayName','GNN-based Learning', 'MarkerIndices', marker_indices);
hold on
plot(0:param.comm, loss_perfect, 'r-^', 'LineWidth', line_width, 'DisplayName','Error Free', 'MarkerIndices', marker_indices);
xlabel('Communication round', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Training loss', 'Interpreter', 'latex', 'FontSize', 14);
grid on
legend('location','southwest')

%% plot test accuracy
figure;
hold on;
plot(0:param.comm, Acc_perfect, 'r-^', 'LineWidth', line_width, 'DisplayName','Error Free', 'MarkerIndices', marker_indices);
hold on
plot(0:param.comm, Acc_EE, 'b-s', 'LineWidth', line_width, 'DisplayName','GNN-based learning', 'MarkerIndices', marker_indices);
hold on
plot(0:param.comm, Acc_proposed, 'g-o', 'LineWidth', line_width,'DisplayName','Optimization-based', 'MarkerIndices', marker_indices);
hold on
plot(0:param.comm, Acc_noirs, '->', 'Color',[0.4,0.2,0.3],'LineWidth', line_width, 'DisplayName','No RIS', 'MarkerIndices', marker_indices);
hold on
xlabel('Communication round', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Test accuracy', 'Interpreter', 'latex', 'FontSize', 14);
ylim([0.65,0.85])
yticks([0.65:0.05:0.85])
grid on
box on;
legend('location','southeast')