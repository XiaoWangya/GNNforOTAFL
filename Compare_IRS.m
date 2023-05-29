% clc, clear;
% if isempty(gcp('nocreate'))
%     parpool('local', 20);
% end
tic;
%% IRS-assisted Over-the-air Computation FL
%                                               %
%   Author: Zixin Wang                         %
%   Email: wangzx2 @ shanghaitech.edu.cn        %
%                                               %
%% ========================================
addpath('./func_fed');
% addpath('./test');
addpath('./plot_data/different_N_IRS/');
%%
param.num_dev = 10;
param.variable = 'IRS';
data_type = 'mnist';
dist_type = 'iid';
[param] = fed_initial(data_type, dist_type, param);
param.agg_type = 'grad'; % 'model', 'diff', or 'grad'
param.comm = 40;
param.verb = 1; % print testing info
param.rep = 5;
max_iter = 10;
%% fig settings
line_width = 2;
%% AirComP
N_irs = [5, 10];
noise_in_dbm = -70;%dbm
Ac.eta_l = 1e-15;
Ac.frame = 4;
Ac.sigma = db2pow(noise_in_dbm)*1e-3;
Ac.p = 10;% transmit power
Ac.mode = 'no_cm_error';% [DL, LMMSE, no_cm_error]
% Acc_perfect = zeros(1, param.comm+1);
% loss_perfect = zeros(1, param.comm+1);
% Acc_noirs = zeros(1, param.comm+1);
% loss_noirs = zeros(1, param.comm+1);
% Acc_noirs_lmmse = zeros(1, param.comm+1);
% loss_noirs_lmmse = zeros(1, param.comm+1);
% Acc_proposed = zeros(length(N_irs), param.comm+1);
% loss_proposed = zeros(length(N_irs), param.comm+1);
% Acc_proposed_lmmse = zeros(length(N_irs), param.comm+1);
% loss_proposed_lmmse = zeros(length(N_irs), param.comm+1);
% Acc_EE = zeros(length(N_irs), param.comm+1);
% loss_E2E = zeros(length(N_irs), param.comm+1);
for jj = 1:length(N_irs)
    %% refresh number of frame
    Ac.N = N_irs(jj);
    %% import E2E data
    file1 = append('E2E_fq_',num2str(Ac.N,'%d'),'_irs.mat');
    file2 = append('E2E_p_',num2str(Ac.N,'%d'),'_irs.mat');
    file3 = append('E2E_eta_',num2str(Ac.N,'%d'),'_irs.mat');
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
    Acc_EE(jj,:) = Acc_EE(jj,:) + temp2;
    toc;
    %% DC with perfect CSI
%     Ac.mode = 'no_cm_error';
%     tic;
%     temp1 = 0;
%     temp2 = 0;
%     parfor ii = 1:max_iter
%         [fq, p, eta] = AirComp(param, Ac, ii);
%         [acc, loss] = train_logr_aircomp(param, fq, p, eta, Ac.sigma);
%         temp1= temp1 + loss;
%         temp2 = temp2 + acc;
%     end
%     loss_proposed(jj,:) = loss_proposed(jj,:) + temp1;
%     Acc_proposed(jj,:) = Acc_proposed(jj,:) + temp2;
%     toc;
end
%% Perfect Transmission
% parfor ii = 1:max_iter
%     [acc_n, loss_n] = train_logr(param);
%     loss_perfect = loss_perfect + loss_n;
%     Acc_perfect = Acc_perfect + acc_n;
% end
%% Data averaging
Acc_proposed = Acc_proposed/max_iter;
Acc_proposed_lmmse = Acc_proposed_lmmse/max_iter;
Acc_noirs_lmmse = Acc_noirs_lmmse/max_iter;
Acc_noirs = Acc_noirs/max_iter;
Acc_EE = Acc_EE/max_iter;
Acc_perfect = Acc_perfect/max_iter;
loss_perfect = loss_perfect/max_iter;
loss_proposed = loss_proposed/max_iter;
loss_proposed_lmmse = loss_proposed_lmmse/max_iter;
loss_noirs_lmmse = loss_noirs_lmmse/max_iter;
loss_noirs = loss_noirs/max_iter;
loss_E2E = loss_E2E/max_iter;
%% save
save irs.mat
%% plot convergence
xx = N_irs;
figure;
hold on;
box on;
plot(xx, loss_proposed(:,end), 'g-o', 'LineWidth', line_width,'DisplayName','Optimization Based');
hold on
plot(xx, loss_E2E(:,end), 'b-s', 'LineWidth', line_width, 'DisplayName','GNN Based');
hold on
plot(xx, loss_perfect(end)*ones(length(N_irs),1), 'r-^', 'LineWidth', line_width, 'DisplayName','Error Free');
xlabel('Number of RIS elements', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Training loss', 'Interpreter', 'latex', 'FontSize', 14);
grid on
xticks(xx)
legend('location','northeast')
%% plot test accuracy
figure;
hold on;
box on;
plot(xx, Acc_perfect(end)*ones(length(N_irs),1), 'r-^', 'LineWidth', line_width, 'DisplayName','Error Free');
hold on
plot(xx, Acc_EE(:,end), 'b-s', 'LineWidth', line_width, 'DisplayName','GNN Based');
hold on
plot(xx, Acc_proposed(:,end), 'g-o', 'LineWidth', line_width,'DisplayName','Optimization Based');
xlabel('Number of RIS elements', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Test accuracy', 'Interpreter', 'latex', 'FontSize', 14);
xticks(xx)
grid on
legend('location','southeast')
tic;