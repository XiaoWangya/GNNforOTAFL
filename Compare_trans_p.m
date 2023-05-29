clc, clear;
if isempty(gcp('nocreate'))
    parpool('local', 20);
end
%% IRS-assisted Over-the-air Computation FL
%                                               %
%   Author: Zixin Wang                         %
%   Email: wangzx2 @ shanghaitech.edu.cn        %
%                                               %
%% ========================================
addpath('./func_fed');
% addpath('./test');
addpath(genpath('./plot_data/different_trans_power/'));
%%
param.num_dev = 10;
param.variable = 'power';
data_type = 'mnist';
dist_type = 'iid';
[param] = fed_initial(data_type, dist_type, param);
param.agg_type = 'grad'; % 'model', 'diff', or 'grad'
param.rep = 5;
param.comm = 40;
param.verb = 1; % print testing info
max_iter = 10;
%% fig settings、
%% AirComP
noise_in_dbm = -75;%dbm
Ac.eta_l = 1e-15;
Ac.frame = 4;
Ac.sigma = db2pow(noise_in_dbm)*1e-3;
Ac.p = 10;% transmit power
Ac.mode = 'no_cm_error';% [DL, LMMSE, no_cm_error]
Acc_perfect = zeros(1, param.comm+1);
loss_perfect = zeros(1, param.comm+1);
Acc_noirs = zeros(1, param.comm+1);
loss_noirs = zeros(1, param.comm+1);
Acc_noirs_lmmse = zeros(5, param.comm+1);
loss_noirs_lmmse = zeros(5, param.comm+1);
Acc_proposed = zeros(1, param.comm+1);
loss_proposed = zeros(1, param.comm+1);
Acc_proposed_lmmse = zeros(5, param.comm+1);
loss_proposed_lmmse = zeros(5, param.comm+1);
Acc_EE = zeros(5, param.comm+1);
loss_E2E = zeros(5, param.comm+1);
Pilot_power = [10, 12, 15, 17, 20];
%% No IRS with perfect CSI
fprintf('no_cm_error')
Ac.mode = 'no_cm_error';
tic;
Ac.N = 0;
temp1 = 0;
temp2 = 0;
parfor ii = 1:max_iter
    [fq, p, eta] = AirComp(param, Ac, ii);
    [acc, loss] = train_logr_aircomp(param, fq, p, eta, Ac.sigma);
    temp1= temp1 + loss;
    temp2 = temp2 + acc;
end
loss_noirs = loss_noirs + temp1;
Acc_noirs  = Acc_noirs  + temp2;
toc;
%% DC with perfect CSI
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
loss_proposed = loss_proposed + temp1;
Acc_proposed = Acc_proposed + temp2;
toc;
for jj = 1:5
    %% refresh number of frame
    Ac.mode = 'LMMSE';
    fprintf('LMMSE')
    Ac.p = Pilot_power(jj);
    %% import E2E data
    file1 = append('E2E_fq_',num2str(Pilot_power(jj),'%d'),'_transp.mat');
    file2 = append('E2E_p_',num2str(Pilot_power(jj),'%d'),'_transp.mat');
    file3 = append('E2E_eta_',num2str(Pilot_power(jj),'%d'),'_transp.mat');
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
    %% No IRS with LMMSE
    tic;
    Ac.N = 0;
    temp1 = 0;
    temp2 = 0;
    parfor ii = 1:max_iter
        [fq, p, eta] = AirComp(param, Ac, ii);
        [acc, loss] = train_logr_aircomp(param, fq, p, eta, Ac.sigma);
        temp1= temp1 + loss;
        temp2 = temp2 + acc;
    end
    loss_noirs_lmmse(jj,:) = loss_noirs_lmmse(jj,:) + temp1;
    Acc_noirs_lmmse(jj,:)  = Acc_noirs_lmmse(jj,:)  + temp2;
    toc;
    %% DC with estimation error
    tic;
    Ac.N = 100;
    temp1 = 0;
    temp2 = 0;
    parfor ii = 1:max_iter
        [fq_irs, p_irs, eta_irs] = AirComp(param, Ac, ii);
        [acc_irs, loss_irs]= train_logr_aircomp(param, fq_irs, p_irs, eta_irs, Ac.sigma);
        temp1 = temp1 + loss_irs;
        temp2 = temp2 + acc_irs;
    end
    loss_proposed_lmmse(jj,:) = loss_proposed_lmmse(jj,:) + temp1;
    Acc_proposed_lmmse(jj,:) = Acc_proposed_lmmse(jj,:) + temp2;
    toc;
end

%% Perfect Transmission
parfor ii = 1:max_iter
    [acc_n, loss_n] = train_logr(param);
    loss_perfect = loss_perfect + loss_n;
    Acc_perfect = Acc_perfect + acc_n;
end

Acc_proposed = Acc_proposed/max_iter;
Acc_proposed_lmmse = Acc_proposed_lmmse/max_iter;
Acc_noirs = Acc_noirs/max_iter;
Acc_noirs_lmmse = Acc_noirs_lmmse/max_iter;
Acc_EE = Acc_EE/max_iter;
Acc_perfect = Acc_perfect/max_iter;
loss_perfect = loss_perfect/max_iter;
loss_proposed = loss_proposed/max_iter;
loss_proposed_lmmse = loss_proposed_lmmse/max_iter;
loss_noirs = loss_noirs/max_iter;
loss_noirs_lmmse = loss_noirs_lmmse/max_iter;
loss_E2E = loss_E2E/max_iter;
%% Save
save trans_p.mat
%% plot resutls
figure;
hold on;
box on;
xx = Pilot_power(1:5);
plot(xx, loss_noirs_lmmse(:,end), '-->', 'Color',[0.2,0.5,0.3],'LineWidth', line_width, 'DisplayName','No IRS (with LMMSE channel estimation)');
hold on
plot(xx, loss_noirs(end)*ones(5,1), '->', 'Color',[0.4,0.2,0.3],'LineWidth', line_width, 'DisplayName','No IRS');
hold on
plot(xx, loss_proposed_lmmse(:,end), '-p','Color',[0.2,0.7,0.3], 'LineWidth', line_width, 'DisplayName','LMMSE channel estimation + DC');
hold on
plot(xx, loss_proposed(end)*ones(5,1), 'g-o', 'LineWidth', line_width,'DisplayName','Perfect CSI + DC');
hold on
plot(xx, loss_E2E(:,end), 'b-s', 'LineWidth', line_width, 'DisplayName','RISFEEL');
hold on
plot(xx, loss_perfect(end)*ones(5,1), 'r-^', 'LineWidth', line_width, 'DisplayName','Error free');
xlabel('Pilot transmit power', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Training loss', 'Interpreter', 'latex', 'FontSize', 14);
grid on
xticks(xx)
legend('location','best')

figure;
hold on;
box on;
plot(xx, Acc_perfect(end)*ones(5,1), 'r-^', 'LineWidth', line_width, 'DisplayName','Error free');
hold on
plot(xx, Acc_EE(:,end), 'b-s', 'LineWidth', line_width, 'DisplayName','RISFEEL');
hold on
plot(xx, Acc_proposed(end)*ones(5,1), 'g-o', 'LineWidth', line_width,'DisplayName','Perfect CSI + DC');
hold on
plot(xx, Acc_proposed_lmmse(:,end), '-p','Color',[0.2,0.7,0.3], 'LineWidth', line_width, 'DisplayName','LMMSE channel estimation + DC');
hold on
plot(xx, Acc_noirs(end)*ones(5,1), '->', 'Color',[0.4,0.2,0.3],'LineWidth', line_width, 'DisplayName','No IRS');
hold on
plot(xx, Acc_noirs_lmmse(:,end), '-->', 'Color',[0.2,0.5,0.3],'LineWidth', line_width, 'DisplayName','No IRS (with LMMSE channel estimation)');
xlabel('Pilot transmit power', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Test accuracy', 'Interpreter', 'latex', 'FontSize', 14);
xticks(xx)
ylim([0.77,0.85])
yticks([0.77:0.02:0.85])
grid on
title('Test accuracy vesus transmit power of pilots')
legend('location','best')