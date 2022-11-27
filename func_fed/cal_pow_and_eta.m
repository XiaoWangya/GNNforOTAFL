function[eta,p] = cal_pow_and_eta(channel,bise_acc,P_bar,sigma,eta_h0,eta_l0,p)
[T,K] = size(channel);
count=1;
lambda=ones(1,K);
p_mean=mean(p,1);
subgradient = p_mean-P_bar;
eta=ones(T,1);
log_mse_ls = zeros(1,1e6);
mp = zeros(1e6,K);
while count < 1e4
    %% solve lambda
    p_mean=mean(p,1);
    subgradient = p_mean-P_bar;
    %     subgradient = max(0, p_mean-P_bar);
    %     lambda = max(0, lambda + subgradient/1e2);
    lambda = max(1e-50, lambda + subgradient/count);
    %% solve p
    for t=1:T
        for k=1:K
            p(t,k)=(sqrt(abs(channel(t,k))^2*eta(t))/(abs(channel(t,k))^2+eta(t)*lambda(k)))^2;
        end
    end
    %% solve eta
    for t=1:T
        Delta=zeros(K,1);
        eta_h = eta_h0;
        eta_l = eta_l0;
        acc = eta_h-eta_l;
        while bise_acc<acc
            eta_mid=(eta_h+eta_l)/2;
            temp=0;
            for k=1:K
                Delta(k)=abs(channel(t,k))^2/lambda(k);
                temp=temp+Delta(k)/(Delta(k)/eta_mid+1)^2;
            end
            if temp >= sigma
                eta_h=eta_mid;
            else
                eta_l=eta_mid;
            end
            acc = eta_h-eta_l;
        end
        eta(t)=(eta_h+eta_l)/2;
    end
    log_mse_ls(count) = log10(mse(channel,eta,p,K,T,sigma)/K^2);
    count=count+1;
                mp(count,:) = mean(p)';
    if count>1e2
        if var(log_mse_ls(count-1e2:count-1)) < 1e-7 && max(abs(subgradient.*lambda)) <1e-3
            %         if max(abs(subgradient.*lambda)) <1e-7
            break
        end
    end
end
log_mse_ls = log_mse_ls(log_mse_ls<0);
end