function[MSE,signal_misalignment_error,noise_induced_error]=mse(channel,eta,p,K,T,sigma)
Mse=zeros(T,1);
signal_misalignment_error=zeros(T,1);
noise_induced_error=zeros(T,1);
% [~,n_devices,~] = size(hd);
for t=1:T
    for k=1:K
        signal_misalignment_error(t)=signal_misalignment_error(t)+((sqrt(p(t,k))*abs(channel(t,k)))/sqrt(eta(t))-1)^2;
    end
    noise_induced_error(t) = sigma/eta(t);
    Mse(t)=signal_misalignment_error(t)+noise_induced_error(t);
end
MSE=mean(Mse);
signal_misalignment_error = mean(signal_misalignment_error);
noise_induced_error = mean(noise_induced_error(t));
end