function[v, status] = DC_V(alpha,hd,Ite_DC,eta,p)
rng('default')
rng(33)
[K,N] = size(alpha);
v=rand(N,1);
cvx_clear;
R = zeros(N+1,N+1,K);
for k=1:K
    R(:,:,k) = [alpha(k,:)'*alpha(k,:), alpha(k,:)'*hd(1,k);
        hd(1,k)'*alpha(k,:),hd(1,k)'*hd(1,k)];
end
%% SDR initialization
% cvx_begin quiet
% variable V(N+1,N+1) complex semidefinite
% temp = 0;
% for k = 1:K
%     temp = temp + (real(trace(R(:,:,k)*V))*p(k)/eta-1)^2;
% end
% minimize temp
% subject to
% diag(V)==1;
% cvx_end
% cvx_clear
% [u,~] = eigs(V);
%% random initialization
tmp = randn(N+1,N+1)+1i*randn(N+1,N+1);
V_rand = tmp*tmp';
[u,~] = eigs(V_rand);
%%
V_partial = u(:,1)*u(:,1)';
err_l = [];
for ite2= 1:Ite_DC
    cvx_begin quiet
    cvx_solver sdpt3
    variable V(N+1,N+1) complex semidefinite
    temp = 0;
    for k = 1:K
        temp = temp + (real(trace(R(:,:,k)*V))-sqrt(eta/p(k)))^2;
    end
    minimize 1e1*real(trace(V'*(eye(N+1)-V_partial))) + temp
    subject to
    diag(V)==1;
    cvx_end
    cvx_clear
    if strcmp(cvx_status,'Infeasible')
        vt=zeros(N,1);
        status = 0;
        return
    else
        [U,S,~]=svd(V);
        V_partial=U(:,1)*U(:,1)';
        err=trace(S)-S(1,1);
        err_l = [err_l, err];
        if err<=1e-7
            break
        end
    end
end
v_hat=U(:,1)*sqrt(S(1,1));
vt=v_hat./v_hat(end);
vt=vt(1:N);
v=vt;
status = 1;
end