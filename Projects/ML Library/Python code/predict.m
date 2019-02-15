function [Y_hat,Z_hat] = predict(Train_Data_X_Test,B_zero,theta_zero,eta,yy,q)
    [n,~,~] = size(Train_Data_X_Test);
    ita_X = [];
    for k = 1:n
        ita_X_k = repmat({Train_Data_X_Test(k,:)}, 1, q); % inital beta (p*1) estimation required
        ita_X_k = blkdiag(ita_X_k{:});
        ita_X = vertcat(ita_X,ita_X_k);
    end
    B_zero_stack = repmat(B_zero,n,1);
    theta_zero_stack = repmat(theta_zero',n,1);
    Y_hat = reshape((ita_X.*B_zero_stack)*eta,[q,n])';
    Z_hat = (Y_hat.*theta_zero_stack)*yy;

