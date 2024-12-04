function [z, history] = Gene_BSR(A, b, lambda, p, tau)
%Generalized Block Sparse Representation
%based on ADMM by S.Boyd
%% Decription of inputs and output
%A: mxn; m--number of measurements; n--number of variables;
%b: mx1
%lambda: positive real number; 
%p:n_Gx1; n_G--number of groups;n_G(i)--size of the i-th group
%tau:n_Gx1; tau(i)--weights of the i-th group

t_start = tic;
%% Global constants and defaults
QUIET    = 1;
MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
rho=1;
alpha=1;
%% Data preprocessing

[m, n] = size(A);

% save a matrix-vector multiply
Atb = A'*b;
% check that sum(p) = total number of elements in x
if (sum(p) ~= n)
    error('invalid partition');
end

% cumulative partition
cum_part = cumsum(p);

%% ADMM solver
x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);

% pre-factor
[L,U] = factor(A, rho);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER

    % x-update
    q = Atb + rho*(z - u);    % temporary value
    if( m >= n )    % if skinny
       x = U \ (L \ q);
    else            % if fat
       x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
    end

    % z-update
    zold = z;
    start_ind = 1;
    x_hat = alpha*x + (1-alpha)*zold;
    for i = 1:length(p),
        sel = start_ind:cum_part(i);
        z(sel) = shrinkage(x_hat(sel) + u(sel), tau(i)*lambda/rho);
        start_ind = cum_part(i) + 1;
    end
    u = u + (x_hat - z);
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, lambda, cum_part, x, z,tau);
    
    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));
    
    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end

end

if ~QUIET
    toc(t_start);
end

end

function p = objective(A, b, lambda, cum_part, x, z,tau)
    obj = 0;
    start_ind = 1;
    for i = 1:length(cum_part),
        sel = start_ind:cum_part(i);
        obj = obj + tau(i)*norm(z(sel));
        start_ind = cum_part(i) + 1;
    end
    p = ( 1/2*sum((A*x - b).^2) + lambda*obj );
end

function z = shrinkage(x, kappa)
    z = pos(1 - kappa/norm(x))*x;
end

function x=pos(y)
x=max(y,0);
end

function [L,U] = factor(A, rho)
    [m, n] = size(A);
    if ( m >= n )    % if skinny
       L = chol( A'*A + rho*speye(n), 'lower' );
    else            % if fat
       L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
    end
    
    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end