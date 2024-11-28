
% Detection of node and layer core sizes s_node and s_layer by evaluating
% the QUBO objective functions [eq.'s (6) and (7), 1].

% [1] Kai Bergermann and Francesco Tudisco. Core-periphery detection in
% multilayer networks. Preprint.

%% Select numerical experiment
% openalex
year = '2023';
file_str = ['openalex_fully_weighted_year_',year,'_a_10_b_10_p_22_q_22'];

% EUAir
% file_str = 'EUAir_a_10_b_10_p_22_q_22';

% WIOT
% year = '2014';
% file_str = ['WIOT_',year,'_a_10_b_10_p_22_q_22'];

%% load coreness vectors and supra-adjacency matrix
x=load(['results/x_vec_',file_str,'.txt']);
n=size(x,1);
[x_sorted,x_ind] = sort(x,'descend');

c=load(['results/c_vec_',file_str,'.txt']);
L=size(c,1);
[c_sorted,c_ind] = sort(c,'descend');

if strcmp(file_str(1:8),'openalex')
    loadstr = ['data/openalex_fully_weighted_',year,'_supra_adj.mat'];
    load(loadstr)
elseif strcmp(file_str(1:5),'EUAir')
    load 'data/EUAir_supra_adjacency.mat'
elseif strcmp(file_str(1:4),'WIOT')
    loadstr = ['data/WIOT_',year,'_supra_adj.mat'];
    load(loadstr)
else
    error('file_str unknown.')
end

% normalize adjacency matrix
A = A/(max(max(A)));

%% evaluate QUBO objective functions

run_QUBO(A,n,L,x,x_ind,c,c_ind)

function run_QUBO(A,n,L,x,x_ind,c,c_ind)
    % This function implements Algorithm 1 (Efficient QUBO objective
    % function evaluation) from the supplementary information of [1].

    %% s_node, block version

    I_weighted = zeros(nnz(A),1); J_weighted = zeros(nnz(A),1); V_weighted = zeros(nnz(A),1);
    I_D = zeros(nnz(A),1); J_D = zeros(nnz(A),1); V_D = zeros(nnz(A),1); % nnz(A) upper bound on nnz(DD). Cut zeros later
    c_node = 0;
    A_weighted_list_ind = 1; D_list_ind = 1;
    const_ID = 0; const_ONES = 0;
    bar=waitbar(0,'Pre-computations, loop over layers...');
    tic
    for k=1:L
        max_c = max(c(k),c');
        A_rowblock = A((k-1)*n+1:k*n,:);
        D_rowblock1 = sum(A_rowblock,1);
        D_rowblock2 = reshape(A_rowblock*kron(speye(L,L),ones(n,1)),[1,n*L]);
        N1_row_ = full(sum(reshape(D_rowblock1,[n,L]),1));
        N2_row_ = n^2*ones(1,L) - N1_row_;

        % blocks w/o edges don't contribute
        N1_row = N1_row_; N2_row = N2_row_;
        N1_row(N1_row_.*N2_row_==0)=Inf; N2_row(N1_row_.*N2_row_==0)=Inf;

        % diagonal matrix in L x nL format -- to be multiplied (elementwise) by
        % repmat(x_bar',[L,L]) and then summed up
        D_row = sparse((D_rowblock1+D_rowblock2).*(kron(max_c.*(1./N1_row + 1./N2_row),ones(1,n))));
        [~,J,V] = find(D_row);
        I_D(D_list_ind:D_list_ind+length(J)-1) = k*ones(1,length(J));
        J_D(D_list_ind:D_list_ind+length(J)-1) = J;
        V_D(D_list_ind:D_list_ind+length(J)-1) = V;
        D_list_ind = D_list_ind + length(J);

        % sparse AA block matrix -- to be put in an inner product with repmat(c_bar,[n,1])
        [I,J,V] = find(A_rowblock);
        J_weighted(A_weighted_list_ind:A_weighted_list_ind+length(I)-1) = J;
        I_weighted(A_weighted_list_ind:A_weighted_list_ind+length(I)-1) = (k-1)*n*ones(size(J)) + I;
        ind = ceil(J/n);
        V_weighted(A_weighted_list_ind:A_weighted_list_ind+length(I)-1) = ((1./N1_row(ind) + 1./N2_row(ind)).*max_c(ind).*V')';
        A_weighted_list_ind = A_weighted_list_ind + length(I);

        % identity part -- to be multiplied by sum(x)
        const_ID = const_ID + 2*sum((n-1/2)*max_c./N2_row);

        % ones matrix part -- to be multiplied by sum(x)^2
        const_ONES = const_ONES + sum(max_c./N2_row);

        % normalization constant
        c_node = c_node + sum(max_c);

        waitbar(k/L,bar);
    end
    close(bar)

    A_weighted = sparse(I_weighted, J_weighted, V_weighted, n*L, n*L);
    D = sparse(I_D(1:D_list_ind-1),J_D(1:D_list_ind-1),V_D(1:D_list_ind-1),L,n*L);

    x_bar = zeros(n,1);
    QUBO_vals_node_block = zeros(n,1);
    bar=waitbar(0,'QUBO evaluation, loop over nodes...');
    for i=1:n
        x_bar(x_ind(i)) = 1;
        x_bar_block = repmat(x_bar,[L,1]);
        QUBO_vals_node_block(i) = sum(sum(D,1)'.*repmat(x_bar,[L,1])) - const_ID*sum(x_bar) - x_bar_block'*A_weighted*x_bar_block + const_ONES*sum(x_bar)^2;
        QUBO_vals_node_block(i) = QUBO_vals_node_block(i)/c_node;
        waitbar(i/n,bar);
    end
    close(bar)
    [Q_node,s_node] = max(QUBO_vals_node_block)

    %% s_layer, block version
    perm = repmat(n*(0:L-1)',[n,1]) + kron((1:n)',ones(L,1));

    AA = A(perm,perm);

    I_weighted = zeros(nnz(AA),1); J_weighted = zeros(nnz(AA),1); V_weighted = zeros(nnz(AA),1);
    I_DD = zeros(nnz(AA),1); J_DD = zeros(nnz(AA),1); V_DD = zeros(nnz(AA),1); % nnz(AA) upper bound on nnz(DD). Cut zeros later
    c_layer = 0;
    AA_weighted_list_ind = 1; DD_list_ind = 1;
    const_ID = 0; const_ONES = 0;
    bar=waitbar(0,'Pre-computations, loop over nodes...');
    for i=1:n
        max_x = max(x(i),x');
        AA_rowblock = AA((i-1)*L+1:i*L,:);
        DD_rowblock1 = sum(AA_rowblock,1);
        DD_rowblock2 = reshape(AA_rowblock*kron(speye(n,n),ones(L,1)),[1,n*L]);
        N1_row_ = full(reshape(sum(reshape(DD_rowblock1,[L,n]),1),[1,n]));
        N2_row_ = L^2*ones(1,n) - N1_row_;

        % blocks w/o edges don't contribute
        N1_row = N1_row_; N2_row = N2_row_;
        N1_row(N1_row_.*N2_row_==0)=Inf; N2_row(N1_row_.*N2_row_==0)=Inf;

        % diagonal matrix in n x nL format -- to be multiplied (elementwise) by
        % repmat(c_bar',[n,n]) and then summed up. BUT: n x n too large to
        % store, so:
        %%% sum(DD,2).*repmat(c_bar,[n,1])
        DD_row = sparse((DD_rowblock1+DD_rowblock2).*(kron(max_x.*(1./N1_row + 1./N2_row),ones(1,L))));
        [~,J,V] = find(DD_row);
        I_DD(DD_list_ind:DD_list_ind+length(J)-1) = i*ones(1,length(J));
        J_DD(DD_list_ind:DD_list_ind+length(J)-1) = J;
        V_DD(DD_list_ind:DD_list_ind+length(J)-1) = V;
        DD_list_ind = DD_list_ind + length(J);

        % sparse AA block matrix -- to be put in an inner product with repmat(c_bar,[n,1])
        [I,J,V] = find(AA_rowblock);
        J_weighted(AA_weighted_list_ind:AA_weighted_list_ind+length(I)-1) = J;
        I_weighted(AA_weighted_list_ind:AA_weighted_list_ind+length(I)-1) = (i-1)*L*ones(size(J)) + I;
        ind = ceil(J/L);
        V_weighted(AA_weighted_list_ind:AA_weighted_list_ind+length(I)-1) = ((1./N1_row(ind) + 1./N2_row(ind)).*max_x(ind).*V')';
        AA_weighted_list_ind = AA_weighted_list_ind + length(I);

        % identity part -- to be multiplied by sum(c)
        const_ID = const_ID + 2*sum((L-1/2)*max_x./N2_row);

        % ones matrix part -- to be multiplied by sum(c)^2
        const_ONES = const_ONES + sum(max_x./N2_row);

        % normalization constant
        c_layer = c_layer + sum(max_x);

        waitbar(i/n,bar);
    end
    close(bar)

    AA_weighted = sparse(I_weighted, J_weighted, V_weighted, n*L, n*L);
    DD = sparse(I_DD(1:DD_list_ind-1),J_DD(1:DD_list_ind-1),V_DD(1:DD_list_ind-1),n,n*L);

    c_bar = zeros(L,1);
    QUBO_vals_layer_block = zeros(L,1);
    for k=1:L
        c_bar(c_ind(k)) = 1;
        c_bar_block = repmat(c_bar,[n,1]);
        QUBO_vals_layer_block(k) = sum(sum(DD,1)'.*repmat(c_bar,[n,1])) - const_ID*sum(c_bar) - c_bar_block'*AA_weighted*c_bar_block + const_ONES*sum(c_bar)^2;
        QUBO_vals_layer_block(k) = QUBO_vals_layer_block(k)/c_layer;
    end
    
    [Q_layer,s_layer] = max(QUBO_vals_layer_block)

end
