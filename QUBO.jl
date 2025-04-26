
# Detection of node and layer core sizes s_node and s_layer by evaluating
# the QUBO objective functions [eq.'s (6) and (7), 1].

# [1] Kai Bergermann and Francesco Tudisco. Core-periphery detection in
# multilayer networks. Preprint.

using LinearAlgebra, SparseArrays, MAT

## Select numerical experiment
# openalex
#year = "2023"
#file_str = string("openalex_fully_weighted_year_",year,"_a_10_b_10_p_22_q_22")

# EUAir
file_str = "EUAir_a_10_b_10_p_22_q_22";

# WIOT
#year = "2000"
#file_str = string("WIOT_",year,"_a_10_b_10_p_22_q_22")

## load coreness vectors and supra-adjacency matrix
x = parse.(Float64, readlines(string("results/x_vec_",file_str,".txt")))
n = size(x,1)
x_sorted = sort(x, rev=true)
x_ind = sortperm(x, rev=true)

c = parse.(Float64, readlines(string("results/c_vec_",file_str,".txt")))
L = size(c,1)
c_sorted = sort(c, rev=true)
c_ind = sortperm(c, rev=true)

if first(file_str,8)=="openalex"
	vars = matread(string("data/openalex_fully_weighted_",year,"_supra_adj.mat"))
elseif first(file_str,5)=="EUAir"
	vars = matread("data/EUAir_supra_adjacency.mat")
elseif first(file_str,4)=="WIOT"
	vars = matread(string("data/WIOT_",year,"_supra_adj.mat"))
else
	println("Error: file_str unknown.")
end
A = vars["A"]
nL = size(A,1)

# normalize adjacency matrix
A = A/maximum(A)

# QUBO evaluation function
function run_QUBO(A,n,L,x,x_ind,c,c_ind)
	# This function implements Algorithm 1 (Efficient QUBO objective
	# function evaluation) from the supplementary information of [1].
	
	## s_node, block version
	I_weighted = zeros(nnz(A),1)
	J_weighted = zeros(nnz(A),1)
	V_weighted = zeros(nnz(A),1)
	# nnz(A) upper bound on nnz(D). Cut zeros later
	I_D = zeros(nnz(A),1)
	J_D = zeros(nnz(A),1)
	V_D = zeros(nnz(A),1)
	c_node = 0
	A_weighted_list_ind = 1
	D_list_ind = 1
	const_ID = 0
	const_ONES = 0
	
	for k=1:L
		max_c = max.(c[k],c')
		A_rowblock = A[(k-1)*n+1:k*n,:]
		D_rowblock1 = sum(A_rowblock, dims=1)
		D_rowblock2 = reshape(A_rowblock*kron(sparse(I,L,L),ones(n,1)), (1,n*L))
		N1_row = sum(reshape(D_rowblock1,(n,L)), dims=1)
		N2_row = n^2*ones(1,L) - N1_row
		
		# blocks w/o edges don't contribute
		zero_inds = (N1_row.*N2_row).==0
		N1_row[zero_inds] .= Inf
		N2_row[zero_inds] .= Inf
		
		# diagonal matrix in L x nL format -- to be multiplied (elementwise) by
		# repmat(x_bar',[L,L]) and then summed up
		D_row = sparse((D_rowblock1+D_rowblock2).*(kron(max_c.*(1 ./ N1_row + 1 ./ N2_row),ones(1,n))))
		D_row_nz = findnz(D_row)
		J = Int64.(D_row_nz[2])
		V = D_row_nz[3]
		I_D[D_list_ind:D_list_ind+length(J)-1] = k*ones(1,length(J))
		J_D[D_list_ind:D_list_ind+length(J)-1] = J
		V_D[D_list_ind:D_list_ind+length(J)-1] = V
		D_list_ind = D_list_ind + length(J)
		
		# sparse AA block matrix -- to be put in an inner product with repmat(c_bar,[n,1])
		A_rowblock_nz = findnz(A_rowblock)
		I_ = Int64.(A_rowblock_nz[1])
		J = Int64.(A_rowblock_nz[2])
		V = A_rowblock_nz[3]
		J_weighted[A_weighted_list_ind:A_weighted_list_ind+length(I_)-1] = J
		I_weighted[A_weighted_list_ind:A_weighted_list_ind+length(I_)-1] = (k-1)*n*ones(size(J)) + I_
		ind = Int64.(ceil.(J./n))'
		V_weighted[A_weighted_list_ind:A_weighted_list_ind+length(I_)-1] = ((1 ./N1_row[ind] + 1 ./N2_row[ind]).*max_c[ind].*V' )'
		A_weighted_list_ind = A_weighted_list_ind + length(I_)
		
		# identity part -- to be multiplied by sum(x)
		const_ID = const_ID + 2*sum((n-1/2) .*max_c ./N2_row)
		
		# ones matrix part -- to be multiplied by sum(x)^2
		const_ONES = const_ONES + sum(max_c./N2_row)
		
		# normalization constant
		c_node = c_node + sum(max_c)
	end
	
	A_weighted = sparse(Int64.(I_weighted[:,1]), Int64.(J_weighted[:,1]), V_weighted[:,1], n*L, n*L)
	D = sparse(Int64.(I_D[1:D_list_ind-1]), Int64.(J_D[1:D_list_ind-1]), V_D[1:D_list_ind-1], L, n*L)
	
	x_bar = zeros(n,1)
	QUBO_vals_node_block = zeros(n,1)
	for i=1:n
		x_bar[x_ind[i]] = 1
		x_bar_block = repeat(x_bar, outer=[L,1])
		QUBO_vals_node_block[i] = sum(sum(D, dims=1)'.*repeat(x_bar, outer=[L,1])) - const_ID*sum(x_bar) - only(x_bar_block'*A_weighted*x_bar_block) + const_ONES*sum(x_bar)^2
		QUBO_vals_node_block[i] = QUBO_vals_node_block[i]/c_node
	end
	
	QUBO_node = findmax(QUBO_vals_node_block)
	Q_node = QUBO_node[1]
	s_node = QUBO_node[2]
	println("Q_node = ", Q_node)
	println("s_node = ", s_node[1])
	 
	 
	 
	## s_layer, block version
	perm = repeat(n*(0:L-1), outer=[n,1]) + kron((1:n),ones(L,1))
	perm = Int64.(perm[:,1])
	
	AA = permute(A, perm, perm)
	
	I_weighted = zeros(nnz(AA),1)
	J_weighted = zeros(nnz(AA),1)
	V_weighted = zeros(nnz(AA),1)
	# nnz(AA) upper bound on nnz(DD). Cut zeros later
	I_DD = zeros(nnz(AA),1)
	J_DD = zeros(nnz(AA),1)
	V_DD = zeros(nnz(AA),1)
	c_layer = 0
	AA_weighted_list_ind = 1
	DD_list_ind = 1
	const_ID = 0
	const_ONES = 0
	
	for i=1:n
		max_x = max.(x[i],x')
		AA_rowblock = AA[(i-1)*L+1:i*L,:]
		DD_rowblock1 = sum(AA_rowblock, dims=1)
		DD_rowblock2 = reshape(AA_rowblock*kron(sparse(I,n,n),ones(L,1)), (1,n*L))
		N1_row = reshape(sum(reshape(DD_rowblock1,(L,n)), dims=1),(1,n))
		N2_row = L^2*ones(1,n) - N1_row
		
		# blocks w/o edges don't contribute
		zero_inds = (N1_row.*N2_row).==0
		N1_row[zero_inds] .= Inf
		N2_row[zero_inds] .= Inf
		
		# diagonal matrix in n x nL format -- to be multiplied (elementwise) by
		# repmat(c_bar',[n,n]) and then summed up
		DD_row = sparse((DD_rowblock1+DD_rowblock2).*(kron(max_x.*(1 ./ N1_row + 1 ./ N2_row),ones(1,L))))
		DD_row_nz = findnz(DD_row)
		J = Int64.(DD_row_nz[2])
		V = DD_row_nz[3]
		I_DD[DD_list_ind:DD_list_ind+length(J)-1] = i*ones(1,length(J))
		J_DD[DD_list_ind:DD_list_ind+length(J)-1] = J
		V_DD[DD_list_ind:DD_list_ind+length(J)-1] = V
		DD_list_ind = DD_list_ind + length(J)
		
		# sparse AA block matrix -- to be put in an inner product with repmat(c_bar,[n,1])
		AA_rowblock_nz = findnz(AA_rowblock)
		I_ = Int64.(AA_rowblock_nz[1])
		J = Int64.(AA_rowblock_nz[2])
		V = AA_rowblock_nz[3]
		J_weighted[AA_weighted_list_ind:AA_weighted_list_ind+length(I_)-1] = J
		I_weighted[AA_weighted_list_ind:AA_weighted_list_ind+length(I_)-1] = (i-1)*L*ones(size(J)) + I_
		ind = Int64.(ceil.(J./L))
		V_weighted[AA_weighted_list_ind:AA_weighted_list_ind+length(I_)-1] = ((1 ./N1_row[ind] + 1 ./N2_row[ind]).*max_x[ind].*V)
		AA_weighted_list_ind = AA_weighted_list_ind + length(I_)
		
		# identity part -- to be multiplied by sum(x)
		const_ID = const_ID + 2*sum((L-1/2) .*max_x ./N2_row)
		
		# ones matrix part -- to be multiplied by sum(x)^2
		const_ONES = const_ONES + sum(max_x./N2_row)
		
		# normalization constant
		c_layer = c_layer + sum(max_x)
	end
	
	AA_weighted = sparse(Int64.(I_weighted[:,1]), Int64.(J_weighted[:,1]), V_weighted[:,1], n*L, n*L)
	DD = sparse(Int64.(I_DD[1:DD_list_ind-1]), Int64.(J_DD[1:DD_list_ind-1]), V_DD[1:DD_list_ind-1], n, n*L)
	
	c_bar = zeros(L,1)
	QUBO_vals_layer_block = zeros(L,1)
	for k=1:L
		c_bar[c_ind[k]] = 1
		c_bar_block = repeat(c_bar, outer=[n,1])
		QUBO_vals_layer_block[k] = sum(sum(DD, dims=1)'.*repeat(c_bar, outer=[n,1])) - const_ID*sum(c_bar) - only(c_bar_block'*AA_weighted*c_bar_block) + const_ONES*sum(c_bar)^2
		QUBO_vals_layer_block[k] = QUBO_vals_layer_block[k]/c_layer
	end
	
	QUBO_layer = findmax(QUBO_vals_layer_block)
	Q_layer = QUBO_layer[1]
	s_layer = QUBO_layer[2]
	println("Q_layer = ", Q_layer)
	println("s_layer = ", s_layer[1])
	
end

## evaluate QUBO objective functions
run_QUBO(A,n,L,x,x_ind,c,c_ind)

