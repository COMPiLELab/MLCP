
# Implements the nonlinear spectral method for core-periphery detection in multilayer networks
# presented in [Algorithm 1, 1].
# 
# [1] Kai Bergermann and Francesco Tudisco. Core-periphery detection in multilayer networks. Preprint.

using LinearAlgebra, SparseArrays, Printf, Plots, Random, Arpack, ProgressBars


struct Sparse4Tensor
	I::Vector{Int64}
	J::Vector{Int64}
	K::Vector{Int64}
	L::Vector{Int64}
	V::Vector{Float64}
	n::Int64
	n_L::Int64
end


function gradfx(A::Sparse4Tensor, x::Vector{Float64}, c::Vector{Float64}, a, b)
	Fx = zeros(Float64, A.n)
	# Computes the gradient of the objective function [eq. (2), 1] w.r.t. the vector x

	for (i, j, k, l, v) in zip(A.I, A.J, A.K, A.L, A.V)
		Fx[i] += v * x[i]^(a-1) * (x[i]^a + x[j]^a)^(1/a-1) * (c[k]^b + c[l]^b)^(1/b)
		Fx[j] += v * x[j]^(a-1) * (x[i]^a + x[j]^a)^(1/a-1) * (c[k]^b + c[l]^b)^(1/b)
	end
	
	return Fx
end


function gradfc(A::Sparse4Tensor, x::Vector{Float64}, c::Vector{Float64}, a, b)
	Fc = zeros(Float64, A.n_L)
	# Computes the gradient of the objective function [eq. (2), 1] w.r.t. the vector c

	for (i, j, k, l, v) in zip(A.I, A.J, A.K, A.L, A.V)
		Fc[k] += v * c[k]^(b-1) * (c[k]^b + c[l]^b)^(1/b-1) * (x[i]^a + x[j]^a)^(1/a)
		Fc[l] += v * c[l]^(b-1) * (c[k]^b + c[l]^b)^(1/b-1) * (x[i]^a + x[j]^a)^(1/a)
	end
	
	return Fc
end


function MLNSM(A::Sparse4Tensor, x0::Vector{Float64}, c0::Vector{Float64}, a, b, p, q, tol, maxIter)
	# Nonlinear spectral method [Algorithm 1, 1].
	# Input: 	A:		Adjacency tensor.
	# 		x0: 		Initial node coreness vector.
	#		c0:		Initial layer coreness vector. 
	#		a:		Parameter alpha.
	#		b:		Parameter beta.
	#		p:		Parameter p from norm constraint on vector x.
	#		q:		Parameter q from norm constraint on vector c.
	#		tol:		Tolerance between consecutive iterates (stopping criterion).
	#		maxIter:	Maximum number of iterations (stopping criterion).
	
	pp = p/(p-1)
	qq = q/(q-1)

	x0 = x0/norm(x0,pp)
	c0 = c0/norm(c0,qq)
	
	x = x0
	c = c0
	
        println("Multilayer Nonlinear Power Method:")
        println("-------------------------------")
        println("alpha:\t\t$a\nbeta:\t\t$b\np:\t\t$p\nq:\t\t$q\ntol:\t\t$tol")
	
	t = @elapsed begin
		for i = 1:maxIter
			x = gradfx(A,x0,c0,a,b)
			x = x/norm(x,pp)
			x = x.^(1/(p-1))
					
			c = gradfc(A,x0,c0,a,b)
			c = c/norm(c,qq)
			c = c.^(1/(q-1))
			
			@printf("Iteration %d, x error norm: %e, c error norm: %e\n", i, norm(x-x0), norm(c-c0))
			
			if norm(x-x0)<tol && norm(c-c0)<tol
				println("Num iter:\t$i")
				break
			else
				x0 = x
				c0 = c
			end
		end
	end
	println("Exec time:\t$t sec")
        println("-------------------------------")
		
	return x, c
end


function setup_adjacency_tensor(problem::String)
	# Sets up the fourth-order adjacency tensor representing the multilayer network.
	# For data availability, see https://doi.org/10.5281/zenodo.14231869 as well as the
	# instructions in the repository's READme.
	
	Random.seed!(1234)
	
	if first(problem,28)=="openalex_fully_weighted_year"
		# Weighted version of the citation network of complex network scientists discussed in the main text [1].
	
		year = last(problem,4)
		
		n = 53423
		n_L = 19
		
		filename = "data/edges_complex_networks_all_concepts_fully_weighted_"*year*".txt"
		n_lines = countlines(filename)
		file = open(filename, "r")
		
		I, J, K, L, V = zeros(Int32,n_lines), zeros(Int32,n_lines), zeros(Int32,n_lines), zeros(Int32,n_lines), zeros(Float32,n_lines)
		I_supra, J_supra = zeros(Int32,n_lines), zeros(Int32,n_lines)
		
		i = 1
		for line in eachline(file)
			node_id_orig,layer_id_orig,node_id_dest,layer_id_dest,edge_weight = split(line,",")
						
			I[i] = parse.(Int32,node_id_orig) + 1
			J[i] = parse.(Int32,node_id_dest) + 1
			K[i] = parse.(Int32,layer_id_orig) + 1
			L[i] = parse.(Int32,layer_id_dest) + 1
			V[i] = parse.(Float32,edge_weight)
			
			I_supra[i] = n*(K[i]-1) + I[i]
			J_supra[i] = n*(L[i]-1) + J[i]
			
			i += 1
		end
		
		if n != max(maximum(I),maximum(J)) || n_L != max(maximum(K),maximum(L))
			println("Warning: Number of nodes and layers is not as expected!")
		end
		
		A = Sparse4Tensor(I, J, K, L, V/maximum(V), n, n_L)
		
	elseif first(problem,13)=="openalex_year"
	# Unweighted version of the citation network of complex network scientists discussed in the supplementary materials of [1].
	
		year = last(problem,4)
		
		n = 53423
		n_L = 19
		
		filename = "data/edges_complex_networks_all_concepts_"*year*".txt"
		n_lines = countlines(filename)
		file = open(filename, "r")
		
		I, J, K, L, V = zeros(Int32,n_lines), zeros(Int32,n_lines), zeros(Int32,n_lines), zeros(Int32,n_lines), zeros(Float32,n_lines)
		I_supra, J_supra = zeros(Int32,n_lines), zeros(Int32,n_lines)
		
		i = 1
		for line in eachline(file)
			node_id_orig,layer_id_orig,node_id_dest,layer_id_dest,edge_weight = split(line,",")
						
			I[i] = parse.(Int32,node_id_orig) + 1
			J[i] = parse.(Int32,node_id_dest) + 1
			K[i] = parse.(Int32,layer_id_orig) + 1
			L[i] = parse.(Int32,layer_id_dest) + 1
			V[i] = parse.(Float32,edge_weight)
			
			I_supra[i] = n*(K[i]-1) + I[i]
			J_supra[i] = n*(L[i]-1) + J[i]
			
			i += 1
		end
		
		if n != max(maximum(I),maximum(J)) || n_L != max(maximum(K),maximum(L))
			println("Warning: Number of nodes and layers is not as expected!")
		end
		
		A = Sparse4Tensor(I, J, K, L, V/maximum(V), n, n_L)
		
	elseif problem=="EUAir"
	
		n = 417
		n_L = 37
		
		filename = "data/EUAir_edges.txt"
		n_lines = countlines(filename)
		file = open(filename, "r")
		
		I, J, K, L, V = zeros(Int32,n_lines), zeros(Int32,n_lines), zeros(Int32,n_lines), zeros(Int32,n_lines), zeros(Float32,n_lines)
		I_supra, J_supra = zeros(Int32,n_lines), zeros(Int32,n_lines)
		
		i = 1
		for line in eachline(file)
			node_id_orig,layer_id_orig,node_id_dest,layer_id_dest,edge_weight = split(line,",")
			
			I[i] = parse.(Int32,node_id_orig)
			J[i] = parse.(Int32,node_id_dest)
			K[i] = parse.(Int32,layer_id_orig)
			L[i] = parse.(Int32,layer_id_dest)
			V[i] = parse.(Float32,edge_weight)
			
			I_supra[i] = n*(K[i]-1) + I[i]
			J_supra[i] = n*(L[i]-1) + J[i]
			
			i += 1
		end
		
		if n != max(maximum(I),maximum(J)) || n_L != max(maximum(K),maximum(L))
			println("Warning: Number of nodes and layers is not as expected!")
		end
		
		A = Sparse4Tensor(I, J, K, L, V/maximum(V), n, n_L)
		
	elseif first(problem,4)=="WIOT"
		year = last(problem,4)
		
		n = 43
		n_L = 56
		
		filename = "data/WIOT_multilayer_network_edges_" * string(year) * ".txt"
		n_lines = countlines(filename)
		file = open(filename, "r")
		
		I, J, K, L, V = zeros(Int32,n_lines), zeros(Int32,n_lines), zeros(Int32,n_lines), zeros(Int32,n_lines), zeros(Float32,n_lines)
		I_supra, J_supra = zeros(Int32,n_lines), zeros(Int32,n_lines)
		
		i = 1
		for line in eachline(file)
			node_id_orig,layer_id_orig,node_id_dest,layer_id_dest,edge_weight = split(line,",")
						
			I[i] = parse.(Int32,node_id_orig)
			J[i] = parse.(Int32,node_id_dest)
			K[i] = parse.(Int32,layer_id_orig)
			L[i] = parse.(Int32,layer_id_dest)
			V[i] = parse.(Float32,edge_weight)
			
			I_supra[i] = n*(K[i]-1) + I[i]
			J_supra[i] = n*(L[i]-1) + J[i]
			
			i += 1
		end
		
		if n != max(maximum(I),maximum(J)) || n_L != max(maximum(K),maximum(L))
			println("Warning: Number of nodes and layers is not as expected!")
		end
		
		A = Sparse4Tensor(I, J, K, L, V/maximum(V), n, n_L)
		
	end
	
	return A, n, n_L
end


function run_MLCP(problem, alpha, beta, p, q, tol, maxIter)
	# Runs a given experiment from [1], displays some results, and saves the optimized coreness vectors.
	
	postfix = problem*"_a_"*string(alpha)*"_b_"*string(beta)*"_p_"*string(p)*"_q_"*string(q)
	
	A, n, L = setup_adjacency_tensor(problem)
	
	println("-------------------------------")
	println("Problem:\t", problem)
	println("# of nodes:\t", n)
	println("# of layers:\t", L)
	println("-------------------------------")

	x0 = ones(Float64, n)
	c0 = ones(Float64, L)

	x, c = MLNSM(A,x0,c0,alpha,beta,p,q,tol,maxIter)
	
	# print x and c vectors to file for post-processing
	io = open("results/x_vec_"*postfix*".txt", "w") do io
		for entry in x
			println(io, entry)
		end
	end
	io = open("results/c_vec_"*postfix*".txt", "w") do io
		for entry in c
			println(io, entry)
		end
	end
	
	# print top_k nodes and layers
	x_ind = sortperm(x,rev=true)
	c_ind = sortperm(c,rev=true)
	
	top_k = min(min(30,n),L)
	
	x_sorted = x[x_ind]
	println("x vector sorted, top ",top_k,": ",x_sorted[1:top_k])
	if first(problem,8)=="openalex"
		node_list = String[]
		file = open("data/openalex_node_names.txt", "r")
		for line in eachline(file)
			author_id,author_name = split(line,",")
			push!(node_list,author_name)
		end
		node_list_sorted = node_list[x_ind]
		println("top ",top_k," nodes: ", node_list_sorted[1:top_k])
	elseif problem=="EUAir"
		node_list = String[]
		file = open("data/EUAir_node_names.txt", "r")
		for line in eachline(file)
			push!(node_list,line)
		end
		node_list_sorted = node_list[x_ind]
		println("top ",top_k," nodes: ", node_list_sorted[1:top_k])
	elseif first(problem,4)=="WIOT"
		node_list = String[]
		file = open("data/WIOT_node_names.txt", "r")
		for line in eachline(file)
			push!(node_list,line)
		end
		node_list_sorted = node_list[x_ind]
		println("top ",top_k," nodes: ", node_list_sorted[1:top_k])
	end
    
	c_sorted = c[c_ind]
	println("c vector sorted, top ",top_k,": ",c_sorted[1:top_k])
	if first(problem,8)=="openalex"
		layer_list = String[]
		file = open("data/openalex_layer_names.txt", "r")
		for line in eachline(file)
			concept_id,concept_name = split(line,",")
			push!(layer_list,concept_name)
		end
		layer_list_sorted = layer_list[c_ind]
		println("top ",top_k," layers: ", layer_list_sorted[1:top_k])
	elseif problem=="EUAir"
		layer_list = String[]
		file = open("data/EUAir_layer_names.txt", "r")
		for line in eachline(file)
			push!(layer_list,line)
		end
		layer_list_sorted = layer_list[c_ind]
		println("top ",top_k," layers: ", layer_list_sorted[1:top_k])
	elseif first(problem,4)=="WIOT"
		layer_list = String[]
		file = open("data/WIOT_layer_names.txt", "r")
		for line in eachline(file)
			push!(layer_list,line)
		end
		layer_list_sorted = layer_list[c_ind]
		println("top ",top_k," layers: ", layer_list_sorted[1:top_k])
	end
end

# choose from: "openalex_fully_weighted_year_2000", "openalex_year_2000", "EUAir", "WIOT_2000"
# (openalex years range from 2000 to 2023 and WIOT years from 2000 to 2014)


# run_MLCP(problem, alpha, beta, p, q, tol, maxIter)
run_MLCP("WIOT_2000", 10, 10, 22, 22, 1e-08, 200)



