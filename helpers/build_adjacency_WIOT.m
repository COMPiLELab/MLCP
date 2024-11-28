
% Constructs sparse supra-adjacency matrix from the txt file containing
% edge information specified below and saves it in .mat format.

year = 2014;
data=load(['../data/WIOT_multilayer_network_edges_',num2str(year),'.txt']);

n_edges = size(data,1);
L=max(max(data(:,2)), max(data(:,4)));
n=max(max(data(:,1)), max(data(:,3)));

A = sparse((data(:,2)-ones(n_edges,1))*n+data(:,1),(data(:,4)-ones(n_edges,1))*n+data(:,3),data(:,5),n*L,n*L);

save(['../data/WIOT_',num2str(year),'_supra_adj.mat'],'A')