
% Constructs sparse supra-adjacency matrix from the txt file containing
% edge information specified below and saves it in .mat format.

year = 2000;
loadstr = ['../data/edges_complex_networks_all_concepts_fully_weighted_',num2str(year),'.txt'];
data=load(loadstr);

n_edges = size(data,1);
L = 19;
n = 53423;

A = sparse(data(:,2)*n+data(:,1)+ones(n_edges,1),data(:,4)*n+data(:,3)+ones(n_edges,1),data(:,5),n*L,n*L);

savestr = ['../data/openalex_fully_weighted_',num2str(year),'_supra_adj.mat'];
save(savestr,'A')