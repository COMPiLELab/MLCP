# MLCP
Core-periphery detection in multilayer networks

This code implements the numerical experiments of the

**Paper:**
[1] Kai Bergermann and Francesco Tudisco. Core-periphery detection in multilayer networks. Preprint, [arXiv:2412.04179](http://arxiv.org/abs/2412.04179). 2024.

This repository contains:

**License:**
 - LICENSE: GNU General Public License v2.0

**Directories:**
 - data: contains files for EUAir experiments. The files for the remaining experimens (openalex and WIOT) are available under https://doi.org/10.5281/zenodo.14231870
 - helpers: Matlab files 'build_adjacency_openalex_fully_weighted.m' and 'build_adjacency_WIOT.m' constructing sparse supra-adjacency matrices from the txt files in the 'data' directory
 - results: .txt-files containing node and layer core vectors optimized by the nonlinear spectral method (MLCP.jl)
 
**Scripts:**
 - MLNSM.jl: julia code applying [Algorithm 1, 1] to the multilayer networks described in [1]
 - QUBO.m: Matlab code determining node and layer core sizes based on the node and layer coreness vectors optimized by 'MLNSM.jl'
