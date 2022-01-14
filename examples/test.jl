using LinearAlgebra
using DynamicPolynomials
using SwitchOnSafety
using Combinatorics
using SparseArrays
using JuMP, Ipopt, MosekTools,NLopt




include("../src/RandomTrajectories.jl")
include("../src/AlgebraicLift.jl")
include("../src/ScenarioOpti.jl")
include("../src/WhiteBoxAnalysis.jl")



dim = 2; numMode = 2; dimIn = 1

numScen_budget = 1000

A = [[1 1; 0 1], [1 0; 1 1]]
B = 2*rand(Float64, (dim, dimIn)).-1

jsrbound = white_box_jsr(A)
println("JSR: $jsrbound")

gaTrue,K = white_box_stabilization_quad(A,B)
jsrboundclosed = white_box_jsr([Ai+B*K for Ai in A])
println("White-box stabilization: $gaTrue")
println("JSR closed: $jsrboundclosed")


(state0_budget,state_budget) = generate_trajectories(1,A,numScen_budget)



batchsize = 1000
K0 = zeros(dimIn,dim)
K1,ga1,P1 = soslyap_alternating(state0_budget,state_budget;B=B,d=1,batchsize=batchsize,K0=K0,tol=1e-3)
println("gamma1: $ga1 || K1: $K1")
K2,ga2,P2 = soslyap_alternating(state0_budget,state_budget;B=B,d=2,batchsize=batchsize,K0=K0,tol=1e-3)
println("gamma2: $ga2 || K2: $K2")

Aclose1 = [Ai+B*K1 for Ai in A]
Aclose2 = [Ai+B*K2 for Ai in A]
jsrboundclose1 = white_box_jsr(Aclose1)
jsrboundclose2 = white_box_jsr(Aclose2)

println(repeat('*', 80))
println("White-box stabilization: $gaTrue")
println("JSR closed: $jsrboundclosed")
println("JSR closed 1: $jsrboundclose1")
println("JSR closed 2: $jsrboundclose2")
println(repeat('*', 80))
