using LinearAlgebra
using DynamicPolynomials
using SwitchOnSafety
using Combinatorics
using SparseArrays
using JuMP, MosekTools
using SpecialFunctions

include("../src/RandomTrajectories.jl")
include("../src/AlgebraicLift.jl")
include("../src/ScenarioOpti.jl")
include("../src/ProbabilisticCertificates.jl")
include("../src/WhiteBoxAnalysis.jl")


dim = 2; numMode = 3; dimIn = 1#Integer(floor(dim/2)+1)

numScen_budget = 2000

A = [[0.7 0.16; 1.1 -1.1], [0.4 -0.84; 0.83 0.35], [0.37 0.96; 0.34 -1.2]]
B = reshape([-0.9; -1.2],2,1)

jsrboundopen= white_box_jsr(A)
println("JSR open loop: $jsrboundopen")
gaWB,KWB = white_box_stabilization_quad(A,B)
println("White-box stabilization: $gaWB")
jsrboundcloseWB = white_box_jsr([Ai+B*KTrue for Ai in A])
println("White-box jsr: $jsrboundcloseWB")


(state0_budget,state_budget) = generate_trajectories(1,A,numScen_budget)
K0 = zeros(dimIn,dim)

K,jsr_bound,ga,P = probabilistc_stability_certificate(state0_budget,state_budget;B=B,numMode=numMode,d=1,batchsize=numScen_budget,K0=K0,beta=0.01,tol=1e-3)

println("JSR bound: $jsr_bound")

Aclose = [Ai+B*K for Ai in A]
jsrboundclose = white_box_jsr(Aclose)
println("JSR closed: $jsrboundclose")
println("gamma: $ga")
println("P: $P")





