using LinearAlgebra
using DynamicPolynomials
using SwitchOnSafety
using Combinatorics
using SparseArrays
using JuMP, MosekTools
using SpecialFunctions




include("RandomTrajectories.jl")
include("AlgebraicLift.jl")
include("ScenarioOpti.jl")
include("JSRCompute.jl")
include("ProbabilisticCertificates.jl")


dim = 3; numMode = 3; dimIn = Integer(floor(dim/2)+1)

numScen_budget = 20000

#=
while true
    A = [2*rand(Float64, (dim, dim)).-1 for i=1:numMode]#generate_switched_linear_systems(numMode,dim)
    B = 2*rand(Float64, (dim, dimIn)).-1

    jsrbound = white_box_jsr(A)
    println("JSR: $jsrbound")
    rhomax = 0
    for i in 1:numMode
        rhoA = maximum(abs.(eigvals(A[i])))
        if rhomax < rhoA
            rhomax = rhoA
        end
    end
    println("Eigenvalue max: $rhomax")

    if jsrbound > 1.5 && rhomax < jsrbound*0.9
        gaTrue = white_box_stabilization_quad(A,B)
        println("White-box stabilization: $gaTrue")

        if gaTrue< 0.95

            (state0_budget,state_budget) = generate_trajectories(1,A,numScen_budget)

            batchsize = 1000
            N = 1000

            K0 = zeros(dimIn,dim)
            K,ga,P = soslyap_alternating(state0_budget[:,1:N],state_budget[:,1:N],B,1,batchsize,K0)
            if ga < 0.9
                println("A: $A")
                println("B: $B")
                break
            end
        end
    end
end
=#

#=
A = [[-0.6 0.92 -0.75; -0.75 -0.66 0.50; 0.16 -0.29 0.0082], [-0.77 0.56 -0.61; 0.79 -0.82 0.12; 0.54 0.62 -0.085], [0.16 -0.065 -0.088; 0.68 -0.57 -0.97; 0.35 -0.0034 -0.54]]
B = [-0.73 -0.31; 0.75 -0.86; 0.39 -0.27]

A = [[1 0.37 -0.19; -0.094 0.2 -0.38; -0.36 0.42 -0.32], [-0.077 -0.63 -0.81; -0.68 0.80 0.11; -0.14 0.94 0.41], [0.39 0.77 -0.16; -0.82 -0.52 -0.55; 0.028 0.83 -0.54]]
B = [-0.5 -0.56; -0.18 0.15; -0.054 0.36]
jsrbound = white_box_jsr(A)
println("JSR: $jsrbound")
(state0_budget,state_budget) = generate_trajectories(1,A,numScen_budget)
batchsize = 1000
N = 1000

K0 = zeros(dimIn,dim)
K,ga,P = soslyap_alternating(state0_budget[:,1:N],state_budget[:,1:N],B,1,batchsize,K0)
=#

#=
Order: 1 || Convergence rate (terminated): 0.8430537588968512
A: [[-0.1144228439831596 -0.501187369965538 -0.37125840489355033; 0.9760642363492855 0.1533976310090046 0.09312238413571805; -0.040636488385452196 -0.9346732339021617 0.8229542595705692], [-0.07241308618438724 0.8749984872431962 -0.4116184320581566; 0.48494778360903723 0.8825488953718352 -0.8296138961714696; -0.8291573747193985 0.44597317820605964 0.5167163786597384], [0.5408480280947159 0.11677922614953617 0.4442097583032836; 0.815758478465916 0.8052196221382406 0.21943815969197855; -0.24356959688956836 -0.9321903504166942 -0.5273102925517699]]
B: [-0.6144458431058699 -0.041012426944017744; 0.586779798444113 0.7523349112829041; -0.7764737978310268 -0.28489346457562803]
=#


#A = [[-0.11 -0.50 -0.37; 0.98 0.15 0.09; -0.04 -0.93 0.82], [-0.07 0.87 -0.41; 0.48 0.88 -0.83; -0.83 0.45 0.52], [0.54 0.12 0.44; 0.82 0.81 0.22; -0.24 -0.93 -0.53]]
#B = [-0.61 -0.04; 0.59 0.75; -0.78 -0.28]

A = [[-0.1 -0.50 -0.4; 1 0.2 0.1; 0 -0.9 0.8], [-0.1 0.9 -0.4; 0.5 0.9 -0.8; -0.8 0.5 0.5], [0.5 0.1 0.4; 0.8 0.8 0.2; -0.2 -0.9 -0.5]]
B = [-0.6 0; 0.6 0.8; -0.8 -0.3]

jsrboundopen= white_box_jsr(A)
println("JSR open loop: $jsrboundopen")
gaTrue = white_box_stabilization_quad(A,B)
println("White-box stabilization: $gaTrue")

(state0_budget,state_budget) = generate_trajectories(1,A,numScen_budget)

beta = 0.01
JSRbound = []
numSample = []

K0 = zeros(dimIn,dim)
tol=1e-3
for N in 1000:500:numScen_budget
    state0 = state0_budget[:,1:N]

    state = state_budget[:,1:N]
    K,jsr_bound = probabilistc_stability_certificate(state0,state,B,numMode,1,N,K0,beta,tol)


    if jsr_bound <100  && ~isempty(JSRbound)
        err = JSRbound[end]-jsr_bound
        if abs(err) < 1e-3
            break
        end
    end
    push!(JSRbound,jsr_bound)
    push!(numSample,N)
    println("N: $N || JSR bound: $jsr_bound")
    println("List all JSR bounds: $JSRbound")


end






