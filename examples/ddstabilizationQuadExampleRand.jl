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


dim = 4; numMode = 4; dimIn = Integer(floor(dim/2)+1)

numScen_budget = 20000


beta = 0.01
batchsize = 1000
tol=1e-3

A = [2*rand(Float64, (dim, dim)).-1 for i=1:numMode]#generate_switched_linear_systems(numMode,dim)
B = 2*rand(Float64, (dim, dimIn)).-1

jsrboundopen= white_box_jsr(A)
println("JSR open loop: $jsrboundopen")
gaTrue = white_box_stabilization_quad(A,B)
println("White-box stabilization: $gaTrue")

(state0_budget,state_budget) = generate_trajectories(1,A,numScen_budget)

jsrMat = []

K0 = zeros(dimIn,dim)

for N in 1000:1000:numScen_budget
    state0 = state0_budget[:,1:N]
    state = state_budget[:,1:N]
    K,jsr_bound = probabilistc_stability_certificate(state0,state,B,numMode,1,batchsize,K0,beta,tol)
    push!(jsrMat,jsr_bound)
    println(jsrMat)
end




#n,M,m = 3,3,2
#Realization 2: Any[1.9726581866167456, 1.4971871770308698, 1.3012298819739316, 1.2304864815756291, 1.2117494674779883, 1.1814407656336983, 1.17194808268181, 1.1597211928556568, 1.1292330213344557, 1.1222576194378215, 1.1165670639963532, 1.1226029696179107, 1.1162240486862525, 1.1245394711764556, 1.1098349065892859, 1.1004923938205808, 1.1050231964144182, 1.1000114470798623, 1.0994239898509492, 1.097569760267111] 
#Realization 1: 1.51408  1.34  1.27905  1.25378  1.23393  1.21777  1.21349  1.19846  1.19579  1.19281  1.18939  1.19761  1.19486  1.18352  1.1897  1.18811  1.18936  1.18811  1.17449  1.175

#n,M,m = 4,4,3
#Any[Inf, 5.568732417102386, 3.1844269410550567, 2.5502244067679665, 1.9360491898822136, 2.2056802180721595, 1.9082474595806171, 2.1344757523234703, 1.8217217360361957, 1.789212827758543, 1.7666912410244968, 1.7216899728253687, 1.7112870808324216, 1.6919229616094742, 1.6750239786596284, 1.5932067724549857, 1.581661326289367, 1.584226846994496, 1.640141131513329, 1.5589344268487348]

#n,M,m = 4,6,3

#n,M,m = 6,6,4
