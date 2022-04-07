using LinearAlgebra
using DynamicPolynomials
using SwitchOnSafety
using Combinatorics
using SparseArrays
using JuMP, Ipopt, MosekTools,NLopt
using SpecialFunctions




include("../src/RandomTrajectories.jl")
include("../src/AlgebraicLift.jl")
include("../src/ScenarioOpti.jl")
include("../src/ProbabilisticCertificates.jl")
include("../src/WhiteBoxAnalysis.jl")



dim = 2; numMode = 2; dimIn = 1

numScen_budget = 1000

Asave = [zeros(Float64, (dim, dim)) for i=1:numMode]
Bsave = zeros(Float64, (dim, dimIn))
ratio = 1


while true

A = [10*rand(Float64, (dim, dim)).-5 for i=1:numMode]
#A = [[1 1; 0 1], [1 0; 1 1]]
#println(A)
jsrbound = white_box_jsr(A)


if jsrbound > 1 #rhomax < jsrbound*0.9 && 


B = 10*rand(Float64, (dim, dimIn)).-5
gaTrue,K = white_box_stabilization_quad(A,B)
#println(K)
if isempty(K)
    println("A: $A")
    println("B: $B")
end
jsrboundclosed = white_box_jsr([Ai+B*K for Ai in A])

println("A: $Asave")
println("B: $Bsave")
println("Ratio: $ratio")

if gaTrue > 1

    (state0_budget,state_budget) = generate_trajectories(1,A,numScen_budget)

    batchsize = numScen_budget

    K0 = zeros(dimIn,dim)

    K1,ga1,P1 = soslyap_alternating(state0_budget,state_budget;B=B,d=1,batchsize=batchsize,K0=K0,tol=1e-4)
    if ga1 > 1

        K2,ga2,P2 = soslyap_alternating(state0_budget,state_budget;B=B,d=2,batchsize=batchsize,K0=K0,tol=1e-4)

        Aclose1 = [Ai+B*K1 for Ai in A]
        jsrboundclose1 = white_box_jsr(Aclose1)
        Aclose2 = [Ai+B*K2 for Ai in A]
        jsrboundclose2 = white_box_jsr(Aclose2)

        println(repeat('*', 80))
        println("White-box stabilization: $gaTrue")
        println("JSR closed: $jsrboundclosed")
        println("JSR closed 1: $jsrboundclose1")
        println("JSR closed 2: $jsrboundclose2")

        println("JSR closed Prob1: $ga1")
        println("JSR closed Prob2: $ga2")

        println(repeat('*', 80))

        ratiotemp = ga2/ga1

        if ratiotemp < ratio && jsrboundclose2 < jsrboundclose1 && jsrboundclose1 < 1.5*jsrboundclosed && jsrboundclose2 < 0.95*jsrboundclosed
            global Asave = A 
            global Bsave = B
            global ratio = ratiotemp
            println("jsr_bound1: $ga1 || K1: $K1")
            println("jsr_bound2: $ga2 || K2: $K2")
        end

        if ratio < 0.7
            println("A: $Asave")
            println("B: $Bsave")
            println("Ratio: $ratio")
            break
        end
    end
end
end



end
