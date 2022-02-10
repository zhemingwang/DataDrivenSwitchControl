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

numScen_budget = 15000

#A = [[0.7218345749085846 0.013504049625046477; -0.763247200146052 0.823150250105976], [0.726724028000425 0.8385770769048322; -0.7232303846748045 0.19537267020502425]]
#B = [0.11672890767112509; 0.4312206258057798]
#A = [[-0.3734630321404846 -0.6513237917108774; 0.611847898506932 -0.14691027837992898], [0.3747919292302151 -0.08207253587307806; -0.8177042535328924 0.9426235103657543]]
#B = [-0.9774997862889991; -0.8641681288852348]

A = [[-2.4863 4.9076; 1.5580 -3.2878], [-0.2529 -1.0603; 2.3881 4.9242]]
B = [1.1435; 1.9476]


##A = [[1.2847752917460404 -4.294823846678941; 0.3680548348155881 0.014732983067902161], [1.1712220013664698 -0.8818060555034961; 1.6530187731608859 -2.3732736764432127]]
##B = [0.32980020816855316; 2.992543632417817]

##A = [[-0.7758 -0.9779; 0.9748 0.7367], [0.7864 -0.2455; 0.9516 0.5357]]
##B = [-0.6358; -0.04498]

##A = [[-0.8350343770388022 -0.006107458495185014; 1.6179435194682048 0.08247284108420949], [-0.5991942074864163 -1.7792515399993318; 1.1279527670550538 0.9404192977577894]]
##B = [0.9917434505323639; 1.3179954593019438]

##A = [[-0.835 -0.006; 1.617 0.0825], [-0.599 -1.779; 1.128 0.9404]]
##B = [0.992; 1.318]

##A= [[2.196 3.187; 1.758 2.183], [2.4684 4.844; 1.383 4.565]]
##B= [-2.212; -1.980]


##A= [[2.1962930715063305 3.187441172268997; 1.7577099273813346 2.183228120757372], [2.4683584311233293 4.84422595233246; 1.3828879364062585 4.565095890995014]]
##B= [-2.212099386961055; -1.9795927901090082]

##A=[[2.671281119684634 -4.876263859500199; -4.392395002400294 2.7156007276877574], [4.645925349681839 2.206066227650134; -4.267978590442674 1.7848087937926742]]
##B= [3.743666731025746; -1.4650217059382165]


##A = [[-0.7757566511432756 -0.9778654656915116; 0.97480024345306 0.7366948958837525], [0.7863913424227396 -0.2454901059378174; 0.9515607611942283 0.535698625247981]]
##B = [-0.6357718059899597; -0.0449794151233891]

##A = [[0.3992966486009579 -0.8825975300456066; 0.9466203607162913 0.8220671916996221], [-0.05865322607237777 -0.32416336552150593; -0.5565696843656855 -0.8985422238439993]]
##B = [-0.5919429619683738; 0.18179898964335361]

##A = [[0.8388521952671368 -0.14857670536251177; 0.8633807216781597 0.490362061793125], [0.6013596101878895 0.6697432384693109; -0.12770751192259544 0.9150610500819121]]
##B = [0.9467520554779116; 0.20399193884377942]

B = reshape(B,dim,dimIn)
jsrbound = white_box_jsr(A)

gaTrue,K = white_box_stabilization_quad(A,B)
jsrboundclosed = white_box_jsr([Ai+B*K for Ai in A])

(state0_budget,state_budget) = generate_trajectories(1,A,numScen_budget)

K0 = zeros(dimIn,dim)

jrs_boundorder1 = []
jrs_boundorder2 = []
#=
for N in 200:100:1000
    K1,jsr_bound1 = probabilistc_stability_certificate(state0_budget[:,1:N],state_budget[:,1:N];B=B,numMode=numMode,d=1,batchsize=N,K0=K0,beta=0.01,tol=1e-3)
    K2,jsr_bound2 = probabilistc_stability_certificate(state0_budget[:,1:N],state_budget[:,1:N];B=B,numMode=numMode,d=2,batchsize=N,K0=K0,beta=0.01,tol=1e-3)

    append!(jrs_boundorder1,jsr_bound1)
    append!(jrs_boundorder2,jsr_bound2)

    Aclose1 = [Ai+B*K1 for Ai in A]
    jsrboundclose1 = white_box_jsr(Aclose1)
    Aclose2 = [Ai+B*K2 for Ai in A]
    jsrboundclose2 = white_box_jsr(Aclose2)

    println(repeat('*', 80))
    println("White-box stabilization: $gaTrue")
    println("JSR closed: $jsrboundclosed")
    println("N: $N|| JSR closed 1: $jsrboundclose1")
    println("N: $N|| JSR closed 2: $jsrboundclose2")

    println("N: $N|| JSR closed Prob1: $jsr_bound1")
    println("N: $N|| JSR closed Prob2: $jsr_bound2")

    println(repeat('*', 80))
    println(jrs_boundorder1)
    println(jrs_boundorder2)

end
=#


for N in 1000:1000:numScen_budget
    K1,jsr_bound1 = probabilistc_stability_certificate(state0_budget[:,1:N],state_budget[:,1:N];B=B,numMode=numMode,d=1,batchsize=1000,K0=K0,beta=0.01,tol=1e-4)
    K2,jsr_bound2 = probabilistc_stability_certificate(state0_budget[:,1:N],state_budget[:,1:N];B=B,numMode=numMode,d=2,batchsize=1000,K0=K0,beta=0.01,tol=1e-4)

    append!(jrs_boundorder1,jsr_bound1)
    append!(jrs_boundorder2,jsr_bound2)

    Aclose1 = [Ai+B*K1 for Ai in A]
    jsrboundclose1 = white_box_jsr(Aclose1)
    Aclose2 = [Ai+B*K2 for Ai in A]
    jsrboundclose2 = white_box_jsr(Aclose2)

    println(repeat('*', 80))
    println("White-box stabilization: $gaTrue || K: $K")
    println("JSR closed: $jsrboundclosed")
    println("N: $N || JSR closed 1: $jsrboundclose1 || K1: $K1")
    println("N: $N || JSR closed 2: $jsrboundclose2 || K2: $K2")

    println("N: $N || JSR closed Prob1: $jsr_bound1")
    println("N: $N || JSR closed Prob2: $jsr_bound2")

    println(repeat('*', 80))
    println(jrs_boundorder1)
    println(jrs_boundorder2)

end

#=
using Plots
gr(size = (400, 400))
fn = plot(400:200:Int64(5e3), Any[jrs_boundorder1,jrs_boundorder2], label = ["Quadratic stabilization" "SOS stabilization (d=2)"],line = [:solid :solid], lw = 2)
xlabel!("N")
ylabel!("Upper bound of the JSR")
savefig(fn,"fn.png")
=#


