using LinearAlgebra
using DynamicPolynomials
using SwitchOnSafety
using Combinatorics
using SparseArrays
using JuMP, Ipopt, MosekTools,NLopt
using SpecialFunctions
using ControlSystems



include("../src/RandomTrajectories.jl")
include("../src/AlgebraicLift.jl")
include("../src/ScenarioOpti.jl")
include("../src/ProbabilisticCertificates.jl")
include("../src/WhiteBoxAnalysis.jl")



dim = 3; numMode = 4; dimIn = 3

numScen_budget = 10000

c = 1375
R12 = 1.5
Ro12 = 3
Ro3 = 2.7
tau = 3*60
A = []
for R13 in [0.8 1.2]
    for R23 in [0.8 1.2]
        Ai = [1-tau/c*(1/R12+1/R13+1/Ro12) tau/c/R12 tau/c/R13;tau/c/R12 1-tau/c*(1/R12+1/R23+1/Ro12) tau/c/R23;tau/c/R13 tau/c/R23 1-tau/c*(1/R13+1/R23+1/Ro3)]
        push!(A,Ai)
    end
end

B = tau/c*Matrix(I,dimIn,dimIn)


Q = Matrix(I,dim,dim)
R = 0.02*Matrix(I,dimIn,dimIn)

(Kw,Pw) = white_box_LQR(A,B,Q,R)
println("K white-box: $Kw")

(state0_budget,state_budget) = generate_trajectories(1,A,numScen_budget)

K0 = zeros(dimIn,dim)

K,xi,P,ratio = probabilistc_LQR_certificate(state0_budget[:,1:N],state_budget[:,1:N];B=B,numMode=numMode,Q=Q,R=R,kappaU=1e2,K0=K0,beta=0.01,tol=1e-4)

state = [15;10;12]
X = zeros(dim,10)
for i in 1:10
    global state
    X[:,i] = state
    b = rand((1:numMode),1)
    u = K*state
    state = A[b[1]]*state+B*u
end

X = X.+24

using Plots
gr(size = (400, 300))
fn = plot(0:3:9*3, Any[X[1,:],X[2,:],X[3,:]],label = ["Zone 1" "Zone 2" "Zone 3"], line = [:solid :solid :solid], lw = 2)
xlabel!("Time (minute)")
ylabel!("Temperature (°C) ")

#=
for N in 500:500:numScen_budget
    K,xi,P,ratio = probabilistc_LQR_certificate(state0_budget[:,1:N],state_budget[:,1:N];B=B,numMode=numMode,Q=Q,R=R,kappaU=1e2,K0=K0,beta=0.01,tol=1e-4)
    push!(xisq,xi)
    push!(ratiosq,ratio)
    println("xi sequence: $xisq")
    println("ratio sequence: $ratiosq")
    println("K: $K")
    println("K white-box: $Kw")
    println("P: $P")
    println("P white-box: $Pw")
end=#

#=
using Plots
using LaTeXStrings
gr(size = (400, 300))
fn = plot(500:500:12000, Any[feas,fill(1.0,24)],label = "", line = [:solid :dashdot], lw = 2)
xlabel!(L"N")
ylabel!(L"\bar{\xi}(\omega_N)")
savefig(fn,"fn.png")
=#

#feas = xisq./ratiosq
#[1.1462996266804206, 1.0304548488231406, 1.1839711666921258, 0.9796333488331512, 0.9668010693346112, 1.1983116266741682, 1.0747101169748587, 0.953405955179486, 0.9531081705886429, 0.9502858772790718]
#K: [-0.028447388063758478 -0.02397013534333861 -0.024868877374478904; -0.022049588263423403 -0.029909164375603235 -0.025289972953130273; -0.02557500118128832 -0.02521458873586027 -0.025649666991379057]   

#xisq =[0.9506955125447197, 0.9500814227738398, 0.9496688890425853, 0.9506770820573243, 0.9491450126708667, 0.9491309513543958, 0.9512678391859847, 0.9511263256916642, 0.9509734949035534, 0.9496753424296864, 0.9495745388568111, 0.9517758890790594, 0.9517405648377967, 0.9516439742658995, 0.9498230924269193, 0.9500617518717405, 0.9499383280065629, 0.9498282336931118, 0.9516072138900373, 0.9518346791078194, 0.9518833690400422, 0.9518458883283956, 0.95181760735376, 0.9520342911681299]
#ratiosq = [0.32929113429561285, 0.6359092714236014, 0.7458849122101749, 0.8080460152288503, 0.8388407832548765, 0.8651945928045293, 0.8807624981871738, 0.895048424191853, 0.9057136189419326, 0.9146100154266903, 0.9218573770210922, 0.9277461652693918, 0.9328689844736177, 0.9374298867002537, 0.9413125864511712, 0.9503108804475847, 0.9529825663527286, 0.9554076570766022, 0.9575460410445409, 0.9595593129229043, 0.9614069403433664, 0.963069357269896, 0.9646071007278422, 0.965914139694424]


# K = [-3.379476201830191 -0.5565284859098218 -0.6653948967484442; -0.5610849053980036 -3.374365413285184 -0.665561226645545; -0.664814150076262 -0.6724159301492334 -3.2436777438358977]
# P = [1.4041357466418602 0.11383342685787007 0.13332235372007795; 0.11383342685787007 1.4041274464054574 0.13332286522268716; 0.13332235372007795 0.13332286522268716 1.3787232916679992]

#P white-box: [1.384392543034221 0.10849589561899299 0.1270429070953619; 0.10849589561899299 1.3843925454395583 0.1270429068809821; 0.1270429070953619 0.1270429068809821 1.3602060587775264]
#K white-box: [-3.173628196889124 -0.4840291283071396 -0.5938068939821004; -0.48402965221411914 -3.1736302770962785 -0.5938067500136359; -0.5881971655090553 -0.588195839338544 -3.031968422173231]

