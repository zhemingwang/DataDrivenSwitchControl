# DataDrivenSwitchControl


This package implements quadratic and SOS methods for controlling switched linear systems using data. It requires a set of trajectories of the open-loop system.

## Dependencies 
* Combinatorics
* MosekTools
* SpecialFunctions

## Examples
Examples can be found in the folder examples.

```julia

dim = 3; numMode = 3; dimIn = 2

#A model is needed to generate data.
A = [[-0.1 -0.50 -0.4; 1 0.2 0.1; 0 -0.9 0.8], [-0.1 0.9 -0.4; 0.5 0.9 -0.8; -0.8 0.5 0.5], [0.5 0.1 0.4; 0.8 0.8 0.2; -0.2 -0.9 -0.5]]
B = [-0.6 0; 0.6 0.8; -0.8 -0.3]

#Set the number of samples
numScen_budget = 2000 

#Generate random samples
(state0_budget,state_budget) = generate_trajectories(1,A,numScen_budget)

#Initialization
K0 = zeros(dimIn,dim)

K,jsr_bound,ga,P = probabilistc_stability_certificate(state0_budget,state_budget;B=B,numMode=numMode,d=1,batchsize=numScen_budget,K0=K0,beta=0.01,tol=1e-3)

#d: the degree of the Lyapunov function (d=1 means quadratic functions)
#batchsize: the size of the subproblem in distributed computation 
#beta: confidence level
#tol: tolerance for alternating minimization
```

## References
* Wang, Z., Berger, G. O., & Jungers, R. M. (2021). Data-driven feedback stabilization of switched linear systems with probabilistic stability guarantees. arXiv preprint arXiv:2103.10823.

