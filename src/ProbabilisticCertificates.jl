function theta2delta(th,dim)
    return beta_inc((dim-1)/2,1/2,sin(th)^2)[1]
end
function delta2theta(delt,dim)
    sintheta = sqrt(beta_inc_inv((dim-1)/2,1/2,delt)[1])
    return asin(sintheta)
end

function epsilon2beta(ep,dim,numMode,N)
    return numMode*(1-theta2delta(delta2theta(ep,dim)/2,dim)/numMode)^N/theta2delta(delta2theta(ep,dim)/4,dim)
end

function looseningratio_quad(ep,dim,kappaP)
    loosenfactor = (1-kappaP*(1-cos(delta2theta(ep,dim))))
    if loosenfactor <= 0
        println("Insufficient samples!")
        return Inf
    else
        return 1/(1-kappaP*(1-cos(delta2theta(ep,dim))))
    end
end

function beta2epsilon(beta,dim,numMode,N)
    ep_u = 1.0
    ep_l = 0.0
    while ep_u - ep_l > 1e-10
        ep = (ep_u+ep_l)/2
        beta_ep = epsilon2beta(ep,dim,numMode,N)
        if beta_ep > beta
            ep_l = ep
        else
            ep_u = ep
        end
    end
    return ep_u
end

function probabilistc_stability_certificate(state0,state;B,numMode,d,batchsize,K0,beta=0.01,tol=1e-4)
    dim = size(B,1)
    N = size(state0,2)
    K,ga,P = soslyap_alternating(state0,state;B=B,d=d,batchsize=batchsize,K0=K0,tol=tol)
    eigP = eigvals(P)
    kappaP = maximum(eigP)/minimum(eigP)
    if d == 1
        ep = beta2epsilon(beta,dim,numMode,N)
        return K,ga*looseningratio_quad(ep,dim,kappaP),ga,P
    else
        epsos = beta2epsilon(beta,dim,numMode,N)
        stateK = zeros(dim,N)
        for i in 1:N
            stateK[:,i] = state[:,i]+B*K*state0[:,i]
        end
        boundnorm = norm_max(state0,stateK)/cos(delta2theta(epsos,dim))

        phid = 0.0
        for i in 1:d
            phid += (2-2*cos(delta2theta(epsos,dim)))^(i/2)*binomial(d,i)
        end
        bound = (ga^d*(1+sqrt(kappaP)*phid)+boundnorm^d*sqrt(kappaP)*phid)^(1/d)

        #=for betasos in 0.1*beta:0.1*beta:0.9*beta
            epsos = beta2epsilon(betasos,dim,numMode,N)
        
            epnorm = minimum([numMode*(1-(beta-betasos)^(1/N)),1])

            stateK = zeros(dim,N)
            for i in 1:N
                stateK[:,i] = state[:,i]+B*K*state0[:,i]
            end

            boundnorm = norm_max(state0,stateK)/cos(delta2theta(epnorm,dim))
            
            phid = 0.0
            for i in 1:d
                phid += (2-2*cos(delta2theta(epsos,dim)))^(i/2)*binomial(d,i)
            end
            boundnormclosed = boundnorm #+ opnorm(B*K)
            bound = (ga^d*(1+sqrt(kappaP)*phid)+boundnormclosed^d*sqrt(kappaP)*phid)^(1/d)
            push!(boundsearch,bound)
        end=#
        #println("Probability bisection: $boundsearch")
        return K,bound,ga,P
        
    end

end

function norm_max(state0,state)
    traj_norm = 0.0
    numTraj = size(state0)[2]
    
    for i = 1:numTraj
        traj_norm_i = norm(state[:,i])/norm(state0[:,i])
        if traj_norm_i > traj_norm
            traj_norm = traj_norm_i
        end
    end
    
    return traj_norm
end
    
function probabilistc_LQR_certificate(state0,state;B,numMode,Q,R,kappaU,K0,beta=0.01,tol=1e-4)
    dim = size(B,1)
    N = size(state0,2)

    ep = beta2epsilon(beta,dim,numMode,N)

    lowbound = 1-kappaU*(1-cos(delta2theta(ep,dim)))
    #println("lowbound: $lowbound")
    if lowbound < 0
        lowbound = 0
    end

    K, xi, P = LQR_alternating(state0,state;B=B,Q=Q,R=R,kappaU=kappaU,lowbound=lowbound,K0=K0,ep=ep,tol=tol)

    Z = P-Q-K'*R*K
    eigZ = eigvals(Z)

    ratio = 1-(maximum(eigZ)/minimum(eigZ))*(1-cos(delta2theta(ep,dim)))
    #println("ratio: $ratio")
    #println("eigZ: $eigZ")

    if ratio < xi
        println("Infeasible!")
    else
        println("A feasible solution is found!")
    end
    return K,xi,P,ratio


end


    