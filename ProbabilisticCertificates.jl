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

function probabilistc_stability_certificate(state0,state,B,numMode,d,batchsize,K0,beta=0.01,tol=1e-4)
    dim = size(B,1)
    N = size(state0,2)
    K,ga,P = soslyap_alternating(state0,state,B,d,batchsize,K0,tol)
    eigP = eigvals(P)
    kappaP = maximum(eigP)/minimum(eigP)
    if d == 1
        ep = beta2epsilon(beta,dim,numMode,N)
        return K,ga*looseningratio_quad(ep,dim,kappaP)
    else
        boundsearch = []
        for betasos in 0.1*beta:0.1*beta:0.9*beta
            epsos = beta2epsilon(betasos,dim,numMode,N)
        
            epnorm = minimum([numMode*(1-(beta-betasos)^(1/N)),1])
            boundnorm = norm_max(state0,state)/cos(delta2theta(epnorm,dim))
            
            phid = 0.0
            for i in 1:d
                phid += (2-2*cos(delta2theta(epsos,dim)))^(i/2)*binomial(d,i)
            end
            boundnormclosed = boundnorm + opnorm(B*K)
            bound = (ga^d*(1+sqrt(kappaP)*phid)+boundnormclosed^d*sqrt(kappaP)*phid)^(1/d)
            push!(boundsearch,bound)
        end
        return K,minimum(boundsearch)
        
    end

end

