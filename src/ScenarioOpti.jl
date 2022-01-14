
function data_driven_lyapb_quad(state0,state;horizon=1,C=1e3,ub=1e2,lb=0,tol=1e-4,numIter=1e2,postprocess=false)
    numTraj = size(state0)[2]
    dim = size(state0)[1]
    iter = 1
    gammaU = ub
    gammaL = lb
    while gammaU-gammaL > tol && iter < numIter
        iter += 1
        gamma = (gammaU + gammaL)/2
        solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
        model = Model(solver)
        @variable(model, P[1:dim, 1:dim] in PSDCone())
        @variable(model, s>=0)
        @SDconstraint(model, P >= Matrix(I,dim,dim))
        @objective(model, Min, s)
        for i in 1:numTraj
          @constraint(model, state[:,i]'*P*state[:,i] <= gamma^(2*horizon)*state0[:,i]'*P*state0[:,i]+s)
        end
        @SDconstraint(model, P <= C*Matrix(I,dim,dim))
        JuMP.optimize!(model)
        if value.(s) < 1e-10
          gammaU=gamma
        else
          gammaL=gamma
        end
    end
    gamma = gammaU
    
    if postprocess == true
        solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
        model = Model(solver)
        @variable(model, P[1:dim, 1:dim] in PSDCone())
        @SDconstraint(model, P >= Matrix(I,dim,dim))
        # @variable(model,t >= 0)
        # @constraint(model,t >= norm(P, 2))
        @objective(model, Min, sum(P[:].^2))
        for i in 1:numTraj
          @constraint(model, state[:,i]'*P*state[:,i] <= gamma^(2*horizon)*state0[:,i]'*P*state0[:,i])
        end
        @SDconstraint(model, P <= C*Matrix(I,dim,dim))
        JuMP.optimize!(model)
        return gamma, value.(P)
    else
        return gamma
    end
    
    end
    

function soslyap_alternating(state0,state;B,d,batchsize,K0,tol=1e-4)
    (n,m) = size(B)
    nlift = binomial(n+d-1, d)

    #transform = kronecker2veronese(n,d)
    Xscale = veroneseliftscale(n,d)

    P0 = Matrix(1.0I, nlift, nlift)
    ga0 = convergerate(state0,state,B,d,P0,K0,Xscale)
    gatemp = copy(ga0)
    Ktemp = copy(K0)
    Ptemp = copy(P0)

    dga = 1e2

    Xscale = veroneseliftscale(n,d)

    while dga > tol
        println(repeat('*', 80))
        println("Order: $d || Convergence rate: $gatemp")
        println("Minimizing P...")
        #P,ga = soslyap_alter_P(state0,state,B,d,Ktemp,gatemp)
        P_update,ga_update = soslyap_alternating_P_distributed(state0,state,batchsize,B,d,Ptemp,Ktemp,gatemp,Xscale)
        cholP = cholesky(P_update)
        println("Minimizing K...")
        K_updated,ga = soslyap_alternating_K_distributed(state0,state,batchsize,B,d,P_update,cholP,Ktemp,ga_update,Xscale)

        dga = gatemp-ga
        gatemp = copy(ga)
        Ktemp = copy(K_updated)
        Ptemp = copy(P_update)
    end
    println(repeat('*', 80))
    println("Order: $d || Convergence rate (terminated): $gatemp")

    return Ktemp, gatemp, Ptemp

end


function soslyap_alternating_K(state0,state,B,d,P,cholP,K0,ga0,Xscale)

    (n,m) = size(B)
    nlift = binomial(n+d-1, d)
    N = size(state0,2)

    V0 = zeros(N)
    for i in 1:N
        xlift0_i = veroneselift(state0[:,i],Xscale,d)
        V0[i] = transpose(xlift0_i)*P*xlift0_i
    end

    if d == 1
        solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
        model = Model(solver)
        @variable(model, K[1:m, 1:n])
        @variable(model, ga >=0)
        @objective(model, Min, ga)
        for i in 1:N
            @SDconstraint(model, [ga*V0[i] (state[:,i]+B*K*state0[:,i])'*P;P*(state[:,i]+B*K*state0[:,i]) P] >=0)
        end
    else
        solver = optimizer_with_attributes(Ipopt.Optimizer, MOI.Silent() => true)
        #solver = optimizer_with_attributes(NLopt.Optimizer)
        model = Model(solver)
        #set_optimizer_attribute(model, "algorithm", :GN_ISRES) #:GN_ISRES GN_AGS GN_ORIG_DIRECT
        #set_optimizer_attribute(model, "local_optimizer", :LD_MMA) #LD_SLSQP LD_MMA LN_COBYLA
        @variable(model, -1e1 <= K[i=1:n*m] <= 1e1)
        set_start_value.(K,reshape(K0,m*n,1))
        @variable(model, 0 <= ga <= ga0^(2*d),start = ga0^(2*d))
        @objective(model, Min, ga)
        
        lyapconstraint = []
        for i in 1:N
            push!(lyapconstraint,(y,x...) -> sum(sum(cholP.U[j,k]*veroneselift(state[:,i]+B*reshape(collect(x),m,n)*state0[:,i],Xscale,d)[k] for k in 1:nlift)^2 for j in 1:nlift)-y*V0[i])
        end
        for (i, f) in enumerate(lyapconstraint)
            f_sym = Symbol("f_$(i)")
            register(model, f_sym, n*m+1, f; autodiff = true)
            add_NL_constraint(model, :($(f_sym)($(ga),$(K...)) <= 0))
        end
    end
    JuMP.optimize!(model) 
    return reshape(JuMP.value.(K),m,n),(sqrt(JuMP.value.(ga)))^(1/d)

end

function soslyap_alternating_K_distributed(state0,state,batchsize,B,d,P,cholP,K0,ga0,Xscale)

    N = size(state0,2)
    batchnum = ceil(Int,N/batchsize)
    partition = []
    for i in 1:batchnum
        if i<batchnum
            push!(partition,(i-1)*batchsize+1:i*batchsize)
        else
            push!(partition,(batchnum-1)*batchsize+1:N)
        end
    end

    gammaopt = copy(ga0)
    Kopt = copy(K0)

    println("K: Total number of batches: $batchnum")
    for i in 1:batchnum
        println("K: Solving batch $i...")
        K_i,gamma_i = soslyap_alternating_K(state0[:,partition[i]],state[:,partition[i]],B,d,P,cholP,K0,ga0,Xscale)
        Kopt_i,gammaopt_i = backtrackinglinesearch_K(state0,state,B,d,P,K_i,K0,Xscale)
        if gammaopt_i < gammaopt
            Kopt = copy(Kopt_i)
            gammaopt = copy(gammaopt_i)
        end
    end
    return Kopt,gammaopt
end



function backtrackinglinesearch_K(state0,state,B,d,P,Ktemp,K0,Xscale,tol=1e-2)

    gammaopt = 1e6
    Kopt = copy(K0)
    for lambda in 0:tol:1
        Klambda = (1-lambda)*K0+lambda*Ktemp
        galambda = convergerate(state0,state,B,d,P,Klambda,Xscale)
        #println(galambda)
        if galambda <= gammaopt
            gammaopt = copy(galambda)
            Kopt = copy(Klambda)
        end
    end

    return Kopt,gammaopt
end

function soslyap_alternating_P(state0,state,B,d,K,ga0,Xscale)

    n = size(B,1)
    nlift = binomial(n+d-1, d)
    N = size(state0,2)

    statelift0 = zeros(nlift,N)
    statelift = zeros(nlift,N)
    for i in 1:N
        statelift0[:,i] = veroneselift(state0[:,i],Xscale,d)
        statelift[:,i] = veroneselift(state[:,i]+B*K*state0[:,i],Xscale,d)
    end

    ga,P =  data_driven_lyapb_quad(statelift0,statelift;C=1e3,ub=ga0^d,postprocess=true)

    return P,ga^(1/d)
end

function soslyap_alternating_P_distributed(state0,state,batchsize,B,d,P0,K0,ga0,Xscale)
    N = size(state0,2)
    batchnum = ceil(Int,N/batchsize)
    partition = []
    for i in 1:batchnum
        if i<batchnum
            push!(partition,(i-1)*batchsize+1:i*batchsize)
        else
            push!(partition,(batchnum-1)*batchsize+1:N)
        end
    end

    gammaopt = copy(ga0)
    Popt = copy(P0)

    println("P: Total number of batches: $batchnum")
    for i in 1:batchnum
        println("P: Solving batch $i...")
        P_i, = soslyap_alternating_P(state0[:,partition[i]],state[:,partition[i]],B,d,K0,ga0,Xscale)
        Popt_i,gammaopt_i = backtrackinglinesearch_P(state0,state,B,d,P_i,P0,K0,Xscale)
        if gammaopt_i < gammaopt
            Popt = copy(Popt_i)
            gammaopt = copy(gammaopt_i)
        end
    end
    return Popt,gammaopt
end

function backtrackinglinesearch_P(state0,state,B,d,Ptemp,P0,K0,Xscale,tol=1e-2)

    gammaopt = 1e6
    Popt = copy(P0)

    for lambda in 0:tol:1
        Plambda = (1-lambda)*P0+lambda*Ptemp
        galambda = convergerate(state0,state,B,d,Plambda,K0,Xscale)
        if galambda <= gammaopt
            gammaopt = copy(galambda)
            Popt = copy(Plambda)
        end
    end

    return Popt,gammaopt
end

function convergerate(state0,state,B,d,P,K,Xscale)
    N = size(state0,2)
    gamma = 0.0
    for i in 1:N
        xlift_i = veroneselift(state[:,i]+B*K*state0[:,i],Xscale,d)
        xlift0_i = veroneselift(state0[:,i],Xscale,d)
        gamma_i = sqrt(transpose(xlift_i)*P*xlift_i/(transpose(xlift0_i)*P*xlift0_i))
        if gamma_i > gamma
            gamma = copy(gamma_i)
        end
    end
    return gamma^(1/d)
end
