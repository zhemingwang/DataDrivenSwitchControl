
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
        @SDconstraint(model, P >= Matrix(I,dim,dim))
        @objective(model, Max, 0)
        for i in 1:numTraj
          @constraint(model, state[:,i]'*P*state[:,i] <= gamma^(2*horizon)*state0[:,i]'*P*state0[:,i])
        end
        @SDconstraint(model, P <= C*Matrix(I,dim,dim))
        JuMP.optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
          gammaU=gamma
        else
          gammaL=gamma
        end
    end
    #=
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
        if value.(s) < 1e-8
          gammaU=gamma
        else
          gammaL=gamma
        end
    end=#
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
        JuMP.optimize!(model) 
        return reshape(JuMP.value.(K),m,n),(sqrt(JuMP.value.(ga)))^(1/d)
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
        JuMP.optimize!(model) 
        return reshape(JuMP.value.(K),m,n),(sqrt(JuMP.value.(ga)))^(1/d)

    end
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
    if d >= 2
        ga,P =  data_driven_lyapb_quad(statelift0,statelift;C=1e2,ub=ga0^d,postprocess=true)
    else
        ga,P =  data_driven_lyapb_quad(statelift0,statelift;C=1e2,ub=ga0^d,postprocess=true)
    end

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




function LQR_alternating(state0,state;B,Q,R,kappaU,lowbound,K0,ep,tol=1e-4)

    (dim,N) = size(state0)
    xitemp = 0
    Ptemp = (Q+K0'*R*K0)*2
    for i in 1:N
        x_i = state[:,i]
        x0_i = state0[:,i]
        xi_i = sqrt((x_i+B*K0*x0_i)'*Ptemp*(x_i+B*K0*x0_i)/(x0_i'*Q*x0_i))
        if xitemp < xi_i
            xitemp = xi_i
        end
    end

    Ptemp,xitemp = LQR_alternating_xi(state0,state;B=B,Q=Q,R=R,kappaU=kappaU,Ktemp=K0,xitemp=xitemp)
    Ktemp = K0
    Ztemp = Ptemp-Q-Ktemp'*R*Ktemp
    eigZtemp = eigvals(Ztemp)
    ratio = 1-(maximum(eigZtemp)/minimum(eigZtemp))*(1-cos(delta2theta(ep,dim)))
    if ratio <0
        ratio = 0
    end
    #println("initializaiton: $Ptemp || $eigZtemp")

    dga = 1e2
    feasible = xitemp/ratio
    while dga > tol && feasible > 1
        println(repeat('*', 80))
        println("Contraction rate: $xitemp")
        println("Minimizing K...")
        K_update,xi_update = LQR_alternating_K(state0,state;B=B,Q=Q,R=R,kappaU=kappaU,lowbound=lowbound,Ptemp=Ptemp,Ktemp=Ktemp,xitemp=xitemp)

        println("Minimizing P...")
        P_update = LQR_alternating_P(state0,state;B=B,Q=Q,R=R,kappaU=kappaU,Ktemp=K_update,xitemp=xi_update)


        dga = xitemp-xi_update
        xitemp = xi_update
        Ptemp = P_update
        Ktemp = K_update

        Ztemp = Ptemp-Q-Ktemp'*R*Ktemp
        eigZtemp = eigvals(Ztemp)
    
        ratio = 1-(maximum(eigZtemp)/minimum(eigZtemp))*(1-cos(delta2theta(ep,dim)))
        if ratio <0
            ratio = 0
        end
        feasible = xitemp/ratio
    end
    println(repeat('*', 80))
    println("Contraction rate: $xitemp")

    return Ktemp, xitemp, Ptemp

end

function LQR_alternating_xi(state0,state;B,Q,R,kappaU,Ktemp,xitemp,tol=1e-4)

    numTraj = size(state0)[2]
    dim = size(state0)[1]
    ub = xitemp
    lb = 1
    if ub < lb
        ub = 1
    end
    stateK = zeros(dim,numTraj)

    for i in 1:numTraj
        stateK[:,i] = state[:,i]+B*Ktemp*state0[:,i]
    end

    while ub-lb > tol
        xi = (ub + lb)/2
        solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
        model = Model(solver)
        @variable(model, P[1:dim, 1:dim] in PSDCone())
        @variable(model, v>=1e-6)
        @objective(model, Min, 0)
        for i in 1:numTraj
          @constraint(model, stateK[:,i]'*P*stateK[:,i] <= xi^2*state0[:,i]'*(P-Q-Ktemp'*R*Ktemp)*state0[:,i])
        end
        @SDconstraint(model, P >= Q+Ktemp'*R*Ktemp)
        @SDconstraint(model, P >= Q+Ktemp'*R*Ktemp+v*Matrix(I,dim,dim))
        @SDconstraint(model, P - Q - Ktemp'*R*Ktemp <=kappaU*v*Matrix(I,dim,dim))

        JuMP.optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
            ub=xi
            #println(value.(v))
            #println(value.(P))
        else
            lb=xi
        end
    end

    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(solver)
    @variable(model, P[1:dim, 1:dim] in PSDCone())
    @variable(model, v>=1e-6)
    @objective(model, Min, tr(P))
    for i in 1:numTraj
      @constraint(model, stateK[:,i]'*P*stateK[:,i] <= ub^2*state0[:,i]'*(P-Q-Ktemp'*R*Ktemp)*state0[:,i])
    end
    @SDconstraint(model, P >= Q+Ktemp'*R*Ktemp)
    @SDconstraint(model, P >= Q+Ktemp'*R*Ktemp+v*Matrix(I,dim,dim))
    @SDconstraint(model, P - Q - Ktemp'*R*Ktemp <=kappaU*v*Matrix(I,dim,dim))
    JuMP.optimize!(model)

    return value.(P), ub
end


function LQR_alternating_K(state0,state;B,Q,R,kappaU,lowbound,Ptemp,Ktemp,xitemp)
    numTraj = size(state0)[2]
    (n,m) = size(B)

    #=ub = xitemp^2
    lb = lowbound^2
    Kopt = zeros(m,n)
    
    while ub-lb > 1e-4
        xi = (ub + lb)/2
        solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
        model = Model(solver)
        @variable(model, K[1:m, 1:n])
        @objective(model, Min, 0)
        for i in 1:numTraj
            @constraint(model, (state[:,i]+B*K*state0[:,i])'*Ptemp*(state[:,i]+B*K*state0[:,i]) <= xi*state0[:,i]'*(Ptemp-Q-K'*R*K)*state0[:,i])
        end
        @SDconstraint(model, [Ptemp-Q K';K inv(R)]>=0)
        JuMP.optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
            ub=xi
            Kopt = value.(K)
        else
            lb=xi
        end
    end
    xiopt = ub=#


    
    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(solver)
    @variable(model, K[1:m, 1:n])
    @variable(model, xitemp^2>=xi>=lowbound^2)
    #@variable(model, v>=0)
    @objective(model, Min, xi+10*(xi-xitemp^2)^2)
    for i in 1:numTraj
        @SDconstraint(model,[state0[:,i]'*(Ptemp-Q)*state0[:,i] (state[:,i]+B*K*state0[:,i])'*Ptemp state0[:,i]'*K';Ptemp*(state[:,i]+B*K*state0[:,i]) xi*Ptemp zeros(n,m);K*state0[:,i] zeros(m,n) inv(R)]>=0)
    end
    @SDconstraint(model, [Ptemp-Q K';K inv(R)]>=0)
    #@SDconstraint(model, [v*Matrix(I,m,m) K';K inv(R)]>=0)
    JuMP.optimize!(model)

    Kopt = value.(K)
    xiopt = value.(xi)

    lambdamin = 0
    for la in 0:0.01:1
        Kla = la*Ktemp+(1-la)*Kopt
        eigZ = eigvals(Ptemp-Q-Kla'*R*Kla)
        if maximum(eigZ)/minimum(eigZ) <= kappaU
            lambdamin = la
            break
        end
    end
    Knew = lambdamin*Ktemp+(1-lambdamin)*Kopt
    xinew = sqrt((1-lambdamin)*xiopt+lambdamin*xitemp^2)

    return Knew, xinew


end

function LQR_alternating_P(state0,state;B,Q,R,kappaU,Ktemp,xitemp)

    numTraj = size(state0)[2]
    (n,m) = size(B)

    stateK = zeros(dim,numTraj)

    for i in 1:numTraj
        stateK[:,i] = state[:,i]+B*Ktemp*state0[:,i]
    end

    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(solver)
    @variable(model, P[1:n, 1:n] in PSDCone())
    @variable(model, v>=1e-6)
    #(model, s>=0)
    @objective(model, Min, tr(P))
    #@SDconstraint(model, P <= s*Matrix(I,n,n))
    @SDconstraint(model, P >= Q+Ktemp'*R*Ktemp)
    @SDconstraint(model, P >= Q+Ktemp'*R*Ktemp+v*Matrix(I,n,n))
    @SDconstraint(model, P - Q -Ktemp'*R*Ktemp <=kappaU*v*Matrix(I,n,n))
    for i in 1:numTraj
        @constraint(model, stateK[:,i]'*P*stateK[:,i] <= xitemp^2*state0[:,i]'*(P-Q-Ktemp'*R*Ktemp)*state0[:,i])
    end
    JuMP.optimize!(model)

    return value.(P)

end