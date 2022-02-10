function white_box_jsr(A,d=2)
    s = discreteswitchedsystem(A)
    optimizer_constructor = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    soslyapb(s, d, optimizer_constructor=optimizer_constructor, tol=1e-4, verbose=0)
    seq = sosbuildsequence(s, d, p_0=:Primal)
    psw = findsmp(seq)
    return psw.growthrate
end
#=
function white_box_stabilization_quad(A,B,tol=1e-4)
    normAmax = 0
    dim,dimIn = size(B)
    for Ai in A
        normAi = opnorm(Ai)
        if normAi>normAmax
            normAmax = normAi
        end
    end
    ub = normAmax
    lb = 0
    K = zeros(dimIn,dim)
    P = Matrix(I,dim,dim)
    while ub-lb > tol
        gamma = (ub + lb)/2
        solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
        model = Model(solver)
        @variable(model, Q[1:dim, 1:dim] in PSDCone())
        @variable(model, Y[1:dimIn,1:dim])
        @variable(model, s>=0)
        @SDconstraint(model, Q >= Matrix(I,dim,dim))
        @objective(model, Min, s)
        for Ai in A
            @SDconstraint(model, [gamma^2*Q+s*Matrix(I,dim,dim) Q*Ai'+(B*Y)';Ai*Q+B*Y Q+s*Matrix(I,dim,dim)] >= 0)
        end
        JuMP.optimize!(model)
        if value.(s) < 1e-10
            ub = gamma
            P = inv(value.(Q))
            K = value.(Y)*P
        else
            lb = gamma
        end
    end
    return ub, K, P
end
=#

function white_box_stabilization_quad(A,B,tol=1e-4)
    normAmax = 0
    dim,dimIn = size(B)
    for Ai in A
        normAi = opnorm(Ai)
        if normAi>normAmax
            normAmax = normAi
        end
    end
    ub = normAmax
    lb = 0
    K = zeros(dimIn,dim)
    P = Matrix(I,dim,dim)
    while ub-lb > tol
        gamma = (ub + lb)/2
        solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
        model = Model(solver)
        @variable(model, Q[1:dim, 1:dim] in PSDCone())
        @variable(model, Y[1:dimIn,1:dim])
        @SDconstraint(model, Q >= Matrix(I,dim,dim))
        @objective(model, Max, 0)
        for Ai in A
            @SDconstraint(model, [gamma^2*Q Q*Ai'+(B*Y)';Ai*Q+B*Y Q] >= 0)
        end
        JuMP.optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
            ub = gamma
            P = inv(value.(Q))
            K = value.(Y)*P
        else
            lb = gamma
        end
    end
    return ub, K, P
end


function white_box_LQR(A,B,Q,R)
    (dim,dimIn) = size(B)
    lower_triangular(P) = [P[i, j] for i = 1:size(P, 1) for j = 1:i]
    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(solver)
    @variable(model, S[1:dim, 1:dim] in PSDCone())
    @variable(model, Y[1:dimIn,1:dim])
    @variable(model, t)
    @SDconstraint(model, S >= 0)
    @constraint(model, [t; 1; lower_triangular(S)] in MOI.LogDetConeTriangle(dim))
    @objective(model, Max, t)
    for Ai in A
        @SDconstraint(model, [S S*Ai'+(B*Y)' S Y';Ai*S+B*Y S zeros(dim,dim) zeros(dim,dimIn);S zeros(dim,dim) inv(Q) zeros(dim,dimIn);Y zeros(dimIn,dim) zeros(dimIn,dim) inv(R)] >= 0)
    end
    JuMP.optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        P = inv(value.(S))
        K = value.(Y)*P
        return K, P
    else
        println("The LQR problem is infeasible!")
        return zeros(dimIn,dim), zeros(dim,dim)
    end
end