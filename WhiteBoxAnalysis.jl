function white_box_jsr(A,d=2)
    s = discreteswitchedsystem(A)
    optimizer_constructor = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    soslyapb(s, d, optimizer_constructor=optimizer_constructor, tol=1e-4, verbose=1)
    seq = sosbuildsequence(s, d, p_0=:Primal)
    psw = findsmp(seq)
    return psw.growthrate
end

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
