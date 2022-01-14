function generate_switched_linear_systems(numMode,dim)
    A = [randn(dim,dim) for i=1:numMode]
    for i = 1:numMode
        A[i] = A[i]/maximum(abs.(eigvals(A[i])))#opnorm(A[i])
    end
    return A
end


function generate_trajectories(horizon,A,numTraj)

numMode = size(A)[1]
dim = size(A[1])[1]

x0 = zeros(dim,numTraj)
x = zeros(dim,numTraj)
for i=1:numTraj
    y0 = randn(dim)
    x0[:,i] = y0/norm(y0)
    x[:,i] = x0[:,i]
    b = rand((1:numMode),horizon)
    for j = 1:horizon
        x[:,i] = A[b[j]]*x[:,i]
    end
end
return x0, x
end
