
function veroneselift(x,liftscale,d::Integer)
    X = collect((prod(y) for y in with_replacement_combinations(x, d)))
    X = X.*liftscale

    return X
end

function veroneseliftscale(n::Integer,d::Integer)
    power = collect(multiexponents(n, d))
    df = factorial(d)
    scaling(m) = sqrt(df / prod(factorial, m))
    liftscale = collect(scaling(poly) for poly in power)
    return liftscale
end
