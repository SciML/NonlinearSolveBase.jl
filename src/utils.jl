# Attempt to use a non-allocating version of isapprox
function __is_approx(x::Number, y::Number; atol = false,
        rtol = atol > 0 ? false : sqrt(eps(promote_type(typeof(x), typeof(y)))))
    isapprox(x, y; atol, rtol)
end
function __is_approx(x, y; atol = false,
        rtol = atol > 0 ? false : sqrt(eps(promote_type(eltype(x), eltype(y)))))
    length(x) != length(y) && return false
    d = __maximum(-, x, y)
    return d ≤ max(atol, rtol * max(maximum(abs, x), maximum(abs, y)))
end

@inline __fast_scalar_indexing(args...) = all(ArrayInterface.fast_scalar_indexing, args)

@inline function __maximum(op::F, x, y) where {F}
    if __fast_scalar_indexing(x, y)
        return maximum(@closure((xᵢyᵢ)->begin
            xᵢ, yᵢ = xᵢyᵢ
            return abs(op(xᵢ, yᵢ))
        end), zip(x, y))
    else
        return mapreduce(@closure((xᵢ, yᵢ)->@.(abs(op(xᵢ, yᵢ)))), max, x, y)
    end
end

@inline function __abs2_and_sum(x, y)
    return reduce(Base.add_sum, x, init = zero(real(value(eltype(x))))) +
           reduce(Base.add_sum, y, init = zero(real(value(eltype(y)))))
end

@inline __value(::Type{T}) where {T} = T
@inline __value(x) = x
