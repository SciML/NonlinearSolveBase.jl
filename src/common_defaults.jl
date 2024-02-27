@inline __default_init(x) = zero(real(__value(eltype(x))))

@inline UNITLESS_ABS2(x::Number) = abs2(x)
@inline function UNITLESS_ABS2(x::AbstractArray)
    return mapreduce(UNITLESS_ABS2, __abs2_and_sum, x, init = __default_init(x))
end

@inline NAN_CHECK(x::Number) = isnan(x)
@inline NAN_CHECK(x::Float64) = isnan(x) || (x > 1e50)
@inline NAN_CHECK(x::Enum) = false
@inline NAN_CHECK(x::AbstractArray) = any(NAN_CHECK, x)

# Default Norm is the L₂-norm
@inline NONLINEARSOLVE_DEFAULT_NORM(u::Union{AbstractFloat, Complex}) = @fastmath abs(u)

@inline function NONLINEARSOLVE_DEFAULT_NORM(u::AbstractArray{T}) where {T}
    if T <: Union{AbstractFloat, Complex}
        u isa StaticArraysCore.StaticArray &&
            return Base.FastMath.sqrt_fast(real(sum(abs2, u)))
        if __fast_scalar_indexing(u)
            x = zero(T)
            @inbounds @fastmath for ui in u
                x += abs2(ui)
            end
            return Base.FastMath.sqrt_fast(real(x))
        end
    end
    return Base.FastMath.sqrt_fast(UNITLESS_ABS2(u))
end

@inline NONLINEARSOLVE_DEFAULT_NORM(u) = norm(u, 2)
