abstract type AbstractNonlinearTerminationMode end
abstract type AbstractSafeNonlinearTerminationMode <: AbstractNonlinearTerminationMode end
abstract type AbstractSafeBestNonlinearTerminationMode <:
              AbstractSafeNonlinearTerminationMode end

@doc doc"""
    NormTerminationMode <: AbstractNonlinearTerminationMode

Terminates if
``\| \frac{\partial u}{\partial t} \| \leq reltol \times \| \frac{\partial u}{\partial t} + u \|``
or ``\| \frac{\partial u}{\partial t} \| \leq abstol``
"""
struct NormTerminationMode <: AbstractNonlinearTerminationMode end

@doc doc"""
    RelTerminationMode <: AbstractNonlinearTerminationMode

Terminates if
``all \left(| \frac{\partial u}{\partial t} | \leq reltol \times | u | \right)``.
"""
struct RelTerminationMode <: AbstractNonlinearTerminationMode end

@doc doc"""
    RelNormTerminationMode <: AbstractNonlinearTerminationMode

Terminates if
``\| \frac{\partial u}{\partial t} \| \leq reltol \times \| \frac{\partial u}{\partial t} + u \|``
"""
struct RelNormTerminationMode <: AbstractNonlinearTerminationMode end

@doc doc"""
    AbsTerminationMode <: AbstractNonlinearTerminationMode

Terminates if ``all \left( | \frac{\partial u}{\partial t} | \leq abstol \right)``.
"""
struct AbsTerminationMode <: AbstractNonlinearTerminationMode end

@doc doc"""
    AbsNormTerminationMode <: AbstractNonlinearTerminationMode

Terminates if ``\| \frac{\partial u}{\partial t} \| \leq abstol``.
"""
struct AbsNormTerminationMode <: AbstractNonlinearTerminationMode end

for norm_type in (:Rel, :Abs), safety in (:Safe, :SafeBest)
    struct_name = Symbol("$(norm_type)$(safety)TerminationMode")
    supertype_name = Symbol("Abstract$(safety)NonlinearTerminationMode")

    doctring = safety == :Safe ?
               "Essentially [`$(norm_type)NormTerminationMode`](@ref) + terminate if there \
                has been no improvement for the last `patience_steps` + terminate if the \
                solution blows up (diverges)." :
               "Essentially [`$(norm_type)SafeTerminationMode`](@ref), but caches the best\
                solution found so far."

    @eval begin
        """
            $($struct_name) <: $($supertype_name)

        $($doctring)

        ## Constructor

            $($struct_name)(; protective_threshold = nothing, patience_steps = 100,
                patience_objective_multiplier = 3, min_max_factor = 1.3, max_stalled_steps = nothing)
        """
        @kwdef @concrete struct $(struct_name){T <: Union{Nothing, Int}} <:
                                $(supertype_name)
            protective_threshold = nothing
            patience_steps::Int = 100
            patience_objective_multiplier = 3
            min_max_factor = 1.3
            max_stalled_steps::T = nothing
        end
    end
end

# Core Implementation
@concrete mutable struct NonlinearTerminationModeCache{uType, T}
    u::uType
    retcode::ReturnCode.T
    abstol::T
    reltol::T
    best_objective_value::T
    mode
    initial_objective
    objectives_trace
    nsteps::Int
    saved_values
    u0_norm
    step_norm_trace
    max_stalled_steps
    u_diff_cache::uType
end

@inline get_termination_mode(cache::NonlinearTerminationModeCache) = cache.mode
@inline get_abstol(cache::NonlinearTerminationModeCache) = cache.abstol
@inline get_reltol(cache::NonlinearTerminationModeCache) = cache.reltol
@inline get_saved_values(cache::NonlinearTerminationModeCache) = cache.saved_values

function __update_u!!(cache::NonlinearTerminationModeCache, u)
    cache.u === nothing && return
    if cache.u isa AbstractArray && ArrayInterface.can_setindex(cache.u)
        copyto!(cache.u, u)
    else
        cache.u = u
    end
end

__cvt_real(::Type{T}, ::Nothing) where {T} = nothing
__cvt_real(::Type{T}, x) where {T} = real(T(x))

get_tolerance(η, ::Type{T}) where {T} = __cvt_real(T, η)
function get_tolerance(::Nothing, ::Type{T}) where {T}
    η = real(oneunit(T)) * (eps(real(one(T))))^(4 // 5)
    return get_tolerance(η, T)
end
## Rational numbers don't work in GPU kernels
get_tolerance(x, tol, ::Type{T}) where {T} = get_tolerance(tol, T)
function get_tolerance(
        x::Union{StaticArraysCore.StaticArray, Number}, tol::Nothing, ::Type{T}) where {T}
    return T(real(oneunit(T)) * (eps(real(one(T))))^(real(T)(0.8)))
end

function SciMLBase.init(
        du, u, mode::AbstractNonlinearTerminationMode, saved_value_prototype...;
        abstol = nothing, reltol = nothing, kwargs...)
    T = promote_type(eltype(du), eltype(u))
    abstol = get_tolerance(abstol, T)
    reltol = get_tolerance(reltol, T)
    TT = typeof(abstol)
    u_ = mode isa AbstractSafeBestNonlinearTerminationMode ?
         (ArrayInterface.can_setindex(u) ? copy(u) : u) : nothing
    if mode isa AbstractSafeNonlinearTerminationMode
        if mode isa AbsSafeTerminationMode || mode isa AbsSafeBestTerminationMode
            initial_objective = maximum(abs, du)
            u0_norm = nothing
        else
            initial_objective = maximum(abs, du) / (__maximum(+, du, u) + eps(TT))
            u0_norm = mode.max_stalled_steps === nothing ? nothing :
                      NONLINEARSOLVE_DEFAULT_NORM(u)
        end
        objectives_trace = Vector{TT}(undef, mode.patience_steps)
        step_norm_trace = mode.max_stalled_steps === nothing ? nothing :
                          Vector{TT}(undef, mode.max_stalled_steps)
        best_value = initial_objective
        max_stalled_steps = mode.max_stalled_steps
        if ArrayInterface.can_setindex(u_) &&
           !(u_ isa Number) &&
           step_norm_trace !== nothing
            u_diff_cache = similar(u_)
        else
            u_diff_cache = u_
        end
    else
        initial_objective = nothing
        objectives_trace = nothing
        u0_norm = nothing
        step_norm_trace = nothing
        best_value = __cvt_real(T, Inf)
        max_stalled_steps = nothing
        u_diff_cache = u_
    end

    length(saved_value_prototype) == 0 && (saved_value_prototype = nothing)

    return NonlinearTerminationModeCache(
        u_, ReturnCode.Default, abstol, reltol, best_value, mode,
        initial_objective, objectives_trace, 0, saved_value_prototype,
        u0_norm, step_norm_trace, max_stalled_steps, u_diff_cache)
end

function SciMLBase.reinit!(
        cache::NonlinearTerminationModeCache, du, u, saved_value_prototype...;
        abstol = nothing, reltol = nothing, kwargs...)
    T = eltype(get_abstol(cache))
    length(saved_value_prototype) != 0 && (cache.saved_values = saved_value_prototype)

    mode = get_termination_mode(cache)
    u_ = mode isa AbstractSafeBestNonlinearTerminationMode ?
         (ArrayInterface.can_setindex(u) ? copy(u) : u) : nothing
    cache.u = u_
    cache.retcode = ReturnCode.Default

    cache.abstol = get_tolerance(abstol, T)
    cache.reltol = get_tolerance(reltol, T)
    cache.nsteps = 0
    TT = typeof(cache.abstol)

    if mode isa AbstractSafeNonlinearTerminationMode
        if mode isa AbsSafeTerminationMode || mode isa AbsSafeBestTerminationMode
            cache.initial_objective = maximum(abs, du)
        else
            cache.initial_objective = maximum(abs, du) / (maximum(abs, du .+ u) + eps(TT))
            cache.max_stalled_steps !== nothing &&
                (cache.u0_norm = NONLINEARSOLVE_DEFAULT_NORM(u_))
        end
        cache.best_objective_value = initial_objective
    else
        cache.best_objective_value = __cvt_real(T, Inf)
    end
    return
end

# This dispatch is needed based on how Terminating Callback works!
function (cache::NonlinearTerminationModeCache)(
        integrator::SciMLBase.AbstractODEIntegrator, abstol::Number, reltol::Number, min_t)
    cache.abstol = abstol
    cache.reltol = reltol
    if min_t === nothing || integrator.t ≥ min_t
        return cache(get_termination_mode(cache), SciMLBase.get_du(integrator),
            integrator.u, integrator.uprev)
    end
    return false
end
function (cache::NonlinearTerminationModeCache)(du, u, uprev, args...)
    return cache(get_termination_mode(cache), du, u, uprev, args...)
end

function (cache::NonlinearTerminationModeCache)(
        mode::AbstractNonlinearTerminationMode, du, u, uprev, args...)
    if check_convergence(mode, du, u, uprev, cache.abstol, cache.reltol)
        cache.retcode = ReturnCode.Success
        return true
    end
    return false
end

function (cache::NonlinearTerminationModeCache)(
        mode::AbstractSafeNonlinearTerminationMode, du, u, uprev, args...)
    if mode isa AbsSafeTerminationMode || mode isa AbsSafeBestTerminationMode
        objective = maximum(abs, du)
        criteria = get_abstol(cache)
    else
        objective = maximum(abs, du) / (__maximum(+, du, u) + eps(eltype(du)))
        criteria = get_reltol(cache)
    end

    # Protective Break
    if !isfinite(objective)
        cache.retcode = ReturnCode.Unstable
        return true
    end
    ## By default we turn this off since it has the potential for false positives
    if mode.protective_threshold !== nothing &&
       (objective > cache.initial_objective * mode.protective_threshold * length(du))
        cache.retcode = ReturnCode.Unstable
        return true
    end

    # Check if best solution
    if mode isa AbstractSafeBestNonlinearTerminationMode &&
       objective < cache.best_objective_value
        cache.best_objective_value = objective
        __update_u!!(cache, u)
        cache.saved_values !== nothing && length(args) ≥ 1 && (cache.saved_values = args)
    end

    # Main Termination Condition
    if objective ≤ criteria
        cache.retcode = ReturnCode.Success
        return true
    end

    # Terminate if there has been no improvement for the last `patience_steps`
    cache.nsteps += 1
    cache.nsteps == 1 && (cache.initial_objective = objective)
    cache.objectives_trace[mod1(cache.nsteps, length(cache.objectives_trace))] = objective

    if objective ≤ mode.patience_objective_multiplier * criteria &&
       cache.nsteps ≥ mode.patience_steps
        if cache.nsteps < length(cache.objectives_trace)
            min_obj, max_obj = extrema(@view(cache.objectives_trace[1:(cache.nsteps)]))
        else
            min_obj, max_obj = extrema(cache.objectives_trace)
        end
        if min_obj < mode.min_max_factor * max_obj
            cache.retcode = ReturnCode.Stalled
            return true
        end
    end

    # Test for stalling if that is not disabled
    if cache.step_norm_trace !== nothing
        if ArrayInterface.can_setindex(cache.u_diff_cache) && !(u isa Number)
            @. cache.u_diff_cache = u - uprev
        else
            cache.u_diff_cache = u .- uprev
        end
        du_norm = NONLINEARSOLVE_DEFAULT_NORM(cache.u_diff_cache)
        cache.step_norm_trace[mod1(cache.nsteps, length(cache.step_norm_trace))] = du_norm
        if cache.nsteps ≥ mode.max_stalled_steps
            max_step_norm = maximum(cache.step_norm_trace)
            if mode isa AbsSafeTerminationMode || mode isa AbsSafeBestTerminationMode
                stalled_step = max_step_norm ≤ cache.abstol
            else
                stalled_step = max_step_norm ≤
                               cache.reltol * (max_step_norm + cache.u0_norm)
            end
            if stalled_step
                cache.retcode = ReturnCode.Stalled
                return true
            end
        end
    end

    return false
end

# Check Convergence
## All norms here are ∞-norms
function check_convergence(::NormTerminationMode, duₙ, uₙ, uₙ₋₁, abstol, reltol)
    du_norm = maximum(abs, duₙ)
    return (du_norm ≤ abstol) || (du_norm ≤ reltol * __maximum(+, duₙ, uₙ))
end
function check_convergence(::RelTerminationMode, duₙ, uₙ, uₙ₋₁, abstol, reltol)
    if __fast_scalar_indexing(duₙ, uₙ)
        return all(@closure(xy->begin
            x, y = xy
            return abs(x) ≤ reltol * abs(y)
        end), zip(duₙ, uₙ))
    else
        return all(@. abs(duₙ) ≤ reltol * abs(uₙ + duₙ))
    end
end
function check_convergence(::AbsTerminationMode, duₙ, uₙ, uₙ₋₁, abstol, reltol)
    return all(@closure(x->abs(x) ≤ abstol), duₙ)
end
function check_convergence(
        ::Union{RelNormTerminationMode, RelSafeTerminationMode, RelSafeBestTerminationMode},
        duₙ, uₙ, uₙ₋₁, abstol, reltol)
    return maximum(abs, duₙ) ≤ reltol * __maximum(+, duₙ, uₙ)
end
function check_convergence(
        ::Union{AbsNormTerminationMode, AbsSafeTerminationMode, AbsSafeBestTerminationMode},
        duₙ, uₙ, uₙ₋₁, abstol, reltol)
    return maximum(abs, duₙ) ≤ abstol
end
