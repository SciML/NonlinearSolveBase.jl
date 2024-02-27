module NonlinearSolveBaseForwardDiffExt

import ForwardDiff, NonlinearSolveBase
import NonlinearSolveBase: __value

__value(x::Type{ForwardDiff.Dual{T, V, N}}) where {T, V, N} = V
__value(x::ForwardDiff.Dual) = __value(ForwardDiff.value(x))

end