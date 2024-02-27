module NonlinearSolveBaseRecursiveArrayToolsExt

import NonlinearSolveBase, RecursiveArrayTools
import NonlinearSolveBase: NAN_CHECK, UNITLESS_ABS2

@inline function UNITLESS_ABS2(x::RecursiveArrayTools.AbstractVectorOfArray)
    return mapreduce(UNITLESS_ABS2, NonlinearSolveBase.__abs2_and_sum,
        x.u, init = NonlinearSolveBase.__default_init(x))
end
@inline function UNITLESS_ABS2(x::RecursiveArrayTools.ArrayPartition)
    return mapreduce(UNITLESS_ABS2, NonlinearSolveBase.__abs2_and_sum,
        x.x, init = NonlinearSolveBase.__default_init(x))
end

@inline NAN_CHECK(x::RecursiveArrayTools.AbstractVectorOfArray) = any(NAN_CHECK, x.u)
@inline NAN_CHECK(x::RecursiveArrayTools.ArrayPartition) = any(NAN_CHECK, x.x)


end