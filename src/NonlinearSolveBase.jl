module NonlinearSolveBase

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    import ArrayInterface, SciMLBase, StaticArraysCore
    import ConcreteStructs: @concrete
    import FastClosures: @closure
    import LinearAlgebra: norm
    import Markdown: @doc_str
    import SciMLBase: ReturnCode
end

include("common_defaults.jl")
include("termination_conditions.jl")
include("recursivearraytools.jl")
include("utils.jl")

export NormTerminationMode, AbsTerminationMode, RelTerminationMode, AbsNormTerminationMode,
       RelNormTerminationMode, AbsSafeTerminationMode, RelSafeTerminationMode,
       AbsSafeBestTerminationMode, RelSafeBestTerminationMode

end
