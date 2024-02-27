module NonlinearSolveBase

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    import ArrayInterface, RecursiveArrayTools, SciMLBase, StaticArraysCore
    import ConcreteStructs: @concrete
    import FastClosures: @closure
    import LinearAlgebra: norm
    import Markdown: @doc_str
end

include("common_defaults.jl")
include("termination_conditions.jl")
include("utils.jl")

end
