module NonlinearSolveBaseSparseArraysExt

import SparseArrays: AbstractSparseMatrixCSC, nonzeros
import NonlinearSolveBase: NAN_CHECK

@inline NAN_CHECK(x::AbstractSparseMatrixCSC) = any(NAN_CHECK, nonzeros(x))

end