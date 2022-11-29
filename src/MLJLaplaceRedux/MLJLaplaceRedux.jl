module MLJLaplaceRedux

using LaplaceRedux
using LaplaceRedux: BaseLaplace
using MLJ
using MLJFlux
import MLJModelInterface as MMI
import MLJModelInterface: predict, fit, save, restore

function has_mlj_support(la::BaseLaplace)
    return typeof(la.model) <: MLJFlux.MLJFluxModel
end

function MMI.fit(la::BaseLaplace, verbosity, X, y)

    @assert has_mlj_support(la) "Atomic model needs to be of type `MLJFlux.MLJFluxModel`."

    # Train atomic model:
    fitresult, cache, report = MMI.fit(la.model, verbosity, X, y)

    # Fit LA:
    data = move.(collate(model, X, y))
    LaplaceRedux.fit!(la, data)
    optimize_prior!(la)

    return (fitresult, cache, report)

end

function MMI.predict(la::BaseLaplace, fitresult, Xnew)

    # Overwrite chain:
    la.model = fitresult[1]

    # Compute predictions:
    ŷ = LaplaceRedux.predict(la, Xnew)

    return ŷ

end