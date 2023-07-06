using LaplaceRedux
using Documenter

DocMeta.setdocmeta!(LaplaceRedux, :DocTestSetup, :(using LaplaceRedux); recursive=true)

makedocs(;
    modules=[LaplaceRedux],
    authors="Patrick Altmeyer",
    repo="https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl/blob/{commit}{path}#{line}",
    sitename="LaplaceRedux.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://juliatrustworthyai.github.io/LaplaceRedux.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Logistic Regression" => "tutorials/logit.md",
            "MLP Binary Classifier" => "tutorials/mlp.md",
            "MLP Multi-Label Classifier" => "tutorials/multi.md",
            "MLP Regression" => "tutorials/regression.md",
            "A note on the prior ..." => "tutorials/prior.md",
        ],
        "Reference" => "_reference.md",
        "Additional Resources" => "resources/_resources.md",
    ],
)

deploydocs(; repo="github.com/JuliaTrustworthyAI/LaplaceRedux.jl", devbranch="main")
