using LaplaceRedux
using Documenter

DocMeta.setdocmeta!(LaplaceRedux, :DocTestSetup, :(using LaplaceRedux); recursive=true)

makedocs(;
    modules=[LaplaceRedux],
    authors="Patrick Altmeyer",
    repo="https://github.com/pat-alt/LaplaceRedux.jl/blob/{commit}{path}#{line}",
    sitename="LaplaceRedux.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pat-alt.github.io/LaplaceRedux.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Logistic Regression" => "tutorials/logit.md",
            "MLP" => "tutorials/mlp.md",
            "A note on the prior ..." => "tutorials/prior.md",
        ],
        "Reference" => "reference.md",
        "Additional Resources" => "resources/resources.md"
    ],
)

deploydocs(;
    repo="github.com/pat-alt/LaplaceRedux.jl",
    devbranch="main",
)
