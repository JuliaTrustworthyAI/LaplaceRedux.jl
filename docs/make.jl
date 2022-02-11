using BayesLaplace
using Documenter

DocMeta.setdocmeta!(BayesLaplace, :DocTestSetup, :(using BayesLaplace); recursive=true)

makedocs(;
    modules=[BayesLaplace],
    authors="Patrick Altmeyer",
    repo="https://github.com/pat-alt/BayesLaplace.jl/blob/{commit}{path}#{line}",
    sitename="BayesLaplace.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pat-alt.github.io/BayesLaplace.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "quick_start.md"
    ],
)

deploydocs(;
    repo="github.com/pat-alt/BayesLaplace.jl",
    devbranch="main",
)
