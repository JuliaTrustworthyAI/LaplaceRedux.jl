var documenterSearchIndex = {"docs":
[{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"CurrentModule = LaplaceRedux","category":"page"},{"location":"tutorials/mlp/#Bayesian-MLP","page":"MLP Binary Classifier","title":"Bayesian MLP","text":"","category":"section"},{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"This time we use a synthetic dataset containing samples that are not linearly separable:","category":"page"},{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"# Number of points to generate.\nxs, ys = LaplaceRedux.Data.toy_data_non_linear(200)\nX = hcat(xs...) # bring into tabular format\ndata = zip(xs,ys)","category":"page"},{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"For the classification task we build a neural network with weight decay composed of a single hidden layer.","category":"page"},{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"n_hidden = 10\nD = size(X,1)\nnn = Chain(\n    Dense(D, n_hidden, σ),\n    Dense(n_hidden, 1)\n)  \nloss(x, y) = Flux.Losses.logitbinarycrossentropy(nn(x), y) ","category":"page"},{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"The model is trained until training loss stagnates.","category":"page"},{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"using Flux.Optimise: update!, Adam\nopt = Adam(1e-3)\nepochs = 100\navg_loss(data) = mean(map(d -> loss(d[1],d[2]), data))\nshow_every = epochs/10\n\nfor epoch = 1:epochs\n  for d in data\n    gs = gradient(Flux.params(nn)) do\n      l = loss(d...)\n    end\n    update!(opt, Flux.params(nn), gs)\n  end\n  if epoch % show_every == 0\n    println(\"Epoch \" * string(epoch))\n    @show avg_loss(data)\n  end\nend","category":"page"},{"location":"tutorials/mlp/#Laplace-Approximation","page":"MLP Binary Classifier","title":"Laplace Approximation","text":"","category":"section"},{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"Laplace approximation can be implemented as follows:","category":"page"},{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"la = Laplace(nn; likelihood=:classification, subset_of_weights=:all)\nfit!(la, data)\nla_untuned = deepcopy(la)   # saving for plotting\noptimize_prior!(la; verbose=true, n_steps=500)","category":"page"},{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"The plot below shows the resulting posterior predictive surface for the plugin estimator (left) and the Laplace approximation (right).","category":"page"},{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"# Plot the posterior distribution with a contour plot.\nzoom=0\np_plugin = plot(la, X, ys; title=\"Plugin\", link_approx=:plugin, clim=(0,1))\np_untuned = plot(la_untuned, X, ys; title=\"LA - raw (λ=$(unique(diag(la_untuned.P₀))[1]))\", clim=(0,1), zoom=zoom)\np_laplace = plot(la, X, ys; title=\"LA - tuned (λ=$(round(unique(diag(la.P₀))[1],digits=2)))\", clim=(0,1), zoom=zoom)\nplot(p_plugin, p_untuned, p_laplace, layout=(1,3), size=(1700,400))","category":"page"},{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"(Image: )","category":"page"},{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"Zooming out we can note that the plugin estimator produces high-confidence estimates in regions scarce of any samples. The Laplace approximation is much more conservative about these regions.","category":"page"},{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"zoom=-50\np_plugin = plot(la, X, ys; title=\"Plugin\", link_approx=:plugin, clim=(0,1))\np_untuned = plot(la_untuned, X, ys; title=\"LA - raw (λ=$(unique(diag(la_untuned.P₀))[1]))\", clim=(0,1), zoom=zoom)\np_laplace = plot(la, X, ys; title=\"LA - tuned (λ=$(round(unique(diag(la.P₀))[1],digits=2)))\", clim=(0,1), zoom=zoom)\nplot(p_plugin, p_untuned, p_laplace, layout=(1,3), size=(1700,400))","category":"page"},{"location":"tutorials/mlp/","page":"MLP Binary Classifier","title":"MLP Binary Classifier","text":"(Image: )","category":"page"},{"location":"resources/resources/#Additional-Resources","page":"Additional Resources","title":"Additional Resources","text":"","category":"section"},{"location":"resources/resources/#JuliaCon-2022","page":"Additional Resources","title":"JuliaCon 2022","text":"","category":"section"},{"location":"resources/resources/","page":"Additional Resources","title":"Additional Resources","text":"Slides: link","category":"page"},{"location":"resources/resources/","page":"Additional Resources","title":"Additional Resources","text":"<iframe style=\"width:560px;height:315px\" src=\"https://www.paltmeyer.com/LaplaceRedux.jl/dev/resources/juliacon22/presentation.html\" title=\"YouTube video player\" frameborder=\"0\" allow=\"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture\" allowfullscreen></iframe>","category":"page"},{"location":"tutorials/logit/","page":"Logistic Regression","title":"Logistic Regression","text":"CurrentModule = LaplaceRedux","category":"page"},{"location":"tutorials/logit/#Bayesian-Logistic-Regression","page":"Logistic Regression","title":"Bayesian Logistic Regression","text":"","category":"section"},{"location":"tutorials/logit/","page":"Logistic Regression","title":"Logistic Regression","text":"We will use synthetic data with linearly separable samples:","category":"page"},{"location":"tutorials/logit/","page":"Logistic Regression","title":"Logistic Regression","text":"# Number of points to generate.\nxs, ys = LaplaceRedux.Data.toy_data_linear(100)\nX = hcat(xs...) # bring into tabular format\ndata = zip(xs,ys)","category":"page"},{"location":"tutorials/logit/","page":"Logistic Regression","title":"Logistic Regression","text":"Logistic regression with weight decay can be implemented in Flux.jl as a single dense (linear) layer with binary logit crossentropy loss:","category":"page"},{"location":"tutorials/logit/","page":"Logistic Regression","title":"Logistic Regression","text":"nn = Chain(Dense(2,1))\nλ = 0.5\nsqnorm(x) = sum(abs2, x)\nweight_regularization(λ=λ) = 1/2 * λ^2 * sum(sqnorm, Flux.params(nn))\nloss(x, y) = Flux.Losses.logitbinarycrossentropy(nn(x), y) + weight_regularization()","category":"page"},{"location":"tutorials/logit/","page":"Logistic Regression","title":"Logistic Regression","text":"The code below simply trains the model. After about 50 training epochs training loss stagnates.","category":"page"},{"location":"tutorials/logit/","page":"Logistic Regression","title":"Logistic Regression","text":"using Flux.Optimise: update!, Adam\nopt = Adam()\nepochs = 50\navg_loss(data) = mean(map(d -> loss(d[1],d[2]), data))\nshow_every = epochs/10\n\nfor epoch = 1:epochs\n  for d in data\n    gs = gradient(Flux.params(nn)) do\n      l = loss(d...)\n    end\n    update!(opt, Flux.params(nn), gs)\n  end\n  if epoch % show_every == 0\n    println(\"Epoch \" * string(epoch))\n    @show avg_loss(data)\n  end\nend","category":"page"},{"location":"tutorials/logit/#Laplace-approximation","page":"Logistic Regression","title":"Laplace approximation","text":"","category":"section"},{"location":"tutorials/logit/","page":"Logistic Regression","title":"Logistic Regression","text":"Laplace approximation for the posterior predictive can be implemented as follows:","category":"page"},{"location":"tutorials/logit/","page":"Logistic Regression","title":"Logistic Regression","text":"la = Laplace(nn; likelihood=:classification, λ=λ, subset_of_weights=:last_layer)\nfit!(la, data)\nla_untuned = deepcopy(la)   # saving for plotting\noptimize_prior!(la; verbose=true, n_steps=500)","category":"page"},{"location":"tutorials/logit/","page":"Logistic Regression","title":"Logistic Regression","text":"The plot below shows the resulting posterior predictive surface for the plugin estimator (left) and the Laplace approximation (right).","category":"page"},{"location":"tutorials/prior/","page":"A note on the prior ...","title":"A note on the prior ...","text":"CurrentModule = LaplaceRedux","category":"page"},{"location":"tutorials/prior/","page":"A note on the prior ...","title":"A note on the prior ...","text":"note: In Progress\n","category":"page"},{"location":"tutorials/prior/","page":"A note on the prior ...","title":"A note on the prior ...","text":"    This documentation is still incomplete.","category":"page"},{"location":"tutorials/prior/#A-quick-note-on-the-prior","page":"A note on the prior ...","title":"A quick note on the prior","text":"","category":"section"},{"location":"tutorials/prior/#General-Effect","page":"A note on the prior ...","title":"General Effect","text":"","category":"section"},{"location":"tutorials/prior/","page":"A note on the prior ...","title":"A note on the prior ...","text":"High prior precision → only observation noise. Low prior precision → high posterior uncertainty.","category":"page"},{"location":"tutorials/prior/","page":"A note on the prior ...","title":"A note on the prior ...","text":"using LaplaceRedux.Data\nn = 150       # number of observations\nσtrue = 0.30  # true observational noise\nx, y = Data.toy_data_regression(n;noise=σtrue)\nxs = [[x] for x in x]\nX = permutedims(x)","category":"page"},{"location":"tutorials/prior/","page":"A note on the prior ...","title":"A note on the prior ...","text":"(Image: )","category":"page"},{"location":"tutorials/prior/#Effect-of-Model-Size-on-Optimal-Choice","page":"A note on the prior ...","title":"Effect of Model Size on Optimal Choice","text":"","category":"section"},{"location":"tutorials/prior/","page":"A note on the prior ...","title":"A note on the prior ...","text":"For larger models, the optimal prior precision λ as evaluated through Empirical Bayes tends to be smaller.","category":"page"},{"location":"tutorials/prior/","page":"A note on the prior ...","title":"A note on the prior ...","text":"(Image: )","category":"page"},{"location":"tutorials/prior/","page":"A note on the prior ...","title":"A note on the prior ...","text":"(Image: )","category":"page"},{"location":"tutorials/regression/","page":"MLP Regression","title":"MLP Regression","text":"CurrentModule = LaplaceRedux","category":"page"},{"location":"tutorials/regression/#Data","page":"MLP Regression","title":"Data","text":"","category":"section"},{"location":"tutorials/regression/","page":"MLP Regression","title":"MLP Regression","text":"We first generate some synthetic data:","category":"page"},{"location":"tutorials/regression/","page":"MLP Regression","title":"MLP Regression","text":"using LaplaceRedux.Data\nn = 300       # number of observations\nσtrue = 0.30  # true observational noise\nx, y = Data.toy_data_regression(n;noise=σtrue)\nxs = [[x] for x in x]\nX = permutedims(x)","category":"page"},{"location":"tutorials/regression/#MLP","page":"MLP Regression","title":"MLP","text":"","category":"section"},{"location":"tutorials/regression/","page":"MLP Regression","title":"MLP Regression","text":"We set up a model and loss with weight regularization:","category":"page"},{"location":"tutorials/regression/","page":"MLP Regression","title":"MLP Regression","text":"data = zip(xs,y)\nn_hidden = 50\nD = size(X,1)\nnn = Chain(\n    Dense(D, n_hidden, tanh),\n    Dense(n_hidden, 1)\n)  \nloss(x, y) = Flux.Losses.mse(nn(x), y)","category":"page"},{"location":"tutorials/regression/","page":"MLP Regression","title":"MLP Regression","text":"We train the model:","category":"page"},{"location":"tutorials/regression/","page":"MLP Regression","title":"MLP Regression","text":"using Flux.Optimise: update!, Adam\nopt = Adam(1e-3)\nepochs = 1000\navg_loss(data) = mean(map(d -> loss(d[1],d[2]), data))\nshow_every = epochs/10\n\nfor epoch = 1:epochs\n  for d in data\n    gs = gradient(Flux.params(nn)) do\n      l = loss(d...)\n    end\n    update!(opt, Flux.params(nn), gs)\n  end\n  if epoch % show_every == 0\n    println(\"Epoch \" * string(epoch))\n    @show avg_loss(data)\n  end\nend","category":"page"},{"location":"tutorials/regression/#Laplace-Approximation","page":"MLP Regression","title":"Laplace Approximation","text":"","category":"section"},{"location":"tutorials/regression/","page":"MLP Regression","title":"MLP Regression","text":"Laplace approximation can be implemented as follows:","category":"page"},{"location":"tutorials/regression/","page":"MLP Regression","title":"MLP Regression","text":"subset_w = :all\nla = Laplace(nn; likelihood=:regression, subset_of_weights=subset_w)\nfit!(la, data)\nplot(la, X, y; zoom=-5, size=(400,400))","category":"page"},{"location":"tutorials/regression/","page":"MLP Regression","title":"MLP Regression","text":"(Image: )","category":"page"},{"location":"tutorials/regression/","page":"MLP Regression","title":"MLP Regression","text":"Next we optimize the prior precision P₀ and and observational noise σ using Empirical Bayes:","category":"page"},{"location":"tutorials/regression/","page":"MLP Regression","title":"MLP Regression","text":"optimize_prior!(la; verbose=true)\nplot(la, X, y; zoom=-5, size=(400,400))","category":"page"},{"location":"tutorials/regression/","page":"MLP Regression","title":"MLP Regression","text":"loss(exp.(logP₀), exp.(logσ)) = 117.66664854620934\nLog likelihood: -85.92178189333164\nLog det ratio: 35.71426712713756\nScatter: 27.77546617861784\nloss(exp.(logP₀), exp.(logσ)) = 121.87231511858798\nLog likelihood: -94.6377872523882\nLog det ratio: 40.7731858243348\nScatter: 13.695869908064768\nloss(exp.(logP₀), exp.(logσ)) = 105.74544738246172\nLog likelihood: -79.26976568466756\nLog det ratio: 43.950740004518195\nScatter: 9.000623391070134\nloss(exp.(logP₀), exp.(logσ)) = 103.88142914896216\nLog likelihood: -77.44149990175734\nLog det ratio: 45.50530524149423\nScatter: 7.374553252915389\nloss(exp.(logP₀), exp.(logσ)) = 104.09479293926637\nLog likelihood: -77.63140243238486\nLog det ratio: 45.99843510214248\nScatter: 6.928345911620529\n\nloss(exp.(logP₀), exp.(logσ)) = 103.3468498181464\nLog likelihood: -76.88770063543016\nLog det ratio: 45.92861066847996\nScatter: 6.989687696952491\nloss(exp.(logP₀), exp.(logσ)) = 103.16246637027459\nLog likelihood: -76.71757535070529\nLog det ratio: 45.64189954216795\nScatter: 7.247882496970637\nloss(exp.(logP₀), exp.(logσ)) = 103.16978958495622\nLog likelihood: -76.7341717145966\nLog det ratio: 45.34308830278303\nScatter: 7.528147437936199\nloss(exp.(logP₀), exp.(logσ)) = 103.16445388078068\nLog likelihood: -76.73183970804324\nLog det ratio: 45.1273375561218\nScatter: 7.737890789353098\nloss(exp.(logP₀), exp.(logσ)) = 103.15526534959244\nLog likelihood: -76.72294852715255\nLog det ratio: 45.01394675305761\nScatter: 7.850686891822177","category":"page"},{"location":"tutorials/regression/","page":"MLP Regression","title":"MLP Regression","text":"(Image: )","category":"page"},{"location":"tutorials/multi/","page":"MLP Multi-Label Classifier","title":"MLP Multi-Label Classifier","text":"CurrentModule = LaplaceRedux","category":"page"},{"location":"tutorials/multi/#Multi-class-problem","page":"MLP Multi-Label Classifier","title":"Multi-class problem","text":"","category":"section"},{"location":"tutorials/multi/","page":"MLP Multi-Label Classifier","title":"MLP Multi-Label Classifier","text":"using LaplaceRedux.Data\nx, y = Data.toy_data_multi()\nX = hcat(x...)\ny_train = Flux.onehotbatch(y, unique(y))\ny_train = Flux.unstack(y_train',1)","category":"page"},{"location":"tutorials/multi/","page":"MLP Multi-Label Classifier","title":"MLP Multi-Label Classifier","text":"data = zip(x,y_train)\nn_hidden = 3\nD = size(X,1)\nout_dim = length(unique(y))\nnn = Chain(\n    Dense(D, n_hidden, σ),\n    Dense(n_hidden, out_dim)\n)  \nloss(x, y) = Flux.Losses.logitcrossentropy(nn(x), y)","category":"page"},{"location":"tutorials/multi/","page":"MLP Multi-Label Classifier","title":"MLP Multi-Label Classifier","text":"using Flux.Optimise: update!, Adam\nopt = Adam()\nepochs = 100\navg_loss(data) = mean(map(d -> loss(d[1],d[2]), data))\nshow_every = epochs/10\n\nfor epoch = 1:epochs\n    for d in data\n        gs = gradient(Flux.params(nn)) do\n            l = loss(d...)\n        end\n        update!(opt, Flux.params(nn), gs)\n    end\n    if epoch % show_every == 0\n        println(\"Epoch \" * string(epoch))\n        @show avg_loss(data)\n    end\nend","category":"page"},{"location":"tutorials/multi/#Laplace-Approximation","page":"MLP Multi-Label Classifier","title":"Laplace Approximation","text":"","category":"section"},{"location":"tutorials/multi/","page":"MLP Multi-Label Classifier","title":"MLP Multi-Label Classifier","text":"la = Laplace(nn; likelihood=:classification)\nfit!(la, data)\noptimize_prior!(la; verbose=true, n_steps=1000)","category":"page"},{"location":"tutorials/multi/","page":"MLP Multi-Label Classifier","title":"MLP Multi-Label Classifier","text":"_labels = sort(unique(y))\nplt_list = []\nfor target in _labels\n    plt = plot(la, X, y; target=target, clim=(0,1))\n    push!(plt_list, plt)\nend\nplot(plt_list...)","category":"page"},{"location":"tutorials/multi/","page":"MLP Multi-Label Classifier","title":"MLP Multi-Label Classifier","text":"(Image: )","category":"page"},{"location":"tutorials/multi/","page":"MLP Multi-Label Classifier","title":"MLP Multi-Label Classifier","text":"_labels = sort(unique(y))\nplt_list = []\nfor target in _labels\n    plt = plot(la, X, y; target=target, clim=(0,1), link_approx=:plugin)\n    push!(plt_list, plt)\nend\nplot(plt_list...)","category":"page"},{"location":"tutorials/multi/","page":"MLP Multi-Label Classifier","title":"MLP Multi-Label Classifier","text":"(Image: )","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"CurrentModule = LaplaceRedux","category":"page"},{"location":"reference/#All-functions-and-types","page":"Reference","title":"All functions and types","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"","category":"page"},{"location":"reference/#Exported-functions","page":"Reference","title":"Exported functions","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [\n    LaplaceRedux,\n    LaplaceRedux.Curvature\n]\nPrivate = false","category":"page"},{"location":"reference/#LaplaceRedux.Laplace-Tuple{AbstractArray}","page":"Reference","title":"LaplaceRedux.Laplace","text":"(la::Laplace)(X::AbstractArray; kwrgs...)\n\nCalling a model with Laplace Approximation on an array of inputs is equivalent to explicitly calling the predict function.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.Laplace-Tuple{Any}","page":"Reference","title":"LaplaceRedux.Laplace","text":"Laplace(model::Any; loss_fun::Union{Symbol, Function}, kwargs...)\n\nWrapper function to prepare Laplace approximation.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.fit!-Tuple{Laplace, Any}","page":"Reference","title":"LaplaceRedux.fit!","text":"fit!(la::Laplace,data)\n\nFits the Laplace approximation for a data set.\n\nExamples\n\nusing Flux, LaplaceRedux\nx, y = LaplaceRedux.Data.toy_data_linear()\ndata = zip(x,y)\nnn = Chain(Dense(2,1))\nla = Laplace(nn)\nfit!(la, data)\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.optimize_prior!-Tuple{Laplace}","page":"Reference","title":"LaplaceRedux.optimize_prior!","text":"optimize_prior!(\n    la::Laplace; \n    n_steps::Int=100, lr::Real=1e-1,\n    λinit::Union{Nothing,Real}=nothing,\n    σinit::Union{Nothing,Real}=nothing\n)\n\nOptimize the prior precision post-hoc through Empirical Bayes (marginal log-likelihood maximization).\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.predict-Tuple{Laplace, AbstractArray}","page":"Reference","title":"LaplaceRedux.predict","text":"predict(la::Laplace, X::AbstractArray; link_approx=:probit)\n\nComputes predictions from Bayesian neural network.\n\nExamples\n\nusing Flux, LaplaceRedux\nx, y = toy_data_linear()\ndata = zip(x,y)\nnn = Chain(Dense(2,1))\nla = Laplace(nn)\nfit!(la, data)\npredict(la, hcat(x...))\n\n\n\n\n\n","category":"method"},{"location":"reference/#Internal-functions","page":"Reference","title":"Internal functions","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [\n    LaplaceRedux,\n    LaplaceRedux.Curvature\n]\nPublic = false","category":"page"},{"location":"reference/#LaplaceRedux.BaseLaplace","page":"Reference","title":"LaplaceRedux.BaseLaplace","text":"Abstract base type of Laplace Approximation.\n\n\n\n\n\n","category":"type"},{"location":"reference/#LaplaceRedux._H_factor-Tuple{LaplaceRedux.BaseLaplace}","page":"Reference","title":"LaplaceRedux._H_factor","text":"_H_factor(la::BaseLaplace)\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux._init_H-Tuple{LaplaceRedux.BaseLaplace}","page":"Reference","title":"LaplaceRedux._init_H","text":"_init_H(la::BaseLaplace)\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux._weight_penalty-Tuple{LaplaceRedux.BaseLaplace}","page":"Reference","title":"LaplaceRedux._weight_penalty","text":"_weight_penalty(la::BaseLaplace)\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.functional_variance-Tuple{Laplace, Any}","page":"Reference","title":"LaplaceRedux.functional_variance","text":"functional_variance(la::Laplace,𝐉)\n\nCompute the linearized GLM predictive variance as 𝐉ₙΣ𝐉ₙ' where 𝐉=∇f(x;θ)|θ̂ is the Jacobian evaluated at the MAP estimate and Σ = P⁻¹.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.get_loss_fun-Tuple{Symbol, Flux.Chain}","page":"Reference","title":"LaplaceRedux.get_loss_fun","text":"get_loss_fun(likelihood::Symbol)\n\nHelper function to choose loss function based on specified model likelihood.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.get_params-Tuple{LaplaceRedux.BaseLaplace}","page":"Reference","title":"LaplaceRedux.get_params","text":"get_params(la::BaseLaplace)\n\nRetrieves the desired (sub)set of model parameters and stores them in a list.\n\nExamples\n\nusing Flux, LaplaceRedux\nnn = Chain(Dense(2,1))\nla = Laplace(nn)\nLaplaceRedux.get_params(la)\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.glm_predictive_distribution-Tuple{Laplace, AbstractArray}","page":"Reference","title":"LaplaceRedux.glm_predictive_distribution","text":"glm_predictive_distribution(la::Laplace, X::AbstractArray)\n\nComputes the linearized GLM predictive.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.hessian_approximation-Tuple{Laplace, Any}","page":"Reference","title":"LaplaceRedux.hessian_approximation","text":"hessian_approximation(la::Laplace, d)\n\nComputes the local Hessian approximation at a single data d.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.log_det_posterior_precision-Tuple{LaplaceRedux.BaseLaplace}","page":"Reference","title":"LaplaceRedux.log_det_posterior_precision","text":"log_det_posterior_precision(la::Laplace)\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.log_det_prior_precision-Tuple{LaplaceRedux.BaseLaplace}","page":"Reference","title":"LaplaceRedux.log_det_prior_precision","text":"log_det_prior_precision(la::Laplace)\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.log_det_ratio-Tuple{LaplaceRedux.BaseLaplace}","page":"Reference","title":"LaplaceRedux.log_det_ratio","text":"log_det_ratio(la::BaseLaplace)\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.log_likelihood-Tuple{LaplaceRedux.BaseLaplace}","page":"Reference","title":"LaplaceRedux.log_likelihood","text":"log_likelihood(la::BaseLaplace)\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.log_marginal_likelihood-Tuple{LaplaceRedux.BaseLaplace}","page":"Reference","title":"LaplaceRedux.log_marginal_likelihood","text":"log_marginal_likelihood(la::BaseLaplace; P₀::Union{Nothing,UniformScaling}=nothing, σ::Union{Nothing, Real}=nothing)\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.outdim-Tuple{Flux.Chain}","page":"Reference","title":"LaplaceRedux.outdim","text":"outdim(model::Chain)\n\nHelper function to determine the output dimension of a Flux.Chain\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.outdim-Tuple{LaplaceRedux.BaseLaplace}","page":"Reference","title":"LaplaceRedux.outdim","text":"outdim(la::BaseLaplace)\n\nHelper function to determine the output dimension of a Flux.Chain with Laplace approximation.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.posterior_covariance","page":"Reference","title":"LaplaceRedux.posterior_covariance","text":"posterior_covariance(la::BaseLaplace, P=la.P)\n\nComputes the posterior covariance as the inverse of the posterior precision: Sigma=P^-1.\n\n\n\n\n\n","category":"function"},{"location":"reference/#LaplaceRedux.posterior_precision","page":"Reference","title":"LaplaceRedux.posterior_precision","text":"posterior_precision(la::BaseLaplace)\n\nComputes the posterior precision P for a fitted Laplace Approximation as follows,\n\nP = sum_n=1^Nnabla_theta^2log p(mathcalD_ntheta)_theta_MAP + nabla_theta^2 log p(theta)_theta_MAP\n\nwhere sum_n=1^Nnabla_theta^2log p(mathcalD_ntheta)_theta_MAP=H and nabla_theta^2 log p(theta)_theta_MAP=P_0.\n\n\n\n\n\n","category":"function"},{"location":"reference/#LaplaceRedux.Curvature.CurvatureInterface","page":"Reference","title":"LaplaceRedux.Curvature.CurvatureInterface","text":"Basetype for any curvature interface.\n\n\n\n\n\n","category":"type"},{"location":"reference/#LaplaceRedux.Curvature.EmpiricalFisher","page":"Reference","title":"LaplaceRedux.Curvature.EmpiricalFisher","text":"Constructor for Empirical Fisher.\n\n\n\n\n\n","category":"type"},{"location":"reference/#LaplaceRedux.Curvature.full-Tuple{LaplaceRedux.Curvature.EmpiricalFisher, Tuple}","page":"Reference","title":"LaplaceRedux.Curvature.full","text":"full(curvature::EmpiricalFisher, d::Union{Tuple,NamedTuple})\n\nCompute the full empirical Fisher.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.Curvature.gradients-Tuple{LaplaceRedux.Curvature.CurvatureInterface, AbstractArray, Union{Number, AbstractArray}}","page":"Reference","title":"LaplaceRedux.Curvature.gradients","text":"gradients(curvature::CurvatureInterface, X::AbstractArray, y::Number)\n\nCompute the gradients with respect to the loss function: ∇ℓ(f(x;θ),y) where f: ℝᴰ ↦ ℝᴷ.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LaplaceRedux.Curvature.jacobians-Tuple{LaplaceRedux.Curvature.CurvatureInterface, AbstractArray}","page":"Reference","title":"LaplaceRedux.Curvature.jacobians","text":"jacobians(curvature::CurvatureInterface, X::AbstractArray)\n\nComputes the Jacobian ∇f(x;θ) where f: ℝᴰ ↦ ℝᴷ.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = LaplaceRedux","category":"page"},{"location":"#LaplaceRedux","page":"Home","title":"LaplaceRedux","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for LaplaceRedux.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"LaplaceRedux.jl is a library written in pure Julia that can be used for effortless Bayesian Deep Learning trough Laplace Approximation (LA). In the development of this package I have drawn inspiration from this Python library and its companion paper (Daxberger et al. 2021).","category":"page"},{"location":"#Installation","page":"Home","title":"🚩 Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The stable version of this package can be installed as follows:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(\"LaplaceRedux.jl\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"The development version can be installed like so:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(\"https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl\")","category":"page"},{"location":"#Basic-Usage","page":"Home","title":"🖥️ Basic Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"LaplaceRedux.jl can be used for any neural network trained in Flux.jl. Below we show basic usage examples involving two simple models for a regression and a classification task, respectively.","category":"page"},{"location":"#Regression","page":"Home","title":"Regression","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A complete worked example for a regression model can be found in the docs. Here we jump straight to Laplace Approximation and take the pre-trained model nn as given. Then LA can be implemented as follows, where we specify the model likelihood. The plot show the fitted values overlaid with a 95% confidence interval. As expected, predictive uncertainty quickly increases in areas that are not populated by any training data.","category":"page"},{"location":"","page":"Home","title":"Home","text":"la = Laplace(nn; likelihood=:regression)\nfit!(la, data)\noptimize_prior!(la)\nplot(la, X, y; zoom=-5, size=(500,500))","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"#Binary-Classification","page":"Home","title":"Binary Classification","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Once again we jump straight to LA and refer to the docs for a complete worked example involving binary classification. In this case we need to specify likelihood=:classification. The plot below shows the resulting posterior predictive distributions as contours in the two-dimensional feature space: note how the Plugin Approximation on the left compares to the Laplace Approximation on the right.","category":"page"},{"location":"","page":"Home","title":"Home","text":"theme(:lime)\n\nla = Laplace(nn; likelihood=:classification)\nfit!(la, data)\nla_untuned = deepcopy(la)   # saving for plotting\noptimize_prior!(la; n_steps=500)\n\n# Plot the posterior predictive distribution:\nzoom=0\np_plugin = plot(la, X, ys; title=\"Plugin\", link_approx=:plugin, clim=(0,1))\np_untuned = plot(la_untuned, X, ys; title=\"LA - raw (λ=$(unique(diag(la_untuned.P₀))[1]))\", clim=(0,1), zoom=zoom)\np_laplace = plot(la, X, ys; title=\"LA - tuned (λ=$(round(unique(diag(la.P₀))[1],digits=2)))\", clim=(0,1), zoom=zoom)\nplot(p_plugin, p_untuned, p_laplace, layout=(1,3), size=(1700,400))","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"#JuliaCon-2022","page":"Home","title":"📢 JuliaCon 2022","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This project was presented at JuliaCon 2022 in July 2022. See here for details.","category":"page"},{"location":"#Contribute","page":"Home","title":"🛠️ Contribute","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Contributions are very much welcome! Please follow the SciML ColPrac guide. You may want to start by having a look at any open issues.","category":"page"},{"location":"#References","page":"Home","title":"🎓 References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Daxberger, Erik, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, and Philipp Hennig. 2021. “Laplace Redux-Effortless Bayesian Deep Learning.” Advances in Neural Information Processing Systems 34.","category":"page"}]
}
