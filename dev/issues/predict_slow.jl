using LaplaceRedux
using MLJBase
using Optimisers

X = MLJBase.table(rand(Float32, 100, 3));
y = coerce(rand("abc", 100), Multiclass);
model = LaplaceClassification(; optimiser=Optimisers.Adam(0.1), epochs=100);
fitresult, _, _ = MLJBase.fit(model, 2, X, y);
la = fitresult[1];
Xmat = matrix(X) |> permutedims;

# Single test sample:
Xtest = Xmat[:, 1:10];
Xtest_tab = MLJBase.table(Xtest');
MLJBase.predict(model, fitresult, Xtest_tab);       # warm up
LaplaceRedux.predict(la, Xmat);                     # warm up
@time MLJBase.predict(model, fitresult, Xtest_tab);
@time LaplaceRedux.predict(la, Xtest);
@time glm_predictive_distribution(la, Xtest);
