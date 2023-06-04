# Issue report
## Refactoring Gradients and Jacobians to multidimensional arrays

Cases:
- Regression:
  - 1-to-1
  - n-to-1
  - [does not exist] n-to-m? (conceptually equivalent to n-to-1 repeated for m output vars)
- Classification:
  - n-to-1 (binary, with sigmoid, this is logistic regression)
  - n-to-m (multi-label, with softmax)

NOTE: pay attention to (edge) cases with n=1

Idea:
Use a batching DataLoader the same way it is used in FluxML training, since this provides a nice user-friendly interface to those already familiar w/ Flux.
Otherwise the user would have to reshape the dataset by themselves.

Internally, `jacobians` and `gradients` always interpret one dimension (either last or first, TBD) as the batch dimension.

To be decided: last or first.

<https://saransh-cpp.github.io/fluxml.github.io/tutorialposts/2021-01-21-data-loader/>
<https://fluxml.ai/Flux.jl/v0.11/data/dataloader/>
<https://fluxml.ai/Flux.jl/stable/data/mlutils/#MLUtils.batch>

Made the decision to pass `data=(X, Y)`, a tuple of (non-batched) input and output instead of a zip,
and to add an argument of `batchSize` (default value of 1).
These tuple is then passed on to

Advantages: this does not require the user the zip the data, or to know the data has to be zipped.
We have also considered typing the argument with the Flux MLUtils DataLoader, but then the users would have to make an additional import.

We may have `data` as a dual union type (Tuple or DataLoader).

TODO: possible additional issue: improve the test suite. For instance
- Edge cases in the constructor and 
- Add documentation for the analytical derivation

TODO: ask about lambda and sigma in the constructor: are they not the reciprocals of each other?

Alternative decision: instead of expecting a tuple argument, leave the argument to be an iterable of input-output pairs, where the input-output *may* be batched.
This makes batching a responsibiliy of the user (who should be instructed to use MLUtils.Dataloader for this purpose).
Since the user is already using Flux, they do not have to make an additional import for it, as Flux re-exports from MLUtils.

---

Testcases TODO:
- Complete workflow testcases: Batching should give the same results as non-batched computation.
- Regression tests

By default, the aggregate function in loss is mean, not sum. This conflicts the above goal of batch-nonbatch result equivalence.
