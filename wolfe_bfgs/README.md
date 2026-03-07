# wolfe_bfgs

`wolfe_bfgs` is the focused BFGS crate from this repository.

It exposes only the first-order BFGS API:

- `Bfgs`
- `Problem`
- `optimize`
- `FirstOrderObjective`
- `BfgsSolution`
- related configuration and error types

Add it with:

```toml
[dependencies]
wolfe_bfgs = "0.3.0"
```
