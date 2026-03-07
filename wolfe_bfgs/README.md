# wolfe_bfgs

[![Crates.io](https://img.shields.io/crates/v/wolfe_bfgs.svg)](https://crates.io/crates/wolfe_bfgs)
[![Docs.rs](https://docs.rs/wolfe_bfgs/badge.svg)](https://docs.rs/wolfe_bfgs)
[![Build Status](https://github.com/SauersML/wolfe_bfgs/actions/workflows/test.yml/badge.svg)](https://github.com/SauersML/wolfe_bfgs/actions)

Focused dense BFGS optimization in Rust with a Strong Wolfe line search.

This crate exposes the first-order API only:
- `Bfgs`
- `Problem`
- `optimize`
- `FirstOrderObjective`
- `BfgsSolution`
- `Bounds`, `Tolerance`, `MaxIterations`, `Profile`
- `BfgsError` and related error/configuration types

If you want the full solver set, including Newton trust-region and ARC, use the companion `opt` crate instead.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
wolfe_bfgs = "0.3.0"
```

## Example

```rust
use wolfe_bfgs::{
    optimize, BfgsSolution, FirstOrderObjective, MaxIterations, Problem, Profile, Tolerance,
};
use ndarray::{array, Array1};

struct Rosenbrock;

impl FirstOrderObjective for Rosenbrock {
    fn eval(
        &mut self,
        x: &Array1<f64>,
        grad_out: &mut Array1<f64>,
    ) -> Result<f64, wolfe_bfgs::ObjectiveEvalError> {
        let a = 1.0;
        let b = 100.0;
        let f = (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2);
        grad_out[0] = -2.0 * (a - x[0]) - 4.0 * b * (x[1] - x[0].powi(2)) * x[0];
        grad_out[1] = 2.0 * b * (x[1] - x[0].powi(2));
        Ok(f)
    }
}

let x0 = array![-1.2, 1.0];

let BfgsSolution {
    final_point: x_min,
    final_value,
    ..
} = optimize(Problem::new(x0, Rosenbrock))
    .with_tolerance(Tolerance::new(1e-6).unwrap())
    .with_max_iterations(MaxIterations::new(100).unwrap())
    .with_profile(Profile::Robust)
    .run()
    .expect("BFGS failed");

assert!((x_min[0] - 1.0).abs() < 1e-5);
assert!((x_min[1] - 1.0).abs() < 1e-5);
assert!(final_value.is_finite());
```

## Relationship to `opt`

`wolfe_bfgs` is the narrow package for users who only want the BFGS solver surface.
The implementation lives in the same workspace as `opt`, which exposes the full optimization toolkit.
