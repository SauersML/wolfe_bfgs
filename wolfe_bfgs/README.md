# wolfe_bfgs

[![Crates.io](https://img.shields.io/crates/v/wolfe_bfgs.svg)](https://crates.io/crates/wolfe_bfgs)
[![Docs.rs](https://docs.rs/wolfe_bfgs/badge.svg)](https://docs.rs/wolfe_bfgs)
[![Build Status](https://github.com/SauersML/opt/actions/workflows/test.yml/badge.svg)](https://github.com/SauersML/opt/actions)

Focused dense BFGS optimization in Rust with a Strong Wolfe line search, reexported from `opt`.

This crate is the smaller companion package for users who only want the BFGS-first API.

This crate exposes the first-order API only:
- `Bfgs`
- `Problem`
- `optimize`
- `FirstOrderObjective`
- `ZerothOrderObjective`
- `FiniteDiffGradient`
- `Solution`
- `Bounds`, `Tolerance`, `MaxIterations`, `Profile`
- `BfgsError` and related error/configuration types

If you want the full solver set, including Newton trust-region and ARC, use the companion `opt` crate instead.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
wolfe_bfgs = "0.4.0"
```

## Example

```rust
use wolfe_bfgs::{
    optimize, FirstOrderObjective, FirstOrderSample, MaxIterations, Problem, Profile, Solution,
    Tolerance, ZerothOrderObjective,
};
use ndarray::{array, Array1};

struct Rosenbrock;

impl ZerothOrderObjective for Rosenbrock {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, wolfe_bfgs::ObjectiveEvalError> {
        let a = 1.0;
        let b = 100.0;
        Ok((a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2))
    }
}

impl FirstOrderObjective for Rosenbrock {
    fn eval_grad(
        &mut self,
        x: &Array1<f64>,
    ) -> Result<FirstOrderSample, wolfe_bfgs::ObjectiveEvalError> {
        let a = 1.0;
        let b = 100.0;
        let f = (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2);
        Ok(FirstOrderSample {
            value: f,
            gradient: array![
                -2.0 * (a - x[0]) - 4.0 * b * (x[1] - x[0].powi(2)) * x[0],
                2.0 * b * (x[1] - x[0].powi(2)),
            ],
        })
    }
}

let x0 = array![-1.2, 1.0];

let Solution {
    final_point: x_min,
    final_value,
    final_gradient_norm: Some(grad_norm),
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
assert!(grad_norm < 1e-5);
```

For cost-only objectives, wrap a `ZerothOrderObjective` with `FiniteDiffGradient`.

## Implementation

`wolfe_bfgs` is published from the same repository as `opt`.
It reexports the first-order API from `opt` rather than maintaining a separate implementation.
