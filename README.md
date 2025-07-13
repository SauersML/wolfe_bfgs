# A Robust BFGS Implementation in Rust

[![Crates.io](https://img.shields.io/crates/v/wolfe-bfgs.svg)](https://crates.io/crates/wolfe-bfgs)
[![Docs.rs](https://docs.rs/wolfe_bfgs/badge.svg)](https://docs.rs/wolfe_bfgs)
[![Build Status](https://github.com/SauersML/bfgs/actions/workflows/test.yml/badge.svg)](https://github.com/SauersML/bfgs/actions)

A pure Rust implementation of the dense BFGS optimization algorithm for unconstrained nonlinear problems. This library provides a powerful solver built upon the principles and best practices outlined in Nocedal & Wright's *Numerical Optimization*.

This work is a rewrite of the original `bfgs` crate by Paul Kernfeld.

## Features

*   **Strong Wolfe Line Search**: Guarantees stability and positive-definiteness of the Hessian approximation by satisfying the Strong Wolfe conditions. This ensures the crucial curvature condition `s_k^T y_k > 0` holds, making the algorithm reliable even for challenging functions.
*   **Initial Hessian Scaling**: Implements the well-regarded scaling heuristic (Eq. 6.20 from Nocedal & Wright) to produce a well-scaled initial Hessian, often improving the rate of convergence.
*   **Ergonomic API**: Uses the builder pattern for clear and flexible configuration of the solver.
*   **Robust Error Handling**: Provides descriptive errors for common failure modes, such as line search failures or reaching the iteration limit, enabling better diagnostics.
*   **Dense BFGS Implementation**: Stores the full `n x n` inverse Hessian approximation, suitable for small- to medium-scale optimization problems.

## Usage

First, add this to your `Cargo.toml`:

```toml
[dependencies]
wolfe_bfgs = "0.1.0"
```

### Example: Minimizing the Rosenbrock Function

Here is an example of minimizing the 2D Rosenbrock function, a classic benchmark for optimization algorithms.

```rust
use wolfe_bfgs::{Bfgs, BfgsSolution};
use ndarray::{array, Array1};

// 1. Define the objective function and its gradient.
// The function must return a tuple: (value, gradient).
let rosenbrock = |x: &Array1<f64>| -> (f64, Array1<f64>) {
    let a = 1.0;
    let b = 100.0;
    let f = (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2);
    let g = array![
        -2.0 * (a - x[0]) - 4.0 * b * (x[1] - x[0].powi(2)) * x[0],
        2.0 * b * (x[1] - x[0].powi(2)),
    ];
    (f, g)
};

// 2. Set the initial guess.
let x0 = array![-1.2, 1.0];

// 3. Configure and run the solver.
let solution = Bfgs::new(x0, rosenbrock)
    .with_tolerance(1e-6)
    .with_max_iterations(100)
    .run()
    .expect("BFGS failed to solve");

println!(
    "Found minimum f({:.3}) = {:.4} in {} iterations.",
    solution.final_point, solution.final_value, solution.iterations
);

// The known minimum is at [1.0, 1.0].
assert!((solution.final_point[0] - 1.0).abs() < 1e-5);
assert!((solution.final_point[1] - 1.0).abs() < 1e-5);
```

## Algorithm Details

This crate implements the standard, dense BFGS algorithm. It is **not** a limited-memory (L-BFGS) implementation. The implementation closely follows the methods described in *Numerical Optimization* (2nd ed.) by Nocedal and Wright:

-   **BFGS Update**: The inverse Hessian `H_k` is updated using the standard formula (Eq. 6.17).
-   **Line Search**: A line search satisfying the Strong Wolfe conditions is implemented according to Algorithm 3.5 from the text. This involves a bracketing phase followed by a `zoom` phase (Algorithm 3.6) that uses cubic interpolation for efficient refinement.
-   **Initial Hessian**: A scaling heuristic is used to initialize the inverse Hessian before the first update (Eq. 6.20).

## Testing and Validation

This library is tested against a suite of standard optimization benchmarks. The results are validated against `scipy.optimize.minimize(method='BFGS')` via a Python test harness.

To run the full test suite:
```bash
cargo test --release
```

To run the benchmarks:
```bash
cargo bench
```

## Acknowledgements

This crate is a fork and rewrite of the original [bfgs](https://github.com/paulkernfeld/bfgs) crate by **Paul Kernfeld**. The new version changes the API and replaces the original grid-search-based line search with an implementation based on the Strong Wolfe conditions.

## License

Licensed under either of:
- Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
- MIT license (http://opensource.org/licenses/MIT)

at your option.
