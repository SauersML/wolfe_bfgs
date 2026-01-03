# Adaptive Hybrid BFGS in Rust

[![Crates.io](https://img.shields.io/crates/v/wolfe-bfgs.svg)](https://crates.io/crates/wolfe-bfgs)
[![Docs.rs](https://docs.rs/wolfe_bfgs/badge.svg)](https://docs.rs/wolfe_bfgs)
[![Build Status](https://github.com/SauersML/wolfe_bfgs/actions/workflows/test.yml/badge.svg)](https://github.com/SauersML/wolfe_bfgs/actions)

A pure Rust implementation of a dense BFGS optimizer with an adaptive, fault-tolerant architecture. It is designed for messy, real-world nonlinear problems (including optional box constraints), and is built on principles from Nocedal & Wright's *Numerical Optimization* with robustness extensions.

This work is a rewrite of the original `bfgs` crate by Paul Kernfeld.

## Features

*   **Adaptive Hybrid Line Search**: Strong Wolfe (cubic interpolation) is the primary strategy, with automatic fallback to nonmonotone Armijo backtracking and approximate-Wolfe/gradient-reduction acceptance when Wolfe fails. The probing grid uses the same nonmonotone/gradient-drop criteria.
*   **Three-Tier Failure Recovery**: Strong Wolfe -> Backtracking Armijo -> Trust-Region Dogleg when line searches break down or produce nonfinite values.
*   **Non-Monotone Acceptance (GLL)**: Uses the Grippo-Lampariello-Lucidi condition to accept steps relative to a recent window, so $f(x_{k+1})$ is not required to decrease every iteration.
*   **Stability Safeguards**: When curvature is weak ($s^T y$ not sufficiently positive), the solver applies Powell damping or skips the update to maintain a stable inverse Hessian.
*   **Bound-Constrained Optimization**: Optional box constraints with projected gradients and coordinate clamping.
*   **Initial Hessian Scaling**: Implements the well-regarded scaling heuristic to produce a well-scaled initial Hessian, often improving the rate of convergence.
*   **Ergonomic API**: Uses the builder pattern for clear and flexible configuration of the solver.
*   **Robust Error Handling**: Provides descriptive errors and returns the best known solution even when the solver exits early.
*   **Dense BFGS Implementation**: Stores the full $n \times n$ inverse Hessian approximation, suitable for small- to medium-scale optimization problems.

## Usage

First, add this to your `Cargo.toml`:

```toml
[dependencies]
wolfe_bfgs = "0.2.0"
```

### Example: Minimizing the Rosenbrock Function

Here is an example of minimizing the 2D Rosenbrock function, a classic benchmark for optimization algorithms.

Note: the objective function can be `FnMut`, so if you store the solver in a variable, call `run()` on a `mut` binding (e.g., `let mut solver = Bfgs::new(...); solver.run();`).

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
    "Found minimum f([{:.3}, {:.3}]) = {:.4} in {} iterations.",
    solution.final_point[0],
    solution.final_point[1],
    solution.final_value,
    solution.iterations
);

// The known minimum is at [1.0, 1.0].
assert!((solution.final_point[0] - 1.0).abs() < 1e-5);
assert!((solution.final_point[1] - 1.0).abs() < 1e-5);
```

### Example: Robust Configuration for Messy Objectives

This example shows a discretized objective (quantized `f`) that can create flat regions. The gradient comes from the underlying smooth model, and the solver is configured to handle plateaus and noisy acceptance.

```rust
use wolfe_bfgs::{Bfgs, BfgsError};
use ndarray::{array, Array1};

let messy = |x: &Array1<f64>| -> (f64, Array1<f64>) {
    let raw = x[0] * x[0] + x[1] * x[1];
    let f = (raw * 100.0).round() / 100.0; // quantized objective (flat regions)
    let g = array![2.0 * x[0], 2.0 * x[1]];
    (f, g)
};

let x0 = array![5.0, -3.0];
let result = Bfgs::new(x0, messy)
    .with_fp_tolerances(1e3, 1e2)
    .with_accept_flat_midpoint_once(true)
    .with_jiggle_on_flats(true, 1e-3)
    .with_multi_direction_rescue(true)
    .with_rescue_hybrid(true)
    .with_rescue_heads(3)
    .with_curvature_slack_scale(1.5)
    .with_flat_stall_exit(true, 4)
    .with_no_improve_stop(1e-7, 8)
    .run();

match result {
    Ok(sol) => println!("Solved: f = {:.4}", sol.final_value),
    Err(BfgsError::MaxIterationsReached { last_solution })
    | Err(BfgsError::LineSearchFailed { last_solution, .. }) => {
        println!("Recovered best f = {:.4}", last_solution.final_value);
    }
    Err(e) => eprintln!("Failed: {e}"),
}
```

## Algorithm Details

This crate implements a dense BFGS algorithm with an adaptive hybrid architecture. It is **not** a limited-memory (L-BFGS) implementation. The implementation is based on *Numerical Optimization* (2nd ed.) by Nocedal and Wright, with robustness extensions:

-   **BFGS Update**: The inverse Hessian $H_k$ is updated to satisfy the secant condition $H_{k+1} y_k = s_k$ while preserving symmetry and positive definiteness.
-   **Line Search (Tier 1)**: Strong Wolfe is attempted first (bracketing + `zoom` with cubic interpolation).
-   **Fallback (Tier 2)**: If Wolfe repeatedly fails, the solver switches to Armijo backtracking with nonmonotone (GLL) acceptance and approximate-Wolfe/gradient-reduction acceptors.
-   **Fallback (Tier 3)**: If line search fails or brackets collapse, a trust-region dogleg step is attempted using CG-based solves on the inverse Hessian.
-   **Non-Monotone Acceptance**: The GLL window allows temporary increases in $f$ as long as the step is good relative to recent history.
-   **Update Safeguards**: Because Armijo/backtracking does not guarantee curvature, stability is enforced via Powell damping or update skipping when $s_k^T y_k$ is insufficient.
-   **Bounds**: When bounds are set, steps are projected and the gradient is zeroed for active constraints (projected gradient).

### Mathematical Formulation

$f(x)$ is the scalar objective and $x_k$ is the current parameter vector. The gradient at $x_k$ is $\nabla f(x_k)$, and $H_k$ is the inverse Hessian approximation used as a local curvature model. The search direction $p_k$ is the quasi-Newton step, and $\alpha_k$ is the line-search step length used to update the parameters:

```math
p_k = -H_k \nabla f(x_k), \quad x_{k+1} = x_k + \alpha_k p_k
```

The BFGS update enforces curvature consistency using the step $s_k = x_{k+1} - x_k$ and the gradient change $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$. The scalar $\rho_k = 1 / (y_k^T s_k)$ normalizes the update, and $I$ is the identity matrix. Together they update $H_k$ so recent gradient changes map to the actual step taken:

```math
H_{k+1} = \left(I - \rho_k s_k y_k^T\right) H_k \left(I - \rho_k y_k s_k^T\right) + \rho_k s_k s_k^T
```

```math
\rho_k = \frac{1}{y_k^T s_k}
```

Strong Wolfe conditions guide the line search by balancing sufficient decrease in $f$ and a drop in the directional derivative along $p_k$. The variable $\alpha$ is a candidate step length along $p_k$, and $c_1$ and $c_2$ are fixed parameters with $0 < c_1 < c_2 < 1$:

```math
f(x_k + \alpha p_k) \le f(x_k) + c_1 \alpha \nabla f(x_k)^T p_k
```

```math
\left|\nabla f(x_k + \alpha p_k)^T p_k\right| \le c_2 \left|\nabla f(x_k)^T p_k\right|
```

When bounds are active, the lower and upper limits are $l$ and $u$, and $\Pi_{[l, u]}(\cdot)$ projects a point into the box. The projected gradient components $g_i$ are set to zero when a coordinate $x_i$ is at a bound and the gradient points outward:

```math
x_{k+1} = \Pi_{[l, u]}(x_k + \alpha_k p_k), \quad
g_i = 0 \text{ if } (x_i = l_i \land g_i \ge 0) \text{ or } (x_i = u_i \land g_i \le 0)
```

If curvature is weak (the inner product $y_k^T s_k$ is too small or nonpositive), damping rescales the update to preserve a stable $H_k$ and maintain descent behavior.

## Advanced Configuration (Rescue Heuristics)

These options are designed for noisy or flat objectives where textbook BFGS can stall:

- `with_multi_direction_rescue(bool)`: After repeated flat accepts, the solver probes small coordinate steps and adopts the one that reduces gradient norm without worsening `f`. Use this on plateaus or when gradients vanish.
- `with_jiggle_on_flats(bool, scale)`: Adds controlled stochastic noise to the backtracking step size when repeated evaluations return the same `f`. Helpful for piecewise-flat or discretized objectives.
- `with_accept_flat_midpoint_once(bool)`: Allows the `zoom` phase to accept a midpoint when the bracket is flat and slopes are nearly identical, preventing infinite loops from floating-point noise.
- `with_rescue_hybrid(bool)`, `with_rescue_heads(usize)`: Configure coordinate-descent rescue behavior (deterministic probing of top-gradient coordinates plus optional random probing).
- `with_curvature_slack_scale(scale)`: Scales the curvature slack used in line-search acceptance tests; higher values are more permissive on noisy objectives.

## Box Constraints

Use `with_bounds(lower, upper, tol)` to enable box constraints:

- **Projection**: Trial points are clamped to `[lower, upper]` per coordinate.
- **Projected gradient**: Active constraints have their gradient components zeroed during search direction updates.

## Termination and Tolerances

The solver can stop for multiple reasons; common ones are:

- `with_tolerance(eps)`: Converges when $\lVert g \rVert < \varepsilon$ (on the projected gradient when bounds are active).
- `with_fp_tolerances(tau_f, tau_g)`: Scales floating-point error compensators so tiny improvements are not lost when $f$ is large (e.g., $10^6$).
- `with_flat_stall_exit(enable, k)`: Stops if $f$ and $x$ remain effectively flat for $k$ consecutive iterations.
- `with_no_improve_stop(tol_f_rel, k)`: Stops after $k$ consecutive iterations without sufficient relative improvement in $f$.

## Error Handling and Recovery

`BfgsError` returns structured errors, and some variants include the best known solution:

- `LineSearchFailed { last_solution, ... }` and `MaxIterationsReached { last_solution }` both return a `BfgsSolution`. You can recover with `error.last_solution.final_point`.
- `GradientIsNaN` is raised before any Hessian update to prevent polluting the inverse Hessian history.
-   **Initial Hessian**: A scaling heuristic is used to initialize the inverse Hessian before the first update.

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

This crate is a fork and rewrite of the original [bfgs](https://github.com/paulkernfeld/bfgs) crate by **Paul Kernfeld**. The new version changes the API and replaces the original grid-search-based line search with a Strong Wolfe primary strategy plus adaptive fallbacks.

## License

Licensed under either of:
- Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
- MIT license (http://opensource.org/licenses/MIT)

at your option.
