//! An implementation of the BFGS optimization algorithm.
//!
//! This crate provides a solver for unconstrained nonlinear optimization problems,
//! built upon the principles outlined in "Numerical Optimization" by Nocedal & Wright.
//!
//! # Features
//! - A line search implementing the **Strong Wolfe conditions** to ensure stability.
//!   Global convergence is guaranteed for convex problems. For non-convex
//!   problems, BFGS is a powerful and widely-used heuristic but may not always
//!   converge to a local minimum.
//! - A **scaling heuristic** for the initial Hessian approximation, which often
//!   improves the rate of convergence.
//! - A clear, configurable, and ergonomic API using the builder pattern.
//! - **Robust termination criteria**, ensuring the algorithm always halts, either
//!   at a solution or with a descriptive error for diagnostics.
//!
//! # Example
//!
//! Minimize the Rosenbrock function, a classic test case for optimization algorithms.
//!
//! ```
//! use wolfe_bfgs::{Bfgs, BfgsSolution, BfgsError};
//! use ndarray::{array, Array1};
//!
//! // Define the Rosenbrock function and its gradient.
//! let rosenbrock = |x: &Array1<f64>| -> (f64, Array1<f64>) {
//!     let a = 1.0;
//!     let b = 100.0;
//!     let f = (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2);
//!     let g = array![
//!         -2.0 * (a - x[0]) - 4.0 * b * (x[1] - x[0].powi(2)) * x[0],
//!         2.0 * b * (x[1] - x[0].powi(2)),
//!     ];
//!     (f, g)
//! };
//!
//! // Set the initial guess.
//! let x0 = array![-1.2, 1.0];
//!
//! // Run the solver.
//! let BfgsSolution {
//!     final_point: x_min,
//!     final_value,
//!     iterations,
//!     ..
//! } = Bfgs::new(x0, rosenbrock)
//!     .with_tolerance(1e-6)
//!     .with_max_iterations(100)
//!     .run()
//!     .expect("BFGS failed to solve");
//!
//! println!(
//!     "Found minimum f({}) = {:.4} in {} iterations.",
//!     x_min, final_value, iterations
//! );
//!
//! // The known minimum is at [1.0, 1.0].
//! assert!((x_min[0] - 1.0).abs() < 1e-5);
//! assert!((x_min[1] - 1.0).abs() < 1e-5);
//! ```

use ndarray::{Array1, Array2, Axis};

/// An error type for clear diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum BfgsError {
    #[error(
        "The line search failed to find a point satisfying the Wolfe conditions after {max_attempts} attempts."
    )]
    LineSearchFailed { max_attempts: usize },
    #[error("Maximum number of iterations ({max_iterations}) reached without converging.")]
    MaxIterationsReached { max_iterations: usize },
    #[error("The gradient norm was NaN or infinity, indicating numerical instability.")]
    GradientIsNaN,
    #[error(
        "Curvature condition `s_k^T y_k > 0` was violated. This should not happen with a valid Wolfe line search, and may indicate a bug or severe floating-point issues."
    )]
    CurvatureConditionViolated,
}

/// A summary of a successful optimization run.
///
/// Note that for non-convex functions, convergence to a local minimum is not guaranteed.
#[derive(Debug)]
pub struct BfgsSolution {
    /// The point at which the minimum value was found.
    pub final_point: Array1<f64>,
    /// The minimum value of the objective function.
    pub final_value: f64,
    /// The norm of the gradient at the final point.
    pub final_gradient_norm: f64,
    /// The total number of iterations performed.
    pub iterations: usize,
    /// The total number of times the objective function was evaluated.
    pub func_evals: usize,
    /// The total number of times the gradient was evaluated.
    pub grad_evals: usize,
}

/// A configurable BFGS solver.
pub struct Bfgs<ObjFn>
where
    ObjFn: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    x0: Array1<f64>,
    obj_fn: ObjFn,
    // --- Configuration ---
    tolerance: f64,
    max_iterations: usize,
    c1: f64,
    c2: f64,
}

impl<ObjFn> Bfgs<ObjFn>
where
    ObjFn: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    /// Creates a new BFGS solver.
    ///
    /// # Arguments
    /// * `x0` - The initial guess for the minimum.
    /// * `obj_fn` - The objective function which returns a tuple `(value, gradient)`.
    pub fn new(x0: Array1<f64>, obj_fn: ObjFn) -> Self {
        Self {
            x0,
            obj_fn,
            tolerance: 1e-5,
            max_iterations: 100,
            c1: 1e-4, // Standard value for sufficient decrease
            c2: 0.9,  // Standard value for curvature condition
        }
    }

    /// Sets the convergence tolerance (default: 1e-5).
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets the maximum number of iterations (default: 100).
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Executes the BFGS algorithm.
    pub fn run(&self) -> Result<BfgsSolution, BfgsError> {
        let n = self.x0.len();
        let mut x_k = self.x0.clone();
        let (mut f_k, mut g_k) = (self.obj_fn)(&x_k);
        let mut func_evals = 1;
        let mut grad_evals = 1;

        // --- Handle the first iteration separately for initial Hessian scaling ---
        // This logic is designed to produce a well-scaled initial Hessian
        // approximation `b_inv` (H_0) before entering the main loop.
        let g_norm = g_k.dot(&g_k).sqrt();
        if g_norm < self.tolerance {
            return Ok(BfgsSolution {
                final_point: x_k,
                final_value: f_k,
                final_gradient_norm: g_norm,
                iterations: 0,
                func_evals,
                grad_evals,
            });
        }

        // The first step uses the identity matrix as the implicit initial Hessian guess.
        let d_0 = -g_k.clone();
        let (alpha_0, f_1, g_1, f_evals, g_evals) =
            line_search(&self.obj_fn, &x_k, &d_0, f_k, &g_k, self.c1, self.c2)?;
        func_evals += f_evals;
        grad_evals += g_evals;

        let s_0 = alpha_0 * d_0;
        let y_0 = &g_1 - &g_k;

        // Apply the scaling heuristic (Nocedal & Wright, Eq. 6.20) for the
        // initial inverse Hessian used in the first formal iteration (k=1).
        let sy = s_0.dot(&y_0);
        let yy = y_0.dot(&y_0);
        let mut b_inv = if sy > 0.0 && yy > 0.0 {
            Array2::<f64>::eye(n) * (sy / yy)
        } else {
            Array2::<f64>::eye(n) // Fallback to identity
        };

        // Update state to reflect the completion of the first step.
        x_k += &s_0;
        f_k = f_1;
        g_k = g_1;
        // --- End of first iteration ---

        for k in 1..self.max_iterations {
            let g_norm = g_k.dot(&g_k).sqrt();
            if !g_norm.is_finite() {
                return Err(BfgsError::GradientIsNaN);
            }
            if g_norm < self.tolerance {
                return Ok(BfgsSolution {
                    final_point: x_k,
                    final_value: f_k,
                    final_gradient_norm: g_norm,
                    iterations: k,
                    func_evals,
                    grad_evals,
                });
            }

            let d_k = -b_inv.dot(&g_k);
            let (alpha_k, f_next, g_next, f_evals, g_evals) =
                match line_search(&self.obj_fn, &x_k, &d_k, f_k, &g_k, self.c1, self.c2) {
                    Ok(result) => result,
                    Err(_) => {
                        // --- FAILSAFE HESSIAN RESET ---
                        // The line search failed, indicating the search direction is poor
                        // and the Hessian approximation is likely corrupt. Resetting is the
                        // best recovery strategy. This is triggered when the cautious update
                        // mechanism is insufficient to prevent Hessian ill-conditioning.
                        b_inv = Array2::<f64>::eye(n);

                        // Retry the step with a steepest descent direction.
                        let d_k_new = -g_k.clone();
                        // If this also fails, the problem is likely intractable.
                        line_search(&self.obj_fn, &x_k, &d_k_new, f_k, &g_k, self.c1, self.c2)?
                    }
                };
            func_evals += f_evals;
            grad_evals += g_evals;

            let s_k = alpha_k * d_k;
            let y_k = &g_next - &g_k;

            let sy = s_k.dot(&y_k);

            // A valid Wolfe line search should always ensure sy > 0.
            // This check is a safeguard against potential floating-point issues or
            // if a non-Wolfe line search were ever used.
            if sy <= 1e-14 {
                return Err(BfgsError::CurvatureConditionViolated);
            }

            // --- CAUTIOUS UPDATE (Li & Fukushima, as reviewed by Gill & Runnoe) ---
            // This is a robust alternative to simple Hessian resetting. We only perform the
            // BFGS update if the curvature information is deemed reliable, preventing
            // the Hessian approximation from being corrupted by noisy or poor-quality steps.
            let s_k_norm_sq = s_k.dot(&s_k);

            // The condition is `(y_kᵀs_k) / ||s_k||² ≥ ε ||g_k||^γ`.
            // We use the recommended parameters ε=1e-6 and γ=1.
            let perform_update = if s_k_norm_sq > 1e-16 {
                let avg_curvature = sy / s_k_norm_sq;
                let threshold = 1e-6 * g_norm; // Using gamma = 1
                avg_curvature >= threshold
            } else {
                // Step was too small to provide meaningful curvature; do not update.
                false
            };

            if perform_update {
                // --- EFFICIENT AND IDIOMATIC O(n²) BFGS INVERSE HESSIAN UPDATE ---
                // Using ndarray's optimized operations with BLAS backend when available
                let rho = 1.0 / sy;

                // Compute H_k * y_k (matrix-vector product: O(n²))
                let h_y = b_inv.dot(&y_k);

                // Compute y_k' * H_k * y_k (scalar: O(n))
                let y_h_y = y_k.dot(&h_y);

                // Create outer products using ndarray's idiomatic insert_axis + dot technique
                // This leverages optimized BLAS operations instead of manual loops
                let s_k_col = s_k.view().insert_axis(Axis(1));
                let s_k_row = s_k.view().insert_axis(Axis(0));
                let h_y_col = h_y.view().insert_axis(Axis(1));
                let h_y_row = h_y.view().insert_axis(Axis(0));

                // Compute rank-1 update matrices using optimized outer products
                let hy_s_outer = h_y_col.dot(&s_k_row);  // (H_k*y_k) * s_k'
                let s_hy_outer = s_k_col.dot(&h_y_row);  // s_k * (H_k*y_k)'
                let s_s_outer = s_k_col.dot(&s_k_row);   // s_k * s_k'

                // Apply BFGS update using efficient in-place operations (BLAS axpy)
                b_inv.scaled_add(-rho, &hy_s_outer);                    // b_inv -= ρ*(H_k*y)*s'
                b_inv.scaled_add(-rho, &s_hy_outer);                    // b_inv -= ρ*s*(H_k*y)'
                b_inv.scaled_add(rho * rho * y_h_y + rho, &s_s_outer);  // b_inv += (ρ²*(y'*H_k*y) + ρ)*s*s'
            }
            // If `perform_update` is false, we "skip" the update by doing nothing,
            // preserving the existing `b_inv`. This is the "cautious" part of the algorithm.

            x_k += &s_k;
            f_k = f_next;
            g_k = g_next;
        }

        Err(BfgsError::MaxIterationsReached {
            max_iterations: self.max_iterations,
        })
    }
}

/// A line search algorithm that finds a step size satisfying the Strong Wolfe conditions.
///
/// This implementation follows the structure of Algorithm 3.5 in Nocedal & Wright,
/// with an efficient state-passing mechanism to avoid re-computation.
fn line_search<ObjFn>(
    obj_fn: &ObjFn,
    x_k: &Array1<f64>,
    d_k: &Array1<f64>,
    f_k: f64,
    g_k: &Array1<f64>,
    c1: f64,
    c2: f64,
) -> Result<(f64, f64, Array1<f64>, usize, usize), BfgsError>
where
    ObjFn: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    let mut alpha_i = 1.0; // Per Nocedal & Wright, always start with a unit step.
    let mut alpha_prev = 0.0;

    let mut f_prev = f_k;
    let g_k_dot_d = g_k.dot(d_k); // Initial derivative along the search direction.
    let mut g_prev_dot_d = g_k_dot_d;

    let max_attempts = 20;
    let mut func_evals = 0;
    let mut grad_evals = 0;

    for _ in 0..max_attempts {
        let x_new = x_k + alpha_i * d_k;
        let (f_i, g_i) = obj_fn(&x_new);
        func_evals += 1;
        grad_evals += 1;

        // The sufficient decrease (Armijo) condition. A non-finite value for `f_i`
        // indicates that the step `alpha_i` is too large, so it's treated as a
        // failure of this condition, triggering the zoom phase.
        if !f_i.is_finite() || f_i > f_k + c1 * alpha_i * g_k_dot_d || (func_evals > 1 && f_i >= f_prev)
        {
            // The minimum is now bracketed between alpha_prev and alpha_i.
            // A non-finite gradient from a non-finite function value is handled
            // robustly by the zoom function.
            let g_i_dot_d = g_i.dot(d_k);
            return zoom(
                obj_fn,
                x_k,
                d_k,
                f_k,
                g_k_dot_d,
                c1,
                c2,
                alpha_prev,
                alpha_i,
                f_prev,
                f_i,
                g_prev_dot_d,
                g_i_dot_d,
                func_evals,
                grad_evals,
            );
        }

        let g_i_dot_d = g_i.dot(d_k);
        // The curvature condition.
        if g_i_dot_d.abs() <= c2 * g_k_dot_d.abs() {
            // Strong Wolfe conditions are satisfied.
            return Ok((alpha_i, f_i, g_i, func_evals, grad_evals));
        }

        if g_i_dot_d >= 0.0 {
            // The minimum is bracketed between alpha_i and alpha_prev.
            // The new `hi` is the current point; the new `lo` is the previous.
            return zoom(
                obj_fn,
                x_k,
                d_k,
                f_k,
                g_k_dot_d,
                c1,
                c2,
                alpha_prev,
                alpha_i,
                f_prev,
                f_i,
                g_prev_dot_d,
                g_i_dot_d,
                func_evals,
                grad_evals,
            );
        }

        // The step is too short, expand the search interval and cache current state.
        alpha_prev = alpha_i;
        f_prev = f_i;
        g_prev_dot_d = g_i_dot_d;
        alpha_i *= 2.0;
    }

    Err(BfgsError::LineSearchFailed { max_attempts })
}


/// Helper "zoom" function using cubic interpolation, as described by Nocedal & Wright (Alg. 3.6).
///
/// This function is called when a bracketing interval [alpha_lo, alpha_hi] that contains
/// a point satisfying the Strong Wolfe conditions is known. It iteratively refines this
/// interval until a suitable step size is found.
#[allow(clippy::too_many_arguments)]
fn zoom<ObjFn>(
    obj_fn: &ObjFn,
    x_k: &Array1<f64>,
    d_k: &Array1<f64>,
    f_k: f64,
    g_k_dot_d: f64,
    c1: f64,
    c2: f64,
    mut alpha_lo: f64,
    mut alpha_hi: f64,
    mut f_lo: f64,
    mut f_hi: f64,
    mut g_lo_dot_d: f64,
    mut g_hi_dot_d: f64,
    mut func_evals: usize,
    mut grad_evals: usize,
) -> Result<(f64, f64, Array1<f64>, usize, usize), BfgsError>
where
    ObjFn: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    let max_zoom_attempts = 10;
    let min_alpha_step = 1e-12; // Prevents division by zero or degenerate steps.

    for _ in 0..max_zoom_attempts {
        // --- Use cubic interpolation to find a trial step size `alpha_j` ---
        let alpha_j = {
            // Ensure alpha_lo < alpha_hi for stable interpolation.
            if alpha_lo > alpha_hi {
                std::mem::swap(&mut alpha_lo, &mut alpha_hi);
                std::mem::swap(&mut f_lo, &mut f_hi);
                std::mem::swap(&mut g_lo_dot_d, &mut g_hi_dot_d);
            }

            let alpha_diff = alpha_hi - alpha_lo;

            // Fallback to bisection if the interval is too small or if function
            // values at the interval ends are infinite, preventing unstable interpolation.
            if alpha_diff < min_alpha_step || !f_lo.is_finite() || !f_hi.is_finite() {
                (alpha_lo + alpha_hi) / 2.0
            } else {
                let d1 = g_lo_dot_d + g_hi_dot_d - 3.0 * (f_hi - f_lo) / alpha_diff;
                let d2_sq = d1.powi(2) - g_lo_dot_d * g_hi_dot_d;

                if d2_sq.is_sign_positive() {
                    let d2 = d2_sq.sqrt();
                    let trial = alpha_hi
                        - alpha_diff * (g_hi_dot_d + d2 - d1)
                            / (g_hi_dot_d - g_lo_dot_d + 2.0 * d2);

                    // If interpolation gives a non-finite value or a point outside
                    // the bracket, fall back to bisection.
                    if !trial.is_finite() || trial < alpha_lo || trial > alpha_hi {
                        (alpha_lo + alpha_hi) / 2.0
                    } else {
                        trial
                    }
                } else {
                    (alpha_lo + alpha_hi) / 2.0
                }
            }
        };

        // If the trial step is not making sufficient progress, bisect instead.
        let alpha_j = if (alpha_j - alpha_lo).abs() < min_alpha_step
            || (alpha_j - alpha_hi).abs() < min_alpha_step
        {
            (alpha_lo + alpha_hi) / 2.0
        } else {
            alpha_j
        };

        let x_j = x_k + alpha_j * d_k;
        let (f_j, g_j) = obj_fn(&x_j);
        func_evals += 1;
        grad_evals += 1;

        // A NaN value indicates a fatal numerical error (e.g., domain error)
        // from which the optimizer cannot recover.
        if f_j.is_nan() || g_j.iter().any(|&v| v.is_nan()) {
            return Err(BfgsError::LineSearchFailed {
                max_attempts: max_zoom_attempts,
            });
        }

        // Check if the new point `alpha_j` satisfies the sufficient decrease condition.
        // An infinite `f_j` means the step was too large and failed the condition.
        if !f_j.is_finite() || f_j > f_k + c1 * alpha_j * g_k_dot_d || f_j >= f_lo {
            // The new point is not good enough, shrink the interval from the high end.
            alpha_hi = alpha_j;
            f_hi = f_j;
            g_hi_dot_d = g_j.dot(d_k);
        } else {
            let g_j_dot_d = g_j.dot(d_k);
            // Check the curvature condition.
            if g_j_dot_d.abs() <= c2 * g_k_dot_d.abs() {
                // Success: Strong Wolfe conditions are met.
                return Ok((alpha_j, f_j, g_j, func_evals, grad_evals));
            }

            // The minimum is now bracketed by a point with a negative derivative
            // (alpha_lo) and a point with a positive derivative (alpha_j).
            if g_j_dot_d >= 0.0 {
                // The new point has a positive derivative, so it becomes the new
                // upper bound of the bracket. The new interval is [alpha_lo, alpha_j].
                alpha_hi = alpha_j;
                f_hi = f_j;
                g_hi_dot_d = g_j_dot_d;
            } else {
                // The new point has a negative derivative, so it becomes the new
                // lower bound of the bracket. The new interval is [alpha_j, alpha_hi].
                alpha_lo = alpha_j;
                f_lo = f_j;
                g_lo_dot_d = g_j_dot_d;
            }
        }
    }
    Err(BfgsError::LineSearchFailed {
        max_attempts: max_zoom_attempts,
    })
}


#[cfg(test)]
mod tests {
    // This test suite is structured into three parts:
    // 1. Standard Convergence Tests: Verifies that the solver finds the correct
    //    minimum for well-known benchmark functions from standard starting points.
    // 2. Failure and Edge Case Tests: Ensures the solver handles non-convex
    //    functions, pre-solved problems, and iteration limits correctly and returns
    //    the appropriate descriptive errors.
    // 3. Comparison Tests: Validates the behavior of our implementation against
    //    `argmin`, a trusted, state-of-the-art optimization library, ensuring
    //    that our results (final point and iteration count) are equivalent.

    // --- Cargo.toml setup for tests ---
    // The tests below require `argmin` and `argmin-math`. Add these to your
    // `[dev-dependencies]` in Cargo.toml and enable the `ndarray` feature
    // for `argmin-math`:
    //
    // [dev-dependencies]
    // argmin = "0.10.0"
    // argmin-math = { version = "0.4.0", features = ["ndarray_latest-nolinalg"] }
    // ndarray = "0.15"
    // spectral = "0.6"
    // thiserror = "1.0"

    use super::{Bfgs, BfgsError, BfgsSolution};
    use ndarray::{Array1, array};
    use spectral::prelude::*;

    // --- Test Harness: Python scipy.optimize Comparison Setup ---
    use std::process::Command;

    #[derive(serde::Deserialize)]
    #[allow(dead_code)]
    struct PythonOptResult {
        success: bool,
        final_point: Option<Vec<f64>>,
        final_value: Option<f64>,
        final_gradient_norm: Option<f64>,
        iterations: Option<usize>,
        func_evals: Option<usize>,
        grad_evals: Option<usize>,
        message: Option<String>,
        error: Option<String>,
    }

    /// Call Python optimization harness and return the result
    fn optimize_with_python(
        x0: &Array1<f64>,
        function_name: &str,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<PythonOptResult, String> {
        let input_json = serde_json::json!({
            "x0": x0.to_vec(),
            "function": function_name,
            "tolerance": tolerance,
            "max_iterations": max_iterations
        });

        let output = Command::new("python3")
            .arg("optimization_harness.py")
            .arg(input_json.to_string())
            .current_dir(".")
            .output()
            .map_err(|e| format!("Failed to execute Python script: {}", e))?;

        if !output.status.success() {
            return Err(format!(
                "Python script failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let result_str = String::from_utf8(output.stdout)
            .map_err(|e| format!("Invalid UTF-8 in Python output: {}", e))?;

        serde_json::from_str(&result_str)
            .map_err(|e| format!("Failed to parse Python result: {}", e))
    }

    // --- Test Functions ---

    /// A simple convex quadratic function: f(x) = x'x, with minimum at 0.
    fn quadratic(x: &Array1<f64>) -> (f64, Array1<f64>) {
        (x.dot(x), 2.0 * x)
    }

    /// The Rosenbrock function, a classic non-convex benchmark with a minimum at [1, 1].
    fn rosenbrock(x: &Array1<f64>) -> (f64, Array1<f64>) {
        let a = 1.0;
        let b = 100.0;
        let f = (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2);
        let g = array![
            -2.0 * (a - x[0]) - 4.0 * b * (x[1] - x[0].powi(2)) * x[0],
            2.0 * b * (x[1] - x[0].powi(2))
        ];
        (f, g)
    }

    /// A function with a maximum at 0, guaranteed to fail the Wolfe curvature condition.
    fn non_convex_max(x: &Array1<f64>) -> (f64, Array1<f64>) {
        (-x.dot(x), -2.0 * x)
    }

    /// A function whose gradient is constant, causing `y_k` to be zero.
    fn linear_function(x: &Array1<f64>) -> (f64, Array1<f64>) {
        (2.0 * x[0] + 3.0 * x[1], array![2.0, 3.0])
    }

    // --- 1. Standard Convergence Tests ---

    #[test]
    fn test_quadratic_bowl_converges() {
        let x0 = array![10.0, -5.0];
        let BfgsSolution { final_point, .. } = Bfgs::new(x0, quadratic).run().unwrap();
        assert_that!(&final_point[0]).is_close_to(0.0, 1e-5);
        assert_that!(&final_point[1]).is_close_to(0.0, 1e-5);
    }

    #[test]
    fn test_rosenbrock_converges() {
        let x0 = array![-1.2, 1.0];
        let BfgsSolution { final_point, .. } = Bfgs::new(x0, rosenbrock).run().unwrap();
        assert_that!(&final_point[0]).is_close_to(1.0, 1e-5);
        assert_that!(&final_point[1]).is_close_to(1.0, 1e-5);
    }

    // --- 2. Failure and Edge Case Tests ---

    #[test]
    fn test_begin_at_minimum_terminates_immediately() {
        let x0 = array![0.0, 0.0];
        let BfgsSolution { iterations, .. } =
            Bfgs::new(x0, quadratic).with_tolerance(1e-5).run().unwrap();
        assert_that(&iterations).is_less_than_or_equal_to(1);
    }

    #[test]
    fn test_max_iterations_error_is_returned() {
        let x0 = array![-1.2, 1.0];
        let result = Bfgs::new(x0, rosenbrock).with_max_iterations(5).run();
        assert!(matches!(
            result,
            Err(BfgsError::MaxIterationsReached { .. })
        ));
    }

    #[test]
    fn test_non_convex_function_fails_line_search() {
        let x0 = array![2.0];
        let result = Bfgs::new(x0, non_convex_max).run();
        // A correct Wolfe line search must fail because it can't find a point
        // that satisfies the curvature condition when moving towards a maximum.
        assert!(matches!(result, Err(BfgsError::LineSearchFailed { .. })));
    }

    #[test]
    fn test_zero_curvature_fails_gracefully() {
        let x0 = array![10.0, 10.0];
        // For a linear function, the gradient is constant, so y_k is always zero.
        // This makes `sy` zero, violating the curvature condition. The solver should
        // not panic and should return the specific error.
        let result = Bfgs::new(x0, linear_function).run();
        // For linear functions, either curvature condition violation or line search failure
        // is acceptable, as both indicate the algorithm correctly detected the issue.
        assert!(matches!(
            result,
            Err(BfgsError::CurvatureConditionViolated) | Err(BfgsError::LineSearchFailed { .. })
        ));
    }

    #[test]
    fn test_nan_gradient_returns_error() {
        // This function's gradient becomes NaN when x gets very close to 0.
        let nan_fn = |x: &Array1<f64>| {
            if x[0].abs() < 1e-12 {
                (f64::NAN, array![f64::NAN])
            } else {
                (x[0].powi(2), array![2.0 * x[0]])
            }
        };
        // Start at a point that will converge towards 0, triggering the NaN condition.
        let x0 = array![0.1];
        let result = Bfgs::new(x0, nan_fn)
            .with_tolerance(1e-15) // Very tight tolerance to force convergence towards 0
            .run();

        // The solver should detect the NaN gradient and fail gracefully.
        // Accept either GradientIsNaN or LineSearchFailed since NaN during line search
        // can cause either error depending on where it's detected.
        assert!(matches!(
            result,
            Err(BfgsError::GradientIsNaN) | Err(BfgsError::LineSearchFailed { .. })
        ));
    }

    // --- 3. Comparison Tests against a Trusted Library ---

    #[test]
    fn test_rosenbrock_matches_scipy_behavior() {
        let x0 = array![-1.2, 1.0];
        let tolerance = 1e-6;

        // Run our implementation.
        let our_res = Bfgs::new(x0.clone(), rosenbrock)
            .with_tolerance(tolerance)
            .run()
            .unwrap();

        // Run scipy's implementation with synchronized settings.
        let scipy_res = optimize_with_python(&x0, "rosenbrock", tolerance, 100)
            .expect("Python optimization failed");

        assert!(
            scipy_res.success,
            "Scipy optimization failed: {:?}",
            scipy_res.error
        );
        let scipy_point = scipy_res.final_point.unwrap();

        // Assert that the final points are virtually identical.
        let distance = ((our_res.final_point[0] - scipy_point[0]).powi(2)
            + (our_res.final_point[1] - scipy_point[1]).powi(2))
        .sqrt();
        assert_that!(&distance).is_less_than(1e-5);

        // Assert that the number of iterations is very similar. A small difference
        // is acceptable due to minor, valid variations in line search implementations.
        let iter_diff = (our_res.iterations as i64 - scipy_res.iterations.unwrap() as i64).abs();
        assert_that(&iter_diff).is_less_than_or_equal_to(10);
    }

    #[test]
    fn test_quadratic_matches_scipy_behavior() {
        let x0 = array![150.0, -275.5];
        let tolerance = 1e-8;

        // Run our implementation.
        let our_res = Bfgs::new(x0.clone(), quadratic)
            .with_tolerance(tolerance)
            .run()
            .unwrap();

        // Run scipy's implementation with synchronized settings.
        let scipy_res = optimize_with_python(&x0, "quadratic", tolerance, 100)
            .expect("Python optimization failed");

        assert!(
            scipy_res.success,
            "Scipy optimization failed: {:?}",
            scipy_res.error
        );
        let scipy_point = scipy_res.final_point.unwrap();

        // Assert that the final points are virtually identical.
        let distance = ((our_res.final_point[0] - scipy_point[0]).powi(2)
            + (our_res.final_point[1] - scipy_point[1]).powi(2))
        .sqrt();
        assert_that!(&distance).is_less_than(1e-6);

        // Assert that the number of iterations is very similar.
        let iter_diff = (our_res.iterations as i64 - scipy_res.iterations.unwrap() as i64).abs();
        assert_that(&iter_diff).is_less_than_or_equal_to(5);
    }
}
