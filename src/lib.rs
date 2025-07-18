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

use log;
use ndarray::{Array1, Array2, Axis};

// An enum to manage the adaptive strategy.
#[derive(Debug, Clone, Copy)]
enum LineSearchStrategy {
    StrongWolfe,
    Backtracking,
}

enum LineSearchError {
    MaxAttempts(usize),
    StepSizeTooSmall,
}

/// An error type for clear diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum BfgsError {
    #[error("The line search failed to find a suitable step after {max_attempts} attempts. The optimization landscape may be pathological.")]
    LineSearchFailed {
        /// The best solution found before the line search failed.
        last_solution: Box<BfgsSolution>,
        /// The number of attempts the line search made before failing.
        max_attempts: usize,
    },
    #[error("Maximum number of iterations ({max_iterations}) reached without converging.")]
    MaxIterationsReached { max_iterations: usize },
    #[error("The gradient norm was NaN or infinity, indicating numerical instability.")]
    GradientIsNaN,
    #[error("The line search step size became smaller than machine epsilon, indicating that the algorithm is stuck.")]
    StepSizeTooSmall,
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
    const FALLBACK_THRESHOLD: usize = 3;

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

    /// Executes the BFGS algorithm with the adaptive hybrid line search.
    pub fn run(&self) -> Result<BfgsSolution, BfgsError> {
        let n = self.x0.len();
        let mut x_k = self.x0.clone();
        let (mut f_k, mut g_k) = (self.obj_fn)(&x_k);
        let mut func_evals = 1;
        let mut grad_evals = 1;

        let mut b_inv = Array2::<f64>::eye(n);

        // --- State for the Adaptive Strategy ---
        let mut primary_strategy = LineSearchStrategy::StrongWolfe;
        let mut fallback_streak = 0;

        for k in 0..self.max_iterations {
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

            let mut present_d_k = -b_inv.dot(&g_k);
            if present_d_k.iter().all(|&v| v.abs() < 1e-16) {
                return Err(BfgsError::StepSizeTooSmall);
            }

            // --- Adaptive Hybrid Line Search Execution ---
            let (alpha_k, f_next, g_next, f_evals, g_evals) = {
                let search_result = match primary_strategy {
                    LineSearchStrategy::StrongWolfe => line_search(&self.obj_fn, &x_k, &present_d_k, f_k, &g_k, self.c1, self.c2),
                    LineSearchStrategy::Backtracking => backtracking_line_search(&self.obj_fn, &x_k, &present_d_k, f_k, &g_k, self.c1),
                };

                match search_result {
                    Ok(result) => {
                        fallback_streak = 0;
                        result
                    }
                    Err(e) => {
                        // The primary strategy failed.
                        if let LineSearchError::StepSizeTooSmall = e {
                            return Err(BfgsError::StepSizeTooSmall);
                        }

                        // Attempt fallback if the primary strategy was StrongWolfe.
                        if matches!(primary_strategy, LineSearchStrategy::StrongWolfe) {
                            fallback_streak += 1;
                            log::warn!("[BFGS Adaptive] Strong Wolfe failed at iter {}. Falling back to Backtracking.", k);
                            let fallback_result = backtracking_line_search(&self.obj_fn, &x_k, &present_d_k, f_k, &g_k, self.c1);
                            if let Ok(result) = fallback_result {
                                // Fallback succeeded.
                                result
                            } else {
                                // The fallback also failed. Terminate with the informative error.
                                let max_attempts = if let Err(LineSearchError::MaxAttempts(attempts)) = fallback_result { attempts } else { 0 };
                                let last_solution = Box::new(BfgsSolution {
                                    final_point: x_k,
                                    final_value: f_k,
                                    final_gradient_norm: g_norm,
                                    iterations: k,
                                    func_evals,
                                    grad_evals,
                                });
                                return Err(BfgsError::LineSearchFailed { last_solution, max_attempts });
                            }
                        } else {
                            // The robust Backtracking strategy has failed. This is a critical problem.
                            // Reset the Hessian and try one last time with a steepest descent direction.
                            log::error!("[BFGS Adaptive] CRITICAL: Backtracking failed at iter {}. Resetting Hessian.", k);
                            b_inv = Array2::<f64>::eye(n);
                            present_d_k = -g_k.clone();
                            let fallback_result = backtracking_line_search(&self.obj_fn, &x_k, &present_d_k, f_k, &g_k, self.c1);
                            if let Ok(result) = fallback_result {
                                result
                            } else {
                                let max_attempts = if let Err(LineSearchError::MaxAttempts(attempts)) = fallback_result { attempts } else { 0 };
                                let last_solution = Box::new(BfgsSolution {
                                    final_point: x_k,
                                    final_value: f_k,
                                    final_gradient_norm: g_norm,
                                    iterations: k,
                                    func_evals,
                                    grad_evals,
                                });
                                return Err(BfgsError::LineSearchFailed { last_solution, max_attempts });
                            }
                        }
                    }
                }
            };

            // The "Learner" part: promote Backtracking if Wolfe keeps failing.
            if fallback_streak >= Self::FALLBACK_THRESHOLD {
                log::warn!("[BFGS Adaptive] Fallback streak ({}) reached. Switching primary to Backtracking.", fallback_streak);
                primary_strategy = LineSearchStrategy::Backtracking;
                fallback_streak = 0; // Reset the streak after switching
            }

            func_evals += f_evals;
            grad_evals += g_evals;

            let s_k = alpha_k * &present_d_k;
            let y_k = &g_next - &g_k;

            // --- Cautious Hessian Update ---
            let sy = s_k.dot(&y_k);

            if k == 0 {
                // For the very first step, apply the scaling heuristic.
                let yy = y_k.dot(&y_k);
                if sy > 1e-10 && yy > 0.0 {
                    b_inv = Array2::eye(n) * (sy / yy);
                }
            } else {
                // For all subsequent steps, perform the standard BFGS update.
                let s_norm = s_k.dot(&s_k).sqrt();
                let y_norm = y_k.dot(&y_k).sqrt();

                if sy > f64::EPSILON * s_norm * y_norm {
                    let rho = 1.0 / sy;
                    let h_y = b_inv.dot(&y_k);
                    let y_h_y = y_k.dot(&h_y);
                    let s_k_col = s_k.view().insert_axis(Axis(1));
                    let s_k_row = s_k.view().insert_axis(Axis(0));
                    let h_y_col = h_y.view().insert_axis(Axis(1));
                    let h_y_row = h_y.view().insert_axis(Axis(0));
                    let hy_s_outer = h_y_col.dot(&s_k_row);
                    let s_hy_outer = s_k_col.dot(&h_y_row);
                    let s_s_outer = s_k_col.dot(&s_k_row);
                    b_inv.scaled_add(-rho, &hy_s_outer);
                    b_inv.scaled_add(-rho, &s_hy_outer);
                    b_inv.scaled_add(rho * rho * y_h_y + rho, &s_s_outer);

                    // Enforce symmetry to counteract floating-point drift.
                    b_inv = (&b_inv + &b_inv.t()) * 0.5;
                } else {
                    log::debug!("[BFGS] Curvature condition failed (sy / (||s||*||y||) is too small). Skipping Hessian update.");
                }
            }

            x_k += &s_k;
            f_k = f_next;
            g_k = g_next;
        }

        Err(BfgsError::MaxIterationsReached { max_iterations: self.max_iterations })
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
) -> Result<(f64, f64, Array1<f64>, usize, usize), LineSearchError>
where
    ObjFn: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    let mut alpha_i = 1.0; // Per Nocedal & Wright, always start with a unit step.
    let mut alpha_prev = 0.0;

    let mut f_prev = f_k;
    let g_k_dot_d = g_k.dot(d_k); // Initial derivative along the search direction.
    if g_k_dot_d >= 0.0 {
        log::warn!("[BFGS Wolfe] Non-descent direction detected (gᵀd = {:.2e} >= 0).", g_k_dot_d);
    }
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
            // The minimum is bracketed between alpha_prev and alpha_i.
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

    Err(LineSearchError::MaxAttempts(max_attempts))
}

/// A simple backtracking line search that satisfies the Armijo (sufficient decrease) condition.
fn backtracking_line_search<ObjFn>(
    obj_fn: &ObjFn,
    x_k: &Array1<f64>,
    d_k: &Array1<f64>,
    f_k: f64,
    g_k: &Array1<f64>,
    c1: f64,
) -> Result<(f64, f64, Array1<f64>, usize, usize), LineSearchError>
where
    ObjFn: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    let mut alpha = 1.0;
    let rho = 0.5;
    let max_attempts = 30;

    let g_k_dot_d = g_k.dot(d_k);
    // A backtracking search is only valid on a descent direction.
    if g_k_dot_d > 0.0 {
        log::warn!("[BFGS Backtracking] Search started with a non-descent direction (gᵀd = {:.2e} > 0). This step will likely fail.", g_k_dot_d);
    }

    let mut func_evals = 0;
    let mut grad_evals = 0;
    for _ in 0..max_attempts {
        let x_new = x_k + alpha * d_k;
        let (f_new, g_new) = obj_fn(&x_new);
        func_evals += 1;
        grad_evals += 1;

        if f_new.is_finite() && f_new <= f_k + c1 * alpha * g_k_dot_d {
            return Ok((alpha, f_new, g_new, func_evals, grad_evals));
        }

        alpha *= rho;
        // Prevent pathologically small steps from causing an infinite loop.
        if alpha < 1e-16 {
            return Err(LineSearchError::StepSizeTooSmall);
        }
    }

    Err(LineSearchError::MaxAttempts(max_attempts))
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
) -> Result<(f64, f64, Array1<f64>, usize, usize), LineSearchError>
where
    ObjFn: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    let max_zoom_attempts = 10;
    let min_alpha_step = 1e-12; // Prevents division by zero or degenerate steps.

    for _ in 0..max_zoom_attempts {
        // --- Use cubic interpolation to find a trial step size `alpha_j` ---
        // If the entire bracket is in an unusable (infinite) region, fail immediately.
        if !f_lo.is_finite() && !f_hi.is_finite() {
            log::warn!("[BFGS Zoom] Line search bracketed an infinite region. Aborting.");
            return Err(LineSearchError::MaxAttempts(max_zoom_attempts));
        }
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
            return Err(LineSearchError::MaxAttempts(max_zoom_attempts));
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

            // The minimum is bracketed by a point with a negative derivative
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
    Err(LineSearchError::MaxAttempts(max_zoom_attempts))
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

    // A highly ill-conditioned quadratic function.
    // The "valley" is 1000x longer than it is wide.
    fn ill_conditioned_quadratic(x: &Array1<f64>) -> (f64, Array1<f64>) {
        let scale = 1000.0;
        let f = scale * x[0].powi(2) + x[1].powi(2);
        let g = array![2.0 * scale * x[0], 2.0 * x[1]];
        (f, g)
    }

    // This function is minimized anywhere on the line x[0] = -x[1].
    // Its Hessian is singular.
    fn singular_hessian_function(x: &Array1<f64>) -> (f64, Array1<f64>) {
        let val = (x[0] + x[1]).powi(2);
        (val, array![2.0 * (x[0] + x[1]), 2.0 * (x[0] + x[1])])
    }

    // Function with a steep exponential "wall".
    fn wall_with_minimum(x: &Array1<f64>) -> (f64, Array1<f64>) {
        if x[0] > 70.0 {
            // The wall
            (f64::INFINITY, array![f64::INFINITY])
        } else {
            // A simple quadratic with minimum at x=60
            ((x[0] - 60.0).powi(2), array![2.0 * (x[0] - 60.0)])
        }
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
    fn test_non_convex_function_is_handled() {
        let x0 = array![2.0];
        let result = Bfgs::new(x0.clone(), non_convex_max).run();
        // The robust solver should not fail. It gets stuck trying to minimize a function with no minimum.
        // It will hit the max iteration limit because it can't find steps that satisfy the descent condition.
        assert!(matches!(result, Err(BfgsError::MaxIterationsReached { .. })));
    }

    #[test]
    fn test_zero_curvature_is_handled() {
        let x0 = array![10.0, 10.0];
        let result = Bfgs::new(x0, linear_function).run();
        // The solver should skip Hessian updates due to sy=0 and eventually
        // hit the max iteration limit as it cannot make progress.
        assert!(matches!(result, Err(BfgsError::MaxIterationsReached { .. })));
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

    // --- 4. Robustness Tests ---

    #[test]
    fn test_ill_conditioned_problem_converges() {
        let x0 = array![1.0, 1000.0]; // Start far up the narrow valley
        let BfgsSolution { final_point, iterations, .. } =
            Bfgs::new(x0, ill_conditioned_quadratic).run().unwrap();

        // It should still converge, even if it takes more iterations.
        assert_that!(&final_point[0]).is_close_to(0.0, 1e-5);
        assert_that!(&final_point[1]).is_close_to(0.0, 1e-5);
        // Expect more iterations than a well-conditioned problem.
        assert_that!(&iterations).is_greater_than(1);
    }

    #[test]
    fn test_singular_hessian_is_handled_gracefully() {
        let x0 = array![10.0, 20.0];
        let result = Bfgs::new(x0, singular_hessian_function)
            .with_tolerance(1e-8)
            .run();

        // The goal is to ensure the solver doesn't panic or return a numerical error.
        // It can either converge (if it gets lucky) or hit the max iteration limit.
        // Both are "graceful" outcomes.
        match result {
            Ok(soln) => {
                // If it did converge, verify it's on the correct line of minima.
                assert_that!(&soln.final_point[0]).is_close_to(-soln.final_point[1], 1e-5);
                assert_that!(&soln.final_gradient_norm).is_less_than(1e-8);
            }
            Err(BfgsError::MaxIterationsReached { .. }) => {
                // Hitting the iteration limit is an acceptable and expected outcome. Pass.
            }
            Err(e) => {
                // Any other error (like LineSearchFailed, GradientIsNaN) is a failure.
                panic!("Solver failed with an unexpected error: {:?}", e);
            }
        }
    }

    #[test]
    fn test_line_search_handles_inf() {
        let x0 = array![10.0]; // Start far from the wall and minimum.
        let result = Bfgs::new(x0, wall_with_minimum).run();

        // The solver should successfully find the minimum without crashing.
        // The line search must be robust enough to avoid stepping into the "wall".
        let soln = result.unwrap();
        assert_that!(&soln.final_point[0]).is_close_to(60.0, 1e-4);
    }
}
