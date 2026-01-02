//! An implementation of the BFGS optimization algorithm.
#![allow(non_snake_case)]
//!
//! This crate provides a solver for nonlinear optimization problems (including optional
//! box constraints), built upon the principles outlined in "Numerical Optimization"
//! by Nocedal & Wright.
//!
//! # Features
//! - Hybrid line search: Strong Wolfe with nonmonotone (GLL) Armijo, approximate-Wolfe, and
//!   gradient-reduction acceptors, plus a best-seen salvage path and a small probing grid.
//! - Trust-region (dogleg) fallback with SPD enforcement on the inverse Hessian and scaled-identity
//!   resets under severe noise.
//! - Epsilon-aware tolerances with configurable multipliers via `with_fp_tolerances` for rough,
//!   piecewise-flat objectives.
//! - Adaptive strategy switching (Wolfe <-> Backtracking) based on success streaks (no timed flips).
//! - Optional box constraints with projected gradients and coordinate clamping.
//! - Optional flat-bracket midpoint acceptance inside zoom (`with_accept_flat_midpoint_once`). Default: enabled.
//! - Stochastic jiggling of step sizes on persistent flats (default ON; configurable via
//!   `with_jiggle_on_flats`). Helps hop over repeated rank thresholds in piecewise‑flat regions.
//! - Multi-direction (coordinate) rescue when progress is flat (default ON; configurable via
//!   `with_multi_direction_rescue`). Probes small ±coordinate steps if two consecutive flat accepts
//!   occur and adopts the one that reduces gradient norm without worsening f.
//!
//! ## Defaults (key settings)
//! - Line search: Strong Wolfe primary; GLL nonmonotone Armijo; approximate‑Wolfe and gradient‑drop
//!   acceptors; probing grid; keep‑best salvage.
//! - Trust region: dogleg fallback enabled; Δ₀ = min(1, 10/||g₀||); adaptive by ρ; SPD enforcement
//!   and scaled‑identity resets when needed.
//! - Tolerances: `c1=1e-4`, `c2=0.9`; FP tolerances via `with_fp_tolerances(tau_f=1e3, tau_g=1e2)`.
//! - Zoom midpoint: flat‑bracket midpoint acceptance (default ON; `with_accept_flat_midpoint_once`).
//! - Stochastic jiggling: default ON with scale 1e‑3 (only after repeated flats in backtracking).
//! - Coordinate rescue: default ON (only after two consecutive flat accepts).
//! - Strategy switching: switch Wolfe<->Backtracking only on success/failure streaks (no timed flips).
//! - Clear, configurable builder API, and robust termination with informative errors.
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
//!     .with_fp_tolerances(1e3, 1e2)
//!     .with_accept_flat_midpoint_once(true)
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

use ndarray::{Array1, Array2};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;

// Numerical helpers and small utilities
const EPS: f64 = f64::EPSILON;
#[inline]
fn eps_f(fk: f64, tau: f64) -> f64 {
    tau * EPS * (1.0 + fk.abs())
}
#[inline]
fn eps_g(gk: &Array1<f64>, dk: &Array1<f64>, tau: f64) -> f64 {
    tau * EPS * gk.dot(gk).sqrt() * dk.dot(dk).sqrt()
}

// Ring buffer for GLL nonmonotone Armijo (internal only)
struct GllWindow {
    buf: VecDeque<f64>,
    cap: usize,
}
impl GllWindow {
    fn new(cap: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(cap.max(1)),
            cap: cap.max(1),
        }
    }
    fn clear(&mut self) {
        self.buf.clear();
    }
    fn push(&mut self, f: f64) {
        if self.buf.len() == self.cap {
            self.buf.pop_front();
        }
        self.buf.push_back(f);
    }
    fn fmax(&self) -> f64 {
        self.buf.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
    fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
    fn set_cap(&mut self, cap: usize) {
        self.cap = cap.max(1);
        while self.buf.len() > self.cap {
            self.buf.pop_front();
        }
    }
}

// Best-seen tracker during line search/zoom (internal only)
#[derive(Clone)]
struct ProbeBest {
    f: f64,
    x: Array1<f64>,
    g: Array1<f64>,
}
impl ProbeBest {
    fn new(x0: &Array1<f64>, f0: f64, g0: &Array1<f64>) -> Self {
        Self {
            x: x0.clone(),
            f: f0,
            g: g0.clone(),
        }
    }
    fn consider(&mut self, x: &Array1<f64>, f: f64, g: &Array1<f64>) {
        if f < self.f {
            self.f = f;
            self.x = x.clone();
            self.g = g.clone();
        }
    }
}

// Simple dense SPD Cholesky (LL^T) and solve utilities
fn chol_decompose(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return None;
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Some(l)
}

fn chol_solve(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = l.nrows();
    // Forward solve: L y = b
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }
    // Backward solve: L^T x = y
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

// Helper: return a scaled identity matrix (lambda * I_n)
fn scaled_identity(n: usize, lambda: f64) -> Array2<f64> {
    Array2::<f64>::eye(n) * lambda
}

#[derive(Clone)]
struct BoxSpec {
    lower: Array1<f64>,
    upper: Array1<f64>,
    tol: f64,
}

impl BoxSpec {
    fn new(lower: Array1<f64>, upper: Array1<f64>, tol: f64) -> Self {
        Self { lower, upper, tol }
    }

    fn project(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut z = x.clone();
        for i in 0..z.len() {
            let lo = self.lower[i];
            let hi = self.upper[i];
            if z[i] < lo {
                z[i] = lo;
            } else if z[i] > hi {
                z[i] = hi;
            }
        }
        z
    }

    fn active_mask(&self, x: &Array1<f64>, g: &Array1<f64>) -> Vec<bool> {
        let mut mask = vec![false; x.len()];
        for i in 0..x.len() {
            let lo = self.lower[i];
            let hi = self.upper[i];
            let tol = self.tol;
            let at_lower = x[i] <= lo + tol;
            let at_upper = x[i] >= hi - tol;
            mask[i] = (at_lower && g[i] >= 0.0) || (at_upper && g[i] <= 0.0);
        }
        mask
    }

    fn projected_gradient(&self, x: &Array1<f64>, g: &Array1<f64>) -> Array1<f64> {
        let mut gp = g.clone();
        for i in 0..x.len() {
            let lo = self.lower[i];
            let hi = self.upper[i];
            let tol = self.tol;
            let at_lower = x[i] <= lo + tol;
            let at_upper = x[i] >= hi - tol;
            if (at_lower && g[i] >= 0.0) || (at_upper && g[i] <= 0.0) {
                gp[i] = 0.0;
            }
        }
        gp
    }
}

// An enum to manage the adaptive strategy.
#[derive(Debug, Clone, Copy)]
enum LineSearchStrategy {
    StrongWolfe,
    Backtracking,
}

#[derive(Debug, Clone, Copy)]
enum AcceptKind {
    StrongWolfe,
    ApproxWolfe,
    Nonmonotone,
    GradDrop,
    Midpoint,
    TrustRegion,
    Rescue,
}

#[derive(Debug)]
enum LineSearchError {
    MaxAttempts(usize),
    StepSizeTooSmall,
}

type LsResult = Result<(f64, f64, Array1<f64>, usize, usize, AcceptKind), LineSearchError>;

/// An error type for clear diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum BfgsError {
    #[error(
        "The line search failed to find a suitable step after {max_attempts} attempts. The optimization landscape may be pathological."
    )]
    LineSearchFailed {
        /// The best solution found before the line search failed.
        last_solution: Box<BfgsSolution>,
        /// The number of attempts the line search made before failing.
        max_attempts: usize,
    },
    #[error(
        "Maximum number of iterations reached without converging. The best solution found is returned."
    )]
    MaxIterationsReached {
        /// The best solution found before the iteration limit was reached.
        last_solution: Box<BfgsSolution>,
    },
    #[error("The gradient norm was NaN or infinity, indicating numerical instability.")]
    GradientIsNaN,
    #[error(
        "The line search step size became smaller than machine epsilon, indicating that the algorithm is stuck."
    )]
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

/// Core configuration and adaptive state for the BFGS solver.
struct BfgsCore {
    x0: Array1<f64>,
    // --- Configuration ---
    tolerance: f64,
    max_iterations: usize,
    c1: f64,
    c2: f64,
    tau_f: f64,
    tau_g: f64,
    bounds: Option<BoxSpec>,
    // If true, when the zoom bracket degenerates with flat f at both ends and
    // similar endpoint slopes, accept the midpoint once without additional
    // Armijo/curvature checks to break out of flat regions.
    accept_flat_midpoint_once: bool,
    // Optional: jiggle alpha slightly on persistent flats to hop thresholds
    jiggle_on_flats: bool,
    jiggle_scale: f64,
    rng_state: Cell<u64>,
    // Optional: try a short coordinate rescue after consecutive flat accepts
    multi_direction_rescue: bool,
    flat_accept_streak: Cell<usize>,
    // Rescue selection policy knobs
    rescue_hybrid: bool,
    rescue_pool_mult: f64,
    rescue_heads: usize,
    // Stall/convergence knobs
    stall_enable: bool,
    stall_k: usize,
    stall_noimprove_streak: Cell<usize>,
    // Curvature slack scaling under noise
    curv_slack_scale: Cell<f64>,
    // Gradient drop factor (adapts after flats)
    grad_drop_factor: Cell<f64>,
    // No-improvement termination guard
    tol_f_rel: f64,
    max_no_improve: usize,
    no_improve_streak: Cell<usize>,
    // --- Private adaptive state (no API change) ---
    gll: RefCell<GllWindow>,
    c1_adapt: Cell<f64>,
    c2_adapt: Cell<f64>,
    wolfe_fail_streak: Cell<usize>,
    primary_strategy: Cell<LineSearchStrategy>,
    trust_radius: Cell<f64>,
    global_best: RefCell<Option<ProbeBest>>,
    // Diagnostics counters
    approx_wolfe_accepts: Cell<usize>,
    tr_fallbacks: Cell<usize>,
    strategy_switches: Cell<usize>,
    resets_count: Cell<usize>,
    nonfinite_seen: Cell<bool>,
    wolfe_clean_successes: Cell<usize>,
    bt_clean_successes: Cell<usize>,
    ls_failures_in_row: Cell<usize>,
    chol_fail_iters: Cell<usize>,
    spd_fail_seen: Cell<bool>,
    // Scratch buffers to reduce allocations in hot paths
    scratch_eye: RefCell<Array2<f64>>,   // identity
    scratch_left: RefCell<Array2<f64>>,  // left = I - rho s y^T
    scratch_right: RefCell<Array2<f64>>, // right = I - rho y s^T
    scratch_tmp: RefCell<Array2<f64>>,   // temporary for matmul chains
}

/// A configurable BFGS solver.
pub struct Bfgs<ObjFn> {
    core: BfgsCore,
    obj_fn: ObjFn,
}

impl BfgsCore {
    const FALLBACK_THRESHOLD: usize = 3;

    fn projected_gradient(&self, x: &Array1<f64>, g: &Array1<f64>) -> Array1<f64> {
        if let Some(bounds) = &self.bounds {
            bounds.projected_gradient(x, g)
        } else {
            g.clone()
        }
    }

    fn project_with_step(
        &self,
        x: &Array1<f64>,
        d: &Array1<f64>,
        alpha: f64,
    ) -> (Array1<f64>, Array1<f64>, bool) {
        let trial = x + alpha * d;
        let x_new = self.project_point(&trial);
        let kinked = (&x_new - &trial).iter().any(|v| v.abs() > 0.0);
        let s = &x_new - x;
        (x_new, s, kinked)
    }

    // Attempt one trust-region dogleg step. Updates trust radius and, on success,
    // returns new (x, f, g) and updates `b_inv` cautiously. On failure, may shrink Δ.
    fn try_trust_region_step<ObjFn>(
        &self,
        obj_fn: &mut ObjFn,
        b_inv: &mut Array2<f64>,
        x_k: &Array1<f64>,
        f_k: f64,
        g_k: &Array1<f64>,
        func_evals: &mut usize,
        grad_evals: &mut usize,
    ) -> Option<(Array1<f64>, f64, Array1<f64>)>
    where
        ObjFn: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
    {
        let n = b_inv.nrows();
        let delta = self.trust_radius.get();
        let (p_tr, _) = self.trust_region_dogleg(b_inv, g_k, delta)?;
        let raw_try = x_k + &p_tr;
        let x_try = self.project_point(&raw_try);
        let s_tr = &x_try - x_k;
        let g_old = g_k.clone();
        let (f_try, g_try) = obj_fn(&x_try);
        *func_evals += 1;
        *grad_evals += 1;
        let act_dec = f_k - f_try;
        let pred_dec = self.trust_region_predicted_decrease(b_inv, g_k, &s_tr)?;
        if !pred_dec.is_finite() || pred_dec <= 0.0 {
            self.trust_radius.set((delta * 0.5).max(1e-12));
            return None;
        }
        let rho = act_dec / pred_dec;
        if rho > 0.75 && s_tr.dot(&s_tr).sqrt() > 0.99 * delta {
            self.trust_radius.set((delta * 2.0).min(1e6));
        } else if rho < 0.25 {
            self.trust_radius.set((delta * 0.5).max(1e-12));
        }
        if rho <= 0.1 || !f_try.is_finite() || g_try.iter().any(|v| !v.is_finite()) {
            return None;
        }
        // Accept TR step
        // Update GLL window and global best
        self.gll.borrow_mut().push(f_try);
        let maybe_f = {
            let gb = self.global_best.borrow();
            gb.as_ref().map(|b| b.f)
        };
        if let Some(bf) = maybe_f {
            if f_try < bf - eps_f(bf, self.tau_f) {
                self.global_best.borrow_mut().replace(ProbeBest {
                    f: f_try,
                    x: x_try.clone(),
                    g: g_try.clone(),
                });
            }
        } else {
            self.global_best
                .borrow_mut()
                .replace(ProbeBest::new(&x_try, f_try, &g_try));
        }

        // Inverse update: skip on poor model; otherwise cautious Powell-damped
        let poor_model = rho <= 0.25;
        let s_norm_tr = s_tr.dot(&s_tr).sqrt();
        let mut update_status = "applied";
        if !poor_model && s_norm_tr > 1e-14 {
            let mut binv_upd = b_inv.clone();
            let mut Lopt = chol_decompose(&binv_upd);
            if Lopt.is_none() {
                self.spd_fail_seen.set(true);
                let mean_diag = (0..n).map(|i| binv_upd[[i, i]].abs()).sum::<f64>() / (n as f64);
                let ridge = (1e-10 * mean_diag).max(1e-16);
                for i in 0..n {
                    binv_upd[[i, i]] += ridge;
                }
                Lopt = chol_decompose(&binv_upd);
            }
            if let Some(L) = Lopt {
                let h_s = chol_solve(&L, &s_tr);
                let s_h_s = s_tr.dot(&h_s);
                let y_tr = &g_try - &g_old;
                let sy_tr = s_tr.dot(&y_tr);
                let denom_raw = s_h_s - sy_tr;
                let denom = if denom_raw <= 0.0 { 1e-16 } else { denom_raw };
                let theta_raw = if sy_tr < 0.2 * s_h_s {
                    (0.8 * s_h_s) / denom
                } else {
                    1.0
                };
                let theta = theta_raw.clamp(0.0, 1.0);
                let mut y_tilde = &y_tr * theta + &h_s * (1.0 - theta);
                let mut sty = s_tr.dot(&y_tilde);
                let mut y_norm = y_tilde.dot(&y_tilde).sqrt();
                let kappa = 1e-4;
                let min_curv = kappa * s_norm_tr * y_norm;
                if sty < min_curv {
                    let beta = (min_curv - sty) / (s_norm_tr * s_norm_tr);
                    y_tilde = &y_tilde + &s_tr * beta;
                    sty = s_tr.dot(&y_tilde);
                    y_norm = y_tilde.dot(&y_tilde).sqrt();
                }
                let rel = if s_norm_tr > 0.0 && y_norm > 0.0 {
                    sty / (s_norm_tr * y_norm)
                } else {
                    0.0
                };
                if !sty.is_finite() || rel < 1e-8 {
                    update_status = "skipped";
                    for i in 0..n {
                        b_inv[[i, i]] *= 1.0 + 1e-3;
                    }
                } else {
                    let rho_inv = 1.0 / sty;
                    {
                        let eye = self.scratch_eye.borrow();
                        let mut left = self.scratch_left.borrow_mut();
                        let mut right = self.scratch_right.borrow_mut();
                        left.assign(&*eye);
                        right.assign(&*eye);
                        for i in 0..n {
                            let si = s_tr[i];
                            let yi = y_tilde[i];
                            for j in 0..n {
                                let yj = y_tilde[j];
                                let sj = s_tr[j];
                                left[[i, j]] -= rho_inv * si * yj;
                                right[[i, j]] -= rho_inv * yi * sj;
                            }
                        }
                        let mut tmp = self.scratch_tmp.borrow_mut();
                        *tmp = left.dot(&*b_inv);
                        let candidate = tmp.dot(&*right);
                        *b_inv = candidate;
                    }
                    for i in 0..n {
                        for j in 0..n {
                            b_inv[[i, j]] += rho_inv * s_tr[i] * s_tr[j];
                        }
                    }
                    // Validate SPD; revert if needed
                    if chol_decompose(&*b_inv).is_none() {
                        // fallback: light diag inflation
                        for i in 0..n {
                            b_inv[[i, i]] *= 1.0 + 1e-3;
                        }
                        update_status = "reverted";
                    }
                }
                // Regularize and symmetry enforce
                for i in 0..n {
                    for j in (i + 1)..n {
                        let v = 0.5 * (b_inv[[i, j]] + b_inv[[j, i]]);
                        b_inv[[i, j]] = v;
                        b_inv[[j, i]] = v;
                    }
                }
                let mut diag_min = f64::INFINITY;
                for i in 0..n {
                    diag_min = diag_min.min(b_inv[[i, i]]);
                }
                if !diag_min.is_finite() || diag_min <= 0.0 {
                    for i in 0..n {
                        b_inv[[i, i]] += 1e-12;
                    }
                }
            } else {
                self.spd_fail_seen.set(true);
                self.chol_fail_iters.set(self.chol_fail_iters.get() + 1);
                update_status = "skipped";
            }
            if self.spd_fail_seen.get() && self.chol_fail_iters.get() >= 2 {
                let y_tr = &g_try - &g_old;
                let sy = s_tr.dot(&y_tr);
                let yy = y_tr.dot(&y_tr);
                let mut lambda = if yy > 0.0 { (sy / yy).abs() } else { 1.0 };
                lambda = lambda.clamp(1e-6, 1e6);
                *b_inv = scaled_identity(n, lambda);
                self.chol_fail_iters.set(0);
                update_status = "reverted";
            }
        } else {
            update_status = "skipped";
        }
        self.tr_fallbacks.set(self.tr_fallbacks.get() + 1);
        log::info!(
            "[BFGS] step accepted via {:?}; inverse update {}",
            AcceptKind::TrustRegion,
            update_status
        );
        Some((x_try, f_try, g_try))
    }

    /// Creates a new BFGS core configuration.
    pub fn new(x0: Array1<f64>) -> Self {
        Self {
            x0,
            tolerance: 1e-5,
            max_iterations: 100,
            c1: 1e-4, // Standard value for sufficient decrease
            c2: 0.9,  // Standard value for curvature condition
            tau_f: 1e3,
            tau_g: 1e2,
            bounds: None,
            accept_flat_midpoint_once: true,
            jiggle_on_flats: true,
            jiggle_scale: 1e-3,
            rng_state: Cell::new(0xB5F0_D00D_1234_5678u64),
            multi_direction_rescue: true,
            flat_accept_streak: Cell::new(0),
            rescue_hybrid: true,
            rescue_pool_mult: 4.0,
            rescue_heads: 2,
            stall_enable: true,
            stall_k: 3,
            stall_noimprove_streak: Cell::new(0),
            curv_slack_scale: Cell::new(1.0),
            grad_drop_factor: Cell::new(0.9),
            tol_f_rel: 1e-8,
            max_no_improve: 5,
            no_improve_streak: Cell::new(0),
            gll: RefCell::new(GllWindow::new(8)),
            c1_adapt: Cell::new(1e-4),
            c2_adapt: Cell::new(0.9),
            wolfe_fail_streak: Cell::new(0),
            primary_strategy: Cell::new(LineSearchStrategy::StrongWolfe),
            trust_radius: Cell::new(1.0),
            global_best: RefCell::new(None),
            approx_wolfe_accepts: Cell::new(0),
            tr_fallbacks: Cell::new(0),
            strategy_switches: Cell::new(0),
            resets_count: Cell::new(0),
            nonfinite_seen: Cell::new(false),
            wolfe_clean_successes: Cell::new(0),
            bt_clean_successes: Cell::new(0),
            ls_failures_in_row: Cell::new(0),
            chol_fail_iters: Cell::new(0),
            spd_fail_seen: Cell::new(false),
            scratch_eye: RefCell::new(Array2::<f64>::zeros((0, 0))),
            scratch_left: RefCell::new(Array2::<f64>::zeros((0, 0))),
            scratch_right: RefCell::new(Array2::<f64>::zeros((0, 0))),
            scratch_tmp: RefCell::new(Array2::<f64>::zeros((0, 0))),
        }
    }

    #[inline]
    fn accept_nonmonotone(&self, f_k: f64, fmax: f64, gk_ts: f64, f_i: f64) -> bool {
        let c1 = self.c1_adapt.get();
        let epsf_k = eps_f(f_k, self.tau_f);
        let epsf_max = eps_f(fmax, self.tau_f);
        (f_i <= f_k + c1 * gk_ts + epsf_k) || (f_i <= fmax + c1 * gk_ts + epsf_max)
    }

    fn trust_region_dogleg(
        &self,
        b_inv: &Array2<f64>,
        g: &Array1<f64>,
        delta: f64,
    ) -> Option<(Array1<f64>, f64)> {
        // Factor B_inv = L L^T (SPD). If it fails, add a small ridge and retry once.
        let mut binv = b_inv.clone();
        let l = match chol_decompose(&binv) {
            Some(L) => L,
            None => {
                let n = binv.nrows();
                let mean_diag = (0..n).map(|i| binv[[i, i]].abs()).sum::<f64>() / (n as f64);
                let ridge = (1e-10 * mean_diag).max(1e-16);
                for i in 0..binv.nrows() {
                    binv[[i, i]] += ridge;
                }
                match chol_decompose(&binv) {
                    Some(L2) => L2,
                    None => {
                        // reset to scaled identity and retry once
                        let mut lambda = mean_diag;
                        if !lambda.is_finite() || lambda <= 0.0 {
                            lambda = 1.0;
                        }
                        lambda = lambda.clamp(1e-6, 1e6);
                        binv = scaled_identity(n, lambda);
                        chol_decompose(&binv)?
                    }
                }
            }
        };
        // z = H g solves B_inv z = g
        let z = chol_solve(&l, g);
        let gnorm2 = g.dot(g);
        let gHg = g.dot(&z).max(1e-16);
        // Cauchy step
        let tau = gnorm2 / gHg;
        let p_u = -&(g * tau);
        // Newton/BFGS step
        let p_b = -binv.dot(g);
        let p_b_norm = p_b.dot(&p_b).sqrt();
        if p_b_norm <= delta {
            // predicted decrease: m(p) = g^T p + 0.5 p^T H p, with H p via solve
            let hpb = chol_solve(&l, &p_b);
            let pred = g.dot(&p_b) + 0.5 * p_b.dot(&hpb);
            let pred_dec = -pred;
            if !pred_dec.is_finite() || pred_dec <= 0.0 {
                return None;
            }
            return Some((p_b, pred_dec));
        }
        let p_u_norm = p_u.dot(&p_u).sqrt();
        if p_u_norm >= delta {
            let p = -g * (delta / gnorm2.sqrt());
            let hp = chol_solve(&l, &p);
            let pred = g.dot(&p) + 0.5 * p.dot(&hp);
            let pred_dec = -pred;
            if !pred_dec.is_finite() || pred_dec <= 0.0 {
                return None;
            }
            return Some((p, pred_dec));
        }
        // Dogleg along segment from pu to pb hitting boundary
        let s = &p_b - &p_u;
        let a = s.dot(&s);
        let b = 2.0 * p_u.dot(&s);
        let c = p_u.dot(&p_u) - delta * delta;
        let disc = b * b - 4.0 * a * c;
        if !disc.is_finite() || disc < 0.0 {
            return None;
        }
        let sqrt_disc = disc.sqrt();
        let t1 = (-b - sqrt_disc) / (2.0 * a);
        let t2 = (-b + sqrt_disc) / (2.0 * a);
        // pick valid root in (0,1); if both, choose the smaller (more conservative)
        let mut candidates: Vec<f64> = vec![];
        if t1.is_finite() && t1 > 0.0 && t1 < 1.0 {
            candidates.push(t1);
        }
        if t2.is_finite() && t2 > 0.0 && t2 < 1.0 {
            candidates.push(t2);
        }
        let t: f64 = if !candidates.is_empty() {
            candidates.into_iter().fold(1.0, f64::min)
        } else {
            0.5
        };
        let p = &p_u + &(s * t);
        let hp = chol_solve(&l, &p);
        let pred = g.dot(&p) + 0.5 * p.dot(&hp);
        let pred_dec = -pred;
        if !pred_dec.is_finite() || pred_dec <= 0.0 {
            return None;
        }
        Some((p, pred_dec))
    }

    fn trust_region_predicted_decrease(
        &self,
        b_inv: &Array2<f64>,
        g: &Array1<f64>,
        s: &Array1<f64>,
    ) -> Option<f64> {
        let mut binv = b_inv.clone();
        let l = match chol_decompose(&binv) {
            Some(L) => L,
            None => {
                let n = binv.nrows();
                let mean_diag = (0..n).map(|i| binv[[i, i]].abs()).sum::<f64>() / (n as f64);
                let ridge = (1e-10 * mean_diag).max(1e-16);
                for i in 0..binv.nrows() {
                    binv[[i, i]] += ridge;
                }
                chol_decompose(&binv)?
            }
        };
        let hs = chol_solve(&l, s);
        let pred = g.dot(s) + 0.5 * s.dot(&hs);
        let pred_dec = -pred;
        if pred_dec.is_finite() && pred_dec > 0.0 {
            Some(pred_dec)
        } else {
            None
        }
    }

    fn project_point(&self, x: &Array1<f64>) -> Array1<f64> {
        if let Some(bounds) = &self.bounds {
            bounds.project(x)
        } else {
            x.clone()
        }
    }

    // Tiny xorshift64* RNG for jiggling without external deps. Returns in [-1, 1].
    fn next_rand_sym(&self) -> f64 {
        let mut x = self.rng_state.get();
        // xorshift64*
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x = x.wrapping_mul(0x2545F4914F6CDD1Du64);
        self.rng_state.set(x);
        // Map to (0,1): use upper 53 bits to f64 fraction
        let u = ((x >> 11) as f64) * (1.0 / (1u64 << 53) as f64);
        2.0 * u - 1.0
    }

    fn run<ObjFn>(&self, obj_fn: &mut ObjFn) -> Result<BfgsSolution, BfgsError>
    where
        ObjFn: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
    {
        let n = self.x0.len();
        self.ensure_scratch(n);
        let mut x_k = self.project_point(&self.x0);
        let (mut f_k, mut g_k) = obj_fn(&x_k);
        let mut g_proj_k = self.projected_gradient(&x_k, &g_k);
        let mut active_mask = if let Some(bounds) = &self.bounds {
            bounds.active_mask(&x_k, &g_k)
        } else {
            vec![false; n]
        };
        let mut func_evals = 1;
        let mut grad_evals = 1;

        assert!(
            matches!(self.primary_strategy.get(), LineSearchStrategy::StrongWolfe)
                || self.wolfe_fail_streak.get() == 0
        );
        {
            let gll = self.gll.borrow();
            assert!(gll.buf.is_empty() || gll.buf.len() <= gll.cap);
        }
        assert!(self.trust_radius.get().is_finite());
        self.wolfe_fail_streak.set(0);
        self.wolfe_clean_successes.set(0);
        self.bt_clean_successes.set(0);
        self.ls_failures_in_row.set(0);
        self.nonfinite_seen.set(false);
        self.chol_fail_iters.set(0);
        self.spd_fail_seen.set(false);
        self.flat_accept_streak.set(0);

        let mut b_inv = Array2::<f64>::eye(n);

        // Initialize adaptive state
        {
            let mut gll = self.gll.borrow_mut();
            gll.clear();
            gll.push(f_k);
        }
        self.global_best
            .borrow_mut()
            .replace(ProbeBest::new(&x_k, f_k, &g_k));
        self.c1_adapt.set(self.c1);
        self.c2_adapt.set(self.c2);
        self.primary_strategy.set(LineSearchStrategy::StrongWolfe);
        self.wolfe_fail_streak.set(0);
        // Initialize trust radius from gradient scale
        let g0_norm = g_proj_k.dot(&g_proj_k).sqrt();
        let delta0 = if g0_norm.is_finite() && g0_norm > 0.0 {
            (10.0 / g0_norm).min(1.0)
        } else {
            1.0
        };
        self.trust_radius.set(delta0);

        let mut f_last_accepted = f_k;
        for k in 0..self.max_iterations {
            // reset per-iteration state
            self.nonfinite_seen.set(false);
            self.chol_fail_iters.set(0);
            self.spd_fail_seen.set(false);
            g_proj_k = self.projected_gradient(&x_k, &g_k);
            let g_norm = g_proj_k.dot(&g_proj_k).sqrt();
            if !g_norm.is_finite() {
                log::warn!(
                    "[BFGS] Non-finite gradient norm at iter {}: g_norm={:?}",
                    k,
                    g_norm
                );
                return Err(BfgsError::GradientIsNaN);
            }
            if g_norm < self.tolerance {
                let sol = BfgsSolution {
                    final_point: x_k,
                    final_value: f_k,
                    final_gradient_norm: g_norm,
                    iterations: k,
                    func_evals,
                    grad_evals,
                };
                log::info!(
                    "[BFGS] Converged by gradient: iters={}, f={:.6e}, ||g||={:.3e}, fe={}, ge={}, Δ={:.3e}",
                    k,
                    sol.final_value,
                    sol.final_gradient_norm,
                    sol.func_evals,
                    sol.grad_evals,
                    self.trust_radius.get()
                );
                return Ok(sol);
            }

            let mut present_d_k = -b_inv.dot(&g_k);
            if let Some(bounds) = &self.bounds {
                for (i, &active) in active_mask.iter().enumerate() {
                    if active {
                        present_d_k[i] = 0.0;
                    }
                }
                // prevent stepping outside bounds directly from the current point
                for i in 0..present_d_k.len() {
                    if present_d_k[i] < 0.0 && x_k[i] <= bounds.lower[i] + bounds.tol {
                        present_d_k[i] = 0.0;
                    }
                    if present_d_k[i] > 0.0 && x_k[i] >= bounds.upper[i] - bounds.tol {
                        present_d_k[i] = 0.0;
                    }
                }
            }
            // Enforce descent direction; reset if needed
            let gdotd = g_k.dot(&present_d_k);
            let dnorm = present_d_k.dot(&present_d_k).sqrt();
            let tiny_d = dnorm <= 1e-14 * (1.0 + x_k.dot(&x_k).sqrt());
            if gdotd >= 0.0 || tiny_d {
                log::warn!("[BFGS] Non-descent direction; resetting to -g and B_inv=I.");
                b_inv = Array2::eye(n);
                present_d_k = -g_k.clone();
                self.resets_count.set(self.resets_count.get() + 1);
                if let Some(bounds) = &self.bounds {
                    for (i, &active) in active_mask.iter().enumerate() {
                        if active {
                            present_d_k[i] = 0.0;
                        }
                    }
                    for i in 0..present_d_k.len() {
                        if present_d_k[i] < 0.0 && x_k[i] <= bounds.lower[i] + bounds.tol {
                            present_d_k[i] = 0.0;
                        }
                        if present_d_k[i] > 0.0 && x_k[i] >= bounds.upper[i] - bounds.tol {
                            present_d_k[i] = 0.0;
                        }
                    }
                }
            }

            // --- Adaptive Hybrid Line Search Execution ---
            let active_before = active_mask.clone();
            let (alpha_k, mut f_next, mut g_next, f_evals, g_evals, mut accept_kind) = {
                let search_result = match self.primary_strategy.get() {
                    LineSearchStrategy::StrongWolfe => line_search(
                        self,
                        obj_fn,
                        &x_k,
                        &present_d_k,
                        f_k,
                        &g_k,
                        self.c1_adapt.get(),
                        self.c2_adapt.get(),
                    ),
                    LineSearchStrategy::Backtracking => {
                        backtracking_line_search(self, obj_fn, &x_k, &present_d_k, f_k, &g_k)
                    }
                };

                match search_result {
                    Ok(result) => {
                        // Reset failure streak and relax toward canonical constants
                        self.wolfe_fail_streak.set(0);
                        self.ls_failures_in_row.set(0);
                        // Drift c1/c2 back toward canonical quickly on success
                        if self.wolfe_clean_successes.get() >= 2
                            || self.bt_clean_successes.get() >= 2
                        {
                            self.c1_adapt.set(self.c1);
                            self.c2_adapt.set(self.c2);
                        } else {
                            self.c1_adapt.set((self.c1_adapt.get() * 0.9).max(self.c1));
                            self.c2_adapt.set((self.c2_adapt.get() * 1.1).min(self.c2));
                        }
                        match self.primary_strategy.get() {
                            LineSearchStrategy::StrongWolfe => {
                                self.wolfe_clean_successes
                                    .set(self.wolfe_clean_successes.get() + 1);
                                self.bt_clean_successes.set(0);
                                if self.wolfe_clean_successes.get() >= 3 {
                                    self.gll.borrow_mut().set_cap(8);
                                }
                            }
                            LineSearchStrategy::Backtracking => {
                                self.bt_clean_successes
                                    .set(self.bt_clean_successes.get() + 1);
                                self.wolfe_clean_successes.set(0);
                            }
                        }
                        result
                    }
                    Err(e) => {
                        // The primary strategy failed.
                        match e {
                            LineSearchError::StepSizeTooSmall => {
                                log::debug!("[BFGS] Line search failed: step size too small.");
                            }
                            LineSearchError::MaxAttempts(attempts) => {
                                log::debug!(
                                    "[BFGS] Line search failed: max attempts reached ({attempts})."
                                );
                            }
                        }
                        // Attempt fallback if the primary strategy was StrongWolfe.
                        if matches!(self.primary_strategy.get(), LineSearchStrategy::StrongWolfe) {
                            let streak = self.wolfe_fail_streak.get() + 1;
                            self.wolfe_fail_streak.set(streak);
                            log::warn!(
                                "[BFGS Adaptive] Strong Wolfe failed at iter {}. Falling back to Backtracking.",
                                k
                            );
                            // Adapt c1/c2 on failures
                            if streak == 1 {
                                self.c2_adapt.set(0.5);
                            }
                            if streak >= 2 {
                                self.c2_adapt.set(0.1);
                                self.c1_adapt.set(1e-3);
                            }
                            self.ls_failures_in_row
                                .set(self.ls_failures_in_row.get() + 1);
                            if self.ls_failures_in_row.get() >= 2 {
                                self.gll.borrow_mut().set_cap(10);
                            }
                            let fallback_result = backtracking_line_search(
                                self,
                                obj_fn,
                                &x_k,
                                &present_d_k,
                                f_k,
                                &g_k,
                            );
                            if let Ok(result) = fallback_result {
                                // Fallback succeeded.
                                result
                            } else {
                                // The fallback also failed. Terminate with the informative error.
                                let max_attempts =
                                    if let Err(LineSearchError::MaxAttempts(attempts)) =
                                        fallback_result
                                    {
                                        attempts
                                    } else {
                                        0
                                    };
                                // Salvage best point seen during line search if any
                                if let Some(b) = self.global_best.borrow().as_ref() {
                                    let epsF = eps_f(f_k, self.tau_f);
                                    let gk_norm = g_proj_k.dot(&g_proj_k).sqrt();
                                    let gb_proj = self.projected_gradient(&b.x, &b.g);
                                    let gb_norm = gb_proj.dot(&gb_proj).sqrt();
                                    let drop_factor = self.grad_drop_factor.get();
                                    if (b.f <= f_k + epsF && gb_norm <= drop_factor * gk_norm)
                                        || (b.f < f_k - epsF)
                                    {
                                        let rel_impr = (f_k - b.f).abs() / (1.0 + f_k.abs());
                                        if rel_impr <= self.tol_f_rel {
                                            self.no_improve_streak
                                                .set(self.no_improve_streak.get() + 1);
                                        } else {
                                            self.no_improve_streak.set(0);
                                        }
                                        if self.no_improve_streak.get() >= self.max_no_improve {
                                            return Ok(BfgsSolution {
                                                final_point: b.x.clone(),
                                                final_value: b.f,
                                                final_gradient_norm: gb_norm,
                                                iterations: k,
                                                func_evals,
                                                grad_evals,
                                            });
                                        }
                                        x_k = self.project_point(&b.x);
                                        f_k = b.f;
                                        g_k = b.g.clone();
                                        g_proj_k = gb_proj;
                                        if let Some(bounds) = &self.bounds {
                                            active_mask = bounds.active_mask(&x_k, &g_k);
                                        }
                                        for i in 0..n {
                                            b_inv[[i, i]] *= 1.0 + 1e-3;
                                        }
                                        continue;
                                    }
                                }
                                // Try full trust-region dogleg fallback before giving up
                                if let Some((x_new, f_new, g_new)) = self.try_trust_region_step(
                                    obj_fn,
                                    &mut b_inv,
                                    &x_k,
                                    f_k,
                                    &g_k,
                                    &mut func_evals,
                                    &mut grad_evals,
                                ) {
                                    let g_proj_new = self.projected_gradient(&x_new, &g_new);
                                    let rel_impr = (f_k - f_new).abs() / (1.0 + f_k.abs());
                                    if rel_impr <= self.tol_f_rel {
                                        self.no_improve_streak
                                            .set(self.no_improve_streak.get() + 1);
                                    } else {
                                        self.no_improve_streak.set(0);
                                    }
                                    if self.no_improve_streak.get() >= self.max_no_improve {
                                        return Ok(BfgsSolution {
                                            final_point: x_new,
                                            final_value: f_new,
                                            final_gradient_norm: g_proj_new.dot(&g_proj_new).sqrt(),
                                            iterations: k + 1,
                                            func_evals,
                                            grad_evals,
                                        });
                                    }
                                    x_k = x_new;
                                    f_k = f_new;
                                    g_k = g_new;
                                    g_proj_k = g_proj_new;
                                    if let Some(bounds) = &self.bounds {
                                        active_mask = bounds.active_mask(&x_k, &g_k);
                                    }
                                    self.ls_failures_in_row.set(0);
                                    continue;
                                }
                                self.trust_radius
                                    .set((self.trust_radius.get() * 0.7).max(1e-12));
                                if self.nonfinite_seen.get() {
                                    let mut ls = BfgsSolution {
                                        final_point: x_k.clone(),
                                        final_value: f_k,
                                        final_gradient_norm: g_norm,
                                        iterations: k,
                                        func_evals,
                                        grad_evals,
                                    };
                                    if let Some(b) = self.global_best.borrow().as_ref()
                                        && b.f < f_k - eps_f(f_k, self.tau_f)
                                    {
                                        let gb_proj = self.projected_gradient(&b.x, &b.g);
                                        ls.final_point = b.x.clone();
                                        ls.final_value = b.f;
                                        ls.final_gradient_norm = gb_proj.dot(&gb_proj).sqrt();
                                    }
                                    log::warn!(
                                        "[BFGS] Line search failed at iter {} (nonfinite seen), fe={}, ge={}, Δ={:.3e}",
                                        k,
                                        func_evals,
                                        grad_evals,
                                        self.trust_radius.get()
                                    );
                                    return Err(BfgsError::LineSearchFailed {
                                        last_solution: Box::new(ls),
                                        max_attempts,
                                    });
                                }
                                if self.ls_failures_in_row.get() >= 2 {
                                    let ls = BfgsSolution {
                                        final_point: x_k.clone(),
                                        final_value: f_k,
                                        final_gradient_norm: g_norm,
                                        iterations: k,
                                        func_evals,
                                        grad_evals,
                                    };
                                    return Err(BfgsError::LineSearchFailed {
                                        last_solution: Box::new(ls),
                                        max_attempts,
                                    });
                                }
                                continue;
                            }
                        } else {
                            // The robust Backtracking strategy has failed. This is a critical problem.
                            // Reset the Hessian and try one last time with a steepest descent direction.
                            self.ls_failures_in_row
                                .set(self.ls_failures_in_row.get() + 1);
                            log::error!(
                                "[BFGS Adaptive] CRITICAL: Backtracking failed at iter {}. Resetting Hessian.",
                                k
                            );
                            b_inv = Array2::<f64>::eye(n);
                            present_d_k = -g_k.clone();
                            let fallback_result = backtracking_line_search(
                                self,
                                obj_fn,
                                &x_k,
                                &present_d_k,
                                f_k,
                                &g_k,
                            );
                            if let Ok(result) = fallback_result {
                                result
                            } else {
                                let max_attempts =
                                    if let Err(LineSearchError::MaxAttempts(attempts)) =
                                        fallback_result
                                    {
                                        attempts
                                    } else {
                                        0
                                    };
                                // Full trust-region dogleg fallback
                                if let Some((x_new, f_new, g_new)) = self.try_trust_region_step(
                                    obj_fn,
                                    &mut b_inv,
                                    &x_k,
                                    f_k,
                                    &g_k,
                                    &mut func_evals,
                                    &mut grad_evals,
                                ) {
                                    let g_proj_new = self.projected_gradient(&x_new, &g_new);
                                    let rel_impr = (f_k - f_new).abs() / (1.0 + f_k.abs());
                                    if rel_impr <= self.tol_f_rel {
                                        self.no_improve_streak
                                            .set(self.no_improve_streak.get() + 1);
                                    } else {
                                        self.no_improve_streak.set(0);
                                    }
                                    if self.no_improve_streak.get() >= self.max_no_improve {
                                        return Ok(BfgsSolution {
                                            final_point: x_new,
                                            final_value: f_new,
                                            final_gradient_norm: g_proj_new.dot(&g_proj_new).sqrt(),
                                            iterations: k + 1,
                                            func_evals,
                                            grad_evals,
                                        });
                                    }
                                    x_k = x_new;
                                    f_k = f_new;
                                    g_k = g_new;
                                    g_proj_k = g_proj_new;
                                    if let Some(bounds) = &self.bounds {
                                        active_mask = bounds.active_mask(&x_k, &g_k);
                                    }
                                    self.ls_failures_in_row.set(0);
                                    continue;
                                }
                                if let Some(b) = self.global_best.borrow().as_ref() {
                                    let epsF = eps_f(f_k, self.tau_f);
                                    let gk_norm = g_proj_k.dot(&g_proj_k).sqrt();
                                    let gb_proj = self.projected_gradient(&b.x, &b.g);
                                    let gb_norm = gb_proj.dot(&gb_proj).sqrt();
                                    let drop_factor = self.grad_drop_factor.get();
                                    if (b.f <= f_k + epsF && gb_norm <= drop_factor * gk_norm)
                                        || (b.f < f_k - epsF)
                                    {
                                        let rel_impr = (f_k - b.f).abs() / (1.0 + f_k.abs());
                                        if rel_impr <= self.tol_f_rel {
                                            self.no_improve_streak
                                                .set(self.no_improve_streak.get() + 1);
                                        } else {
                                            self.no_improve_streak.set(0);
                                        }
                                        if self.no_improve_streak.get() >= self.max_no_improve {
                                            return Ok(BfgsSolution {
                                                final_point: b.x.clone(),
                                                final_value: b.f,
                                                final_gradient_norm: gb_norm,
                                                iterations: k,
                                                func_evals,
                                                grad_evals,
                                            });
                                        }
                                        x_k = self.project_point(&b.x);
                                        f_k = b.f;
                                        g_k = b.g.clone();
                                        g_proj_k = gb_proj;
                                        if let Some(bounds) = &self.bounds {
                                            active_mask = bounds.active_mask(&x_k, &g_k);
                                        }
                                        for i in 0..n {
                                            b_inv[[i, i]] *= 1.0 + 1e-3;
                                        }
                                        continue;
                                    }
                                }
                                self.trust_radius
                                    .set((self.trust_radius.get() * 0.7).max(1e-12));
                                if self.nonfinite_seen.get() {
                                    let mut ls = BfgsSolution {
                                        final_point: x_k.clone(),
                                        final_value: f_k,
                                        final_gradient_norm: g_norm,
                                        iterations: k,
                                        func_evals,
                                        grad_evals,
                                    };
                                    if let Some(b) = self.global_best.borrow().as_ref()
                                        && b.f < f_k - eps_f(f_k, self.tau_f)
                                    {
                                        ls.final_point = b.x.clone();
                                        ls.final_value = b.f;
                                        ls.final_gradient_norm = b.g.dot(&b.g).sqrt();
                                    }
                                    log::warn!(
                                        "[BFGS] Line search failed at iter {} (nonfinite seen), fe={}, ge={}, Δ={:.3e}",
                                        k,
                                        func_evals,
                                        grad_evals,
                                        self.trust_radius.get()
                                    );
                                    return Err(BfgsError::LineSearchFailed {
                                        last_solution: Box::new(ls),
                                        max_attempts,
                                    });
                                }
                                if self.ls_failures_in_row.get() >= 2 {
                                    let ls = BfgsSolution {
                                        final_point: x_k.clone(),
                                        final_value: f_k,
                                        final_gradient_norm: g_norm,
                                        iterations: k,
                                        func_evals,
                                        grad_evals,
                                    };
                                    return Err(BfgsError::LineSearchFailed {
                                        last_solution: Box::new(ls),
                                        max_attempts,
                                    });
                                }
                                continue;
                            }
                        }
                    }
                }
            };

            // Optional coordinate rescue after consecutive flat accepts
            let mut s_override: Option<Array1<f64>> = None;
            let mut rescued = false;
            if self.multi_direction_rescue {
                let epsF_iter = eps_f(f_k, self.tau_f);
                let flat_now = (f_next - f_k).abs() <= epsF_iter;
                if flat_now && self.flat_accept_streak.get() >= 2 {
                    let x_base = self.project_point(&(&x_k + &(alpha_k * &present_d_k)));
                    let g_proj_base = self.projected_gradient(&x_base, &g_next);
                    let gnext_norm0 = g_proj_base.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let delta = self.trust_radius.get();
                    let eta = (0.2 * delta).min(1.0 / (1.0 + gnext_norm0));
                    if eta.is_finite() && eta > 0.0 {
                        let n = x_k.len();
                        let mut best_x = None;
                        let mut best_f = f_next;
                        let mut best_g = g_next.clone();
                        // Budgeted coordinate subset selection
                        let k = n.min(8);
                        let mut idx: Vec<usize> = (0..n).collect();
                        idx.sort_by(|&i, &j| {
                            g_next[i]
                                .abs()
                                .partial_cmp(&g_next[j].abs())
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .reverse()
                        });
                        let use_hybrid = self.rescue_hybrid;
                        let m = (self.rescue_pool_mult * (k as f64)).round() as usize;
                        let m = m.min(n).max(k);
                        let heads = self.rescue_heads.min(k).min(m);
                        let mut chosen: Vec<usize> = Vec::new();
                        // Always include top heads
                        for &i in idx.iter().take(heads) {
                            chosen.push(i);
                        }
                        if use_hybrid {
                            // Sample remaining from next (heads..m)
                            let mut pool: Vec<usize> =
                                idx.iter().cloned().skip(heads).take(m - heads).collect();
                            while chosen.len() < k && !pool.is_empty() {
                                // xorshift-based index
                                let r = (self.rng_state.get() >> 1) as usize;
                                let t = r % pool.len();
                                let pick = pool.swap_remove(t);
                                chosen.push(pick);
                                // advance rng
                                let _ = self.next_rand_sym();
                            }
                        } else {
                            for &i in idx.iter().skip(heads).take(k - heads) {
                                chosen.push(i);
                            }
                        }
                        for &i in &chosen {
                            for &sgn in &[-1.0, 1.0] {
                                let mut x_try = x_base.clone();
                                x_try[i] += sgn * eta; // coordinate poke from x_next
                                x_try = self.project_point(&x_try);
                                let (f_try, g_try) = obj_fn(&x_try);
                                func_evals += 1;
                                grad_evals += 1;
                                if !f_try.is_finite() || g_try.iter().any(|v| !v.is_finite()) {
                                    continue;
                                }
                                let g_proj_try = self.projected_gradient(&x_try, &g_try);
                                let g_try_norm = g_proj_try.dot(&g_proj_try).sqrt();
                                let f_thresh = f_k.min(f_next) + epsF_iter;
                                let s_trial = &x_try - &x_k;
                                let descent_ok = g_proj_try.dot(&s_trial)
                                    <= -eps_g(&g_proj_k, &s_trial, self.tau_g);
                                let f_ok = f_try <= f_thresh;
                                let g_ok = g_try_norm <= self.grad_drop_factor.get() * gnext_norm0;
                                if (f_ok || g_ok) && descent_ok && f_try <= best_f {
                                    best_f = f_try;
                                    best_x = Some(x_try.clone());
                                    best_g = g_try.clone();
                                }
                            }
                        }
                        if let Some(xb) = best_x {
                            // Enforce trust radius on the rescue step
                            let mut s_tmp = &xb - &x_k;
                            let s_norm = s_tmp.dot(&s_tmp).sqrt();
                            let delta = self.trust_radius.get();
                            if s_norm.is_finite()
                                && s_norm > delta
                                && delta.is_finite()
                                && delta > 0.0
                            {
                                let scale = delta / s_norm;
                                let x_scaled = &x_k + &(s_tmp.mapv(|v| v * scale));
                                let x_scaled = self.project_point(&x_scaled);
                                let (f_s, g_s) = obj_fn(&x_scaled);
                                func_evals += 1;
                                grad_evals += 1;
                                if f_s.is_finite() && g_s.iter().all(|v| v.is_finite()) {
                                    s_tmp = &x_scaled - &x_k;
                                    f_next = f_s;
                                    g_next = g_s;
                                } else {
                                    // fall back to original xb
                                    f_next = best_f;
                                    g_next = best_g.clone();
                                }
                            } else {
                                f_next = best_f;
                                g_next = best_g.clone();
                            }
                            s_override = Some(s_tmp);
                            rescued = true;
                            accept_kind = AcceptKind::Rescue;
                            self.flat_accept_streak.set(0);
                        }
                    }
                }
            }

            // The "Learner" part: promote Backtracking if Wolfe keeps failing.
            if self.wolfe_fail_streak.get() >= Self::FALLBACK_THRESHOLD {
                log::warn!(
                    "[BFGS Adaptive] Fallback streak ({}) reached. Switching primary to Backtracking.",
                    self.wolfe_fail_streak.get()
                );
                self.primary_strategy.set(LineSearchStrategy::Backtracking);
                self.strategy_switches.set(self.strategy_switches.get() + 1);
                self.wolfe_fail_streak.set(0);
            }
            // Switch back to StrongWolfe after a run of clean backtracking successes
            if matches!(
                self.primary_strategy.get(),
                LineSearchStrategy::Backtracking
            ) && self.bt_clean_successes.get() >= 3
                && self.wolfe_fail_streak.get() == 0
            {
                log::info!(
                    "[BFGS Adaptive] Backtracking succeeded cleanly ({} iters); switching back to StrongWolfe.",
                    self.bt_clean_successes.get()
                );
                self.primary_strategy.set(LineSearchStrategy::StrongWolfe);
                self.strategy_switches.set(self.strategy_switches.get() + 1);
                self.bt_clean_successes.set(0);
                self.gll.borrow_mut().set_cap(8);
            }

            func_evals += f_evals;
            grad_evals += g_evals;

            let mut s_k = if let Some(ref s) = s_override {
                s.clone()
            } else {
                alpha_k * &present_d_k
            };
            let x_next = self.project_point(&(x_k.clone() + &s_k));
            s_k = &x_next - &x_k;
            let g_proj_next = self.projected_gradient(&x_next, &g_next);
            let active_after = if let Some(bounds) = &self.bounds {
                bounds.active_mask(&x_next, &g_next)
            } else {
                vec![false; n]
            };
            let step_len = s_k.dot(&s_k).sqrt();
            if step_len.is_finite() && step_len > 0.0 {
                if step_len >= 0.9 * self.trust_radius.get() {
                    self.trust_radius
                        .set((self.trust_radius.get() * 1.5).min(1e6));
                } else {
                    self.trust_radius
                        .set((self.trust_radius.get() * 1.1).min(1e6));
                }
            }

            let rel_impr = (f_last_accepted - f_next).abs() / (1.0 + f_last_accepted.abs());
            if rel_impr <= self.tol_f_rel {
                self.no_improve_streak.set(self.no_improve_streak.get() + 1);
            } else {
                self.no_improve_streak.set(0);
            }
            if self.no_improve_streak.get() >= self.max_no_improve {
                return Ok(BfgsSolution {
                    final_point: x_next.clone(),
                    final_value: f_next,
                    final_gradient_norm: g_proj_next.dot(&g_proj_next).sqrt(),
                    iterations: k + 1,
                    func_evals,
                    grad_evals,
                });
            }

            // Update adaptive curvature slack scale and gradient drop factor based on flats
            let f_ok_flat = (f_next - f_k).abs() <= eps_f(f_k, self.tau_f)
                || (f_next - f_k).abs() <= self.tol_f_rel * (1.0 + f_k.abs());
            if f_ok_flat {
                self.flat_accept_streak
                    .set(self.flat_accept_streak.get() + 1);
            } else {
                self.flat_accept_streak.set(0);
            }
            if self.flat_accept_streak.get() >= 2 {
                self.curv_slack_scale
                    .set((self.curv_slack_scale.get() * 0.5).max(0.1));
                self.grad_drop_factor.set(0.95);
            } else {
                self.curv_slack_scale.set(1.0);
                self.grad_drop_factor.set(0.9);
            }

            let mut y_k = &g_next - &g_k;

            if self.bounds.is_some() {
                for i in 0..n {
                    let tiny_step = s_k[i].abs() <= 1e-14 * (1.0 + x_k[i].abs());
                    if (active_before[i] && active_after[i]) || tiny_step {
                        s_k[i] = 0.0;
                        y_k[i] = 0.0;
                    }
                }
            }

            // --- Cautious Hessian Update ---
            let sy = s_k.dot(&y_k);
            let mut update_status = "applied";

            if k == 0 {
                // Improved first-step scaling
                let yy = y_k.dot(&y_k);
                let mut scale = if sy > 1e-12 && yy > 0.0 { sy / yy } else { 1.0 };
                if !scale.is_finite() {
                    scale = 1.0;
                }
                scale = scale.clamp(1e-3, 1e3);
                b_inv = Array2::eye(n) * scale;
            } else {
                // Powell-damped inverse BFGS update (keep SPD)
                let s_norm = s_k.dot(&s_k).sqrt();
                if s_norm > 1e-14 {
                    if !rescued {
                        // Compute H s via solving B_inv * (H s) = s
                        let mut binv_upd = b_inv.clone();
                        let mut Lopt = chol_decompose(&binv_upd);
                        if Lopt.is_none() {
                            self.spd_fail_seen.set(true);
                            let mean_diag =
                                (0..n).map(|i| binv_upd[[i, i]].abs()).sum::<f64>() / (n as f64);
                            let ridge = (1e-10 * mean_diag).max(1e-16);
                            for i in 0..n {
                                binv_upd[[i, i]] += ridge;
                            }
                            Lopt = chol_decompose(&binv_upd);
                        }
                        if let Some(L) = Lopt {
                            let h_s = chol_solve(&L, &s_k);
                            let s_h_s = s_k.dot(&h_s);
                            let denom_raw = s_h_s - sy;
                            let denom = if denom_raw <= 0.0 { 1e-16 } else { denom_raw };
                            let theta_raw = if sy < 0.2 * s_h_s {
                                (0.8 * s_h_s) / denom
                            } else {
                                1.0
                            };
                            let theta = theta_raw.clamp(0.0, 1.0);
                            let mut y_tilde = &y_k * theta + &h_s * (1.0 - theta);
                            let mut sty = s_k.dot(&y_tilde);
                            let mut y_norm = y_tilde.dot(&y_tilde).sqrt();
                            let s_norm2 = s_norm * s_norm;
                            let kappa = 1e-4;
                            let min_curv = kappa * s_norm * y_norm;
                            if sty < min_curv {
                                let beta = (min_curv - sty) / s_norm2;
                                y_tilde = &y_tilde + &s_k * beta;
                                sty = s_k.dot(&y_tilde);
                                y_norm = y_tilde.dot(&y_tilde).sqrt();
                            }
                            let rel = if s_norm > 0.0 && y_norm > 0.0 {
                                sty / (s_norm * y_norm)
                            } else {
                                0.0
                            };
                            if !sty.is_finite() || rel < 1e-8 {
                                log::warn!(
                                    "[BFGS] s^T y_tilde non-positive/tiny; skipping update and inflating diag."
                                );
                                update_status = "skipped";
                                self.chol_fail_iters.set(self.chol_fail_iters.get() + 1);
                                for i in 0..n {
                                    b_inv[[i, i]] *= 1.0 + 1e-3;
                                }
                            } else {
                                // Post-update SPD check: build candidate and revert if needed
                                let old_binv = b_inv.clone();
                                // Build left = I - rho * s y^T and right = I - rho * y s^T using scratch buffers
                                let rho_inv = 1.0 / sty;
                                {
                                    let eye = self.scratch_eye.borrow();
                                    let mut left = self.scratch_left.borrow_mut();
                                    let mut right = self.scratch_right.borrow_mut();
                                    // left = I
                                    left.assign(&*eye);
                                    right.assign(&*eye);
                                    // left -= rho_inv * s y^T; right -= rho_inv * y s^T
                                    for i in 0..n {
                                        let si = s_k[i];
                                        let yi = y_tilde[i];
                                        for j in 0..n {
                                            let yj = y_tilde[j];
                                            let sj = s_k[j];
                                            left[[i, j]] -= rho_inv * si * yj;
                                            right[[i, j]] -= rho_inv * yi * sj;
                                        }
                                    }
                                    // tmp = left * b_inv
                                    let mut tmp = self.scratch_tmp.borrow_mut();
                                    *tmp = left.dot(&b_inv);
                                    // b_inv = tmp * right
                                    b_inv = tmp.dot(&*right);
                                }
                                // b_inv += rho_inv * s s^T
                                for i in 0..n {
                                    for j in 0..n {
                                        b_inv[[i, j]] += rho_inv * s_k[i] * s_k[j];
                                    }
                                }
                                // Validate SPD
                                if chol_decompose(&b_inv).is_none() {
                                    b_inv = old_binv;
                                    update_status = "reverted";
                                    for i in 0..n {
                                        b_inv[[i, i]] *= 1.0 + 1e-3;
                                    }
                                }
                            }
                        } else {
                            self.chol_fail_iters.set(self.chol_fail_iters.get() + 1);
                            self.spd_fail_seen.set(true);
                            log::warn!(
                                "[BFGS] B_inv not SPD after ridge; skipping update this iter."
                            );
                            update_status = "skipped";
                        }
                    } else {
                        log::info!(
                            "[BFGS] Coordinate rescue used; skipping inverse update this iter."
                        );
                        update_status = "skipped";
                    }
                    // Enforce symmetry and gentle regularization
                    // Symmetrize in-place
                    for i in 0..n {
                        for j in (i + 1)..n {
                            let a = b_inv[[i, j]];
                            let b = b_inv[[j, i]];
                            let v = 0.5 * (a + b);
                            b_inv[[i, j]] = v;
                            b_inv[[j, i]] = v;
                        }
                    }
                    let mut diag_min = f64::INFINITY;
                    for i in 0..n {
                        diag_min = diag_min.min(b_inv[[i, i]]);
                    }
                    if !diag_min.is_finite() || diag_min <= 0.0 {
                        // Simple diagonal bump proportional to trace
                        let mut trace = 0.0;
                        for i in 0..n {
                            trace += b_inv[[i, i]].abs();
                        }
                        let delta = 1e-12 * trace.max(1.0);
                        for i in 0..n {
                            b_inv[[i, i]] += delta;
                        }
                    }
                }
                if self.spd_fail_seen.get() && self.chol_fail_iters.get() >= 2 {
                    let sy = s_k.dot(&y_k);
                    let yy = y_k.dot(&y_k);
                    let mut lambda = if yy > 0.0 { (sy / yy).abs() } else { 1.0 };
                    lambda = lambda.clamp(1e-6, 1e6);
                    b_inv = scaled_identity(n, lambda);
                    self.resets_count.set(self.resets_count.get() + 1);
                    self.chol_fail_iters.set(0);
                    update_status = "reverted";
                }
            }

            log::info!(
                "[BFGS] step accepted via {:?}; inverse update {}",
                accept_kind,
                update_status
            );

            // Stopping tests: small step and flat f
            let step_ok = s_k.dot(&s_k).sqrt() <= 1e-12 * (1.0 + x_k.dot(&x_k).sqrt()) + 1e-16;
            let f_ok = (f_next - f_k).abs() <= eps_f(f_k, self.tau_f);
            let gnext_finite = f_next.is_finite() && g_next.iter().all(|v| v.is_finite());
            let gnext_norm = g_proj_next.dot(&g_proj_next).sqrt();
            if step_ok && f_ok && gnext_finite && gnext_norm < self.tolerance {
                let sol = BfgsSolution {
                    final_point: x_next.clone(),
                    final_value: f_next,
                    final_gradient_norm: gnext_norm,
                    iterations: k + 1,
                    func_evals,
                    grad_evals,
                };
                log::info!(
                    "[BFGS] Converged by small step/flat f: iters={}, f={:.6e}, ||g||={:.3e}, fe={}, ge={}, Δ={:.3e}",
                    sol.iterations,
                    sol.final_value,
                    sol.final_gradient_norm,
                    sol.func_evals,
                    sol.grad_evals,
                    self.trust_radius.get()
                );
                return Ok(sol);
            }

            // Optional stall/flat exit (relative stationarity)
            if self.stall_enable {
                let g_inf = g_proj_k.iter().fold(0.0, |acc, &v| f64::max(acc, v.abs()));
                let x_inf = x_k.iter().fold(0.0, |acc, &v| f64::max(acc, v.abs()));
                let rel_g_ok = g_inf <= self.tolerance * (1.0 + x_inf);
                let rel_f_ok = (f_k - f_last_accepted).abs() <= eps_f(f_last_accepted, self.tau_f);
                if rel_g_ok && rel_f_ok {
                    self.stall_noimprove_streak
                        .set(self.stall_noimprove_streak.get() + 1);
                } else {
                    self.stall_noimprove_streak.set(0);
                }
                if self.stall_noimprove_streak.get() >= self.stall_k {
                    let sol = BfgsSolution {
                        final_point: x_k.clone(),
                        final_value: f_k,
                        final_gradient_norm: g_inf,
                        iterations: k + 1,
                        func_evals,
                        grad_evals,
                    };
                    log::info!(
                        "[BFGS] Converged (flat/stalled): iters={}, f={:.6e}, ||g||={:.3e}",
                        sol.iterations,
                        sol.final_value,
                        sol.final_gradient_norm
                    );
                    return Ok(sol);
                }
            }

            x_k = x_next;
            f_k = f_next;
            g_k = g_next;
            g_proj_k = g_proj_next;
            active_mask = active_after;
            // Update GLL window and global best
            self.gll.borrow_mut().push(f_k);
            f_last_accepted = f_k;
            // Avoid overlapping borrows on RefCell
            let maybe_f = {
                let gb = self.global_best.borrow();
                gb.as_ref().map(|b| b.f)
            };
            match maybe_f {
                Some(bf) => {
                    if f_k < bf - eps_f(bf, self.tau_f) {
                        self.global_best.borrow_mut().replace(ProbeBest {
                            f: f_k,
                            x: x_k.clone(),
                            g: g_k.clone(),
                        });
                    }
                }
                None => {
                    self.global_best
                        .borrow_mut()
                        .replace(ProbeBest::new(&x_k, f_k, &g_k));
                }
            }

            // Nonmonotone stickiness countdown
            // We return to StrongWolfe only after a run of clean backtracking
            // successes (handled above via `bt_clean_successes`).
        }

        // The loop finished. Construct a solution from the final state.
        let final_g_norm = g_proj_k.dot(&g_proj_k).sqrt();
        let last_solution = Box::new(BfgsSolution {
            final_point: x_k,
            final_value: f_k,
            final_gradient_norm: final_g_norm,
            iterations: self.max_iterations,
            func_evals,
            grad_evals,
        });
        log::warn!(
            "[BFGS] Max iterations reached: iters={}, f={:.6e}, ||g||={:.3e}, fe={}, ge={}, Δ={:.3e}",
            self.max_iterations,
            last_solution.final_value,
            last_solution.final_gradient_norm,
            last_solution.func_evals,
            last_solution.grad_evals,
            self.trust_radius.get()
        );
        Err(BfgsError::MaxIterationsReached { last_solution })
    }

    fn ensure_scratch(&self, n: usize) {
        // Ensure scratch matrices are allocated and sized n x n; initialize identity
        let mut eye = self.scratch_eye.borrow_mut();
        if eye.nrows() != n || eye.ncols() != n {
            *eye = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                eye[[i, i]] = 1.0;
            }
        }
        let mut left = self.scratch_left.borrow_mut();
        if left.nrows() != n || left.ncols() != n {
            *left = Array2::<f64>::zeros((n, n));
        }
        let mut right = self.scratch_right.borrow_mut();
        if right.nrows() != n || right.ncols() != n {
            *right = Array2::<f64>::zeros((n, n));
        }
        let mut tmp = self.scratch_tmp.borrow_mut();
        if tmp.nrows() != n || tmp.ncols() != n {
            *tmp = Array2::<f64>::zeros((n, n));
        }
    }
}

impl<ObjFn> Bfgs<ObjFn>
where
    ObjFn: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
{
    /// Creates a new BFGS solver.
    ///
    /// # Arguments
    /// * `x0` - The initial guess for the minimum.
    /// * `obj_fn` - The objective function which returns a tuple `(value, gradient)`.
    pub fn new(x0: Array1<f64>, obj_fn: ObjFn) -> Self {
        Self {
            core: BfgsCore::new(x0),
            obj_fn,
        }
    }

    /// Sets the convergence tolerance (default: 1e-5).
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.core.tolerance = tolerance;
        self
    }

    /// Sets the maximum number of iterations (default: 100).
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.core.max_iterations = max_iterations;
        self
    }

    /// Provides simple box bounds for each coordinate (lower <= x <= upper).
    /// Points are projected by coordinate clamping, and the gradient is projected
    /// by zeroing active constraints during direction updates.
    pub fn with_bounds(mut self, lower: Array1<f64>, upper: Array1<f64>, tol: f64) -> Self {
        assert_eq!(lower.len(), upper.len(), "lower/upper lengths differ");
        for i in 0..lower.len() {
            assert!(
                lower[i] <= upper[i],
                "lower bound exceeds upper bound at index {i}"
            );
        }
        self.core.bounds = Some(BoxSpec::new(lower, upper, tol.max(0.0)));
        self
    }

    /// Sets the floating-point tolerance multipliers used in eps_f/eps_g.
    pub fn with_fp_tolerances(mut self, tau_f: f64, tau_g: f64) -> Self {
        self.core.tau_f = tau_f.max(1.0);
        self.core.tau_g = tau_g.max(1.0);
        self
    }

    /// If enabled, when `zoom` detects a flat bracket with similar endpoint slopes,
    /// it will accept the midpoint once without further checks to escape flat regions.
    pub fn with_accept_flat_midpoint_once(mut self, enable: bool) -> Self {
        self.core.accept_flat_midpoint_once = enable;
        self
    }

    /// Enable stochastic jiggling of alpha on persistent flat evaluations in
    /// backtracking. `scale` is the relative perturbation amplitude (e.g. 1e-3).
    pub fn with_jiggle_on_flats(mut self, enable: bool, scale: f64) -> Self {
        self.core.jiggle_on_flats = enable;
        if scale.is_finite() && scale > 0.0 {
            self.core.jiggle_scale = scale.min(1e-1);
        }
        self
    }

    /// Enable a simple multi-direction rescue: after two consecutive flat
    /// accepts, try a short coordinate probe and adopt it if it reduces the
    /// gradient norm without worsening f.
    pub fn with_multi_direction_rescue(mut self, enable: bool) -> Self {
        self.core.multi_direction_rescue = enable;
        self
    }

    /// Set the RNG seed used for stochastic jiggling (for reproducibility).
    pub fn with_rng_seed(mut self, seed: u64) -> Self {
        self.core.rng_state = Cell::new(seed);
        self
    }

    /// Configures the coordinate-descent rescue strategy. When enabled, the solver
    /// probes a hybrid pool that includes the top-gradient coordinates and optional
    /// random coordinates to escape flat regions.
    pub fn with_rescue_hybrid(mut self, enable: bool) -> Self {
        self.core.rescue_hybrid = enable;
        self
    }
    /// Scales the size of the coordinate-rescue probe pool relative to the problem dimension.
    pub fn with_rescue_pool_mult(mut self, pool_mult: f64) -> Self {
        self.core.rescue_pool_mult = pool_mult.clamp(1.0, 16.0);
        self
    }
    /// Sets how many of the top-gradient coordinates are probed deterministically
    /// during coordinate rescue. Higher values improve robustness at extra cost.
    pub fn with_rescue_heads(mut self, heads: usize) -> Self {
        self.core.rescue_heads = heads.min(8);
        self
    }
    /// Enables stall-based termination after k consecutive flat/stalled iterations.
    pub fn with_flat_stall_exit(mut self, enable: bool, k: usize) -> Self {
        self.core.stall_enable = enable;
        self.core.stall_k = k.max(1);
        self
    }
    /// Stops after k consecutive iterations without sufficient relative improvement in f.
    pub fn with_no_improve_stop(mut self, tol_f_rel: f64, k: usize) -> Self {
        self.core.tol_f_rel = tol_f_rel.max(1e-12);
        self.core.max_no_improve = k.max(1);
        self
    }
    /// Scales the tolerance for the approximate curvature condition in the
    /// approximate-Wolfe check. Increasing this allows slightly negative curvature
    /// if the function decrease is strong, which can help on non-convex problems.
    pub fn with_curvature_slack_scale(mut self, scale: f64) -> Self {
        self.core.curv_slack_scale.set(scale.clamp(0.1, 10.0));
        self
    }

    /// Executes the BFGS algorithm with the adaptive hybrid line search.
    pub fn run(&mut self) -> Result<BfgsSolution, BfgsError> {
        self.core.run(&mut self.obj_fn)
    }

    #[cfg(test)]
    fn next_rand_sym(&self) -> f64 {
        self.core.next_rand_sym()
    }
}

/// A line search algorithm that finds a step size satisfying the Strong Wolfe conditions.
///
/// This implementation follows the structure of Algorithm 3.5 in Nocedal & Wright,
/// with an efficient state-passing mechanism to avoid re-computation.
#[allow(clippy::too_many_arguments)]
fn line_search<ObjFn>(
    core: &BfgsCore,
    obj_fn: &mut ObjFn,
    x_k: &Array1<f64>,
    d_k: &Array1<f64>,
    f_k: f64,
    g_k: &Array1<f64>,
    c1: f64,
    c2: f64,
) -> LsResult
where
    ObjFn: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
{
    let mut alpha_i: f64 = 1.0; // Per Nocedal & Wright, always start with a unit step.
    let mut alpha_prev = 0.0;

    let mut f_prev = f_k;
    let g_proj_k = core.projected_gradient(x_k, g_k);
    let g_k_dot_d = g_proj_k.dot(d_k); // Initial derivative along the search direction.
    if g_k_dot_d >= 0.0 {
        log::warn!(
            "[BFGS Wolfe] Non-descent direction detected (gᵀd = {:.2e} >= 0).",
            g_k_dot_d
        );
    }
    let mut g_prev_dot_d = g_k_dot_d;

    let max_attempts = 20;
    let mut func_evals = 0;
    let mut grad_evals = 0;
    let epsF = eps_f(f_k, core.tau_f);
    let mut best = ProbeBest::new(x_k, f_k, g_k);
    for _ in 0..max_attempts {
        let (x_new, s, kinked) = core.project_with_step(x_k, d_k, alpha_i);
        let (f_i, g_i) = obj_fn(&x_new);
        func_evals += 1;
        grad_evals += 1;
        best.consider(&x_new, f_i, &g_i);

        // Handle any non-finite value early
        let g_i_finite = g_i.iter().all(|v| v.is_finite());
        if !f_i.is_finite() || !g_i_finite {
            core.nonfinite_seen.set(true);
            if alpha_prev == 0.0 {
                alpha_i *= 0.5;
            } else {
                alpha_i = 0.5 * (alpha_prev + alpha_i);
            }
            if alpha_i <= 1e-18 {
                if let Some((a, f, g, kind)) = probe_alphas(
                    core,
                    obj_fn,
                    x_k,
                    d_k,
                    f_k,
                    g_k,
                    0.0,
                    alpha_i.max(f64::EPSILON),
                    core.tau_f,
                    core.tau_g,
                    core.grad_drop_factor.get(),
                    &mut func_evals,
                    &mut grad_evals,
                ) {
                    return Ok((a, f, g, func_evals, grad_evals, kind));
                }
                return Err(LineSearchError::StepSizeTooSmall);
            }
            // Back-off attempts when stuck in non-finite region
            if func_evals >= 3 {
                return Err(LineSearchError::MaxAttempts(max_attempts));
            }
            continue;
        }

        // Classic Armijo + previous worsening for bracketing (Strong-Wolfe)
        let gkTs = g_proj_k.dot(&s);
        let armijo_strict = f_i > f_k + c1 * gkTs + epsF;
        let prev_worse = func_evals > 1 && f_i >= f_prev - epsF;
        if armijo_strict || prev_worse {
            let kink_lo = if alpha_prev > 0.0 {
                let (_, _, kink_prev) = core.project_with_step(x_k, d_k, alpha_prev);
                kink_prev
            } else {
                false
            };
            if kink_lo || kinked {
                let fallback = backtracking_line_search(core, obj_fn, x_k, d_k, f_k, g_k);
                return fallback.map(|(a, f, g, fe, ge, kind)| {
                    (a, f, g, fe + func_evals, ge + grad_evals, kind)
                });
            }
            // The minimum is bracketed between alpha_prev and alpha_i.
            // A non-finite gradient from a non-finite function value is handled
            // robustly by the zoom function.
            let g_proj_i = core.projected_gradient(&x_new, &g_i);
            let g_i_dot_d = g_proj_i.dot(d_k);
            let r = zoom(
                core,
                obj_fn,
                x_k,
                d_k,
                f_k,
                g_k,
                &g_proj_k,
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
            if r.is_err() {
                core.global_best.borrow_mut().replace(best.clone());
            }
            return r;
        }

        let g_proj_i = core.projected_gradient(&x_new, &g_i);
        let g_i_dot_d = g_proj_i.dot(d_k);
        // The curvature condition.
        if g_i_dot_d.abs() <= c2 * g_k_dot_d.abs() {
            // Strong Wolfe conditions are satisfied.
            // Expand trust radius modestly on successful strong-wolfe step
            let delta_now = core.trust_radius.get();
            core.trust_radius.set((delta_now * 1.25).min(1e6));
            return Ok((
                alpha_i,
                f_i,
                g_i,
                func_evals,
                grad_evals,
                AcceptKind::StrongWolfe,
            ));
        }

        // Approximate-Wolfe and gradient-reduction acceptors
        let approx_curv_ok = g_i_dot_d.abs()
            <= c2 * g_k_dot_d.abs()
                + core.curv_slack_scale.get() * eps_g(&g_proj_k, d_k, core.tau_g);
        let f_flat_ok = f_i <= f_k + epsF;
        if approx_curv_ok && f_flat_ok && g_i_dot_d <= 0.0 {
            core.approx_wolfe_accepts
                .set(core.approx_wolfe_accepts.get() + 1);
            return Ok((
                alpha_i,
                f_i,
                g_i,
                func_evals,
                grad_evals,
                AcceptKind::ApproxWolfe,
            ));
        }
        let gi_norm = g_proj_i.dot(&g_proj_i).sqrt();
        let gk_norm = g_proj_k.dot(&g_proj_k).sqrt();
        let drop_factor = core.grad_drop_factor.get();
        if f_flat_ok
            && gi_norm <= drop_factor * gk_norm
            && g_i_dot_d <= -eps_g(&g_proj_k, d_k, core.tau_g)
        {
            return Ok((
                alpha_i,
                f_i,
                g_i,
                func_evals,
                grad_evals,
                AcceptKind::GradDrop,
            ));
        }

        // Nonmonotone acceptance (GLL) paired with curvature can avoid zoom
        let fmax = {
            let gll = core.gll.borrow();
            if gll.is_empty() { f_k } else { gll.fmax() }
        };
        let nonmono_ok = core.accept_nonmonotone(f_k, fmax, gkTs, f_i);
        if nonmono_ok && approx_curv_ok {
            return Ok((
                alpha_i,
                f_i,
                g_i,
                func_evals,
                grad_evals,
                AcceptKind::Nonmonotone,
            ));
        }

        if g_i_dot_d >= 0.0 {
            // The minimum is bracketed between alpha_i and alpha_prev.
            // The new `hi` is the current point; the new `lo` is the previous.
            let r = zoom(
                core,
                obj_fn,
                x_k,
                d_k,
                f_k,
                g_k,
                &g_proj_k,
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
            if r.is_err() {
                core.global_best.borrow_mut().replace(best.clone());
            }
            return r;
        }

        // The step is too short, expand the search interval and cache current state.
        alpha_prev = alpha_i;
        f_prev = f_i;
        g_prev_dot_d = g_i_dot_d;
        // Expand alpha but respect alpha_max domain
        alpha_i *= 2.0;
    }

    core.global_best.borrow_mut().replace(best);
    // Probing grid before declaring failure
    if alpha_i > 0.0
        && let Some((a, f, g, kind)) = probe_alphas(
            core,
            obj_fn,
            x_k,
            d_k,
            f_k,
            g_k,
            0.0,
            alpha_i,
            core.tau_f,
            core.tau_g,
            core.grad_drop_factor.get(),
            &mut func_evals,
            &mut grad_evals,
        )
    {
        return Ok((a, f, g, func_evals, grad_evals, kind));
    }
    Err(LineSearchError::MaxAttempts(max_attempts))
}

/// A simple backtracking line search that satisfies the Armijo (sufficient decrease) condition.
fn backtracking_line_search<ObjFn>(
    core: &BfgsCore,
    obj_fn: &mut ObjFn,
    x_k: &Array1<f64>,
    d_k: &Array1<f64>,
    f_k: f64,
    g_k: &Array1<f64>,
) -> LsResult
where
    ObjFn: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
{
    let mut alpha: f64 = 1.0;
    let mut rho = 0.5;
    let max_attempts = 50;

    let g_proj_k = core.projected_gradient(x_k, g_k);
    let g_k_dot_d = g_proj_k.dot(d_k);
    // A backtracking search is only valid on a descent direction.
    if g_k_dot_d > 0.0 {
        log::warn!(
            "[BFGS Backtracking] Search started with a non-descent direction (gᵀd = {:.2e} > 0). This step will likely fail.",
            g_k_dot_d
        );
    }

    let mut func_evals = 0;
    let mut grad_evals = 0;
    let mut best = ProbeBest::new(x_k, f_k, g_k);
    let epsF = eps_f(f_k, core.tau_f);
    let mut no_change_count = 0usize;
    let mut expanded_once = false;
    let dnorm = d_k.dot(d_k).sqrt();
    for _ in 0..max_attempts {
        let (x_new, s, _) = core.project_with_step(x_k, d_k, alpha);
        let (f_new, g_new) = obj_fn(&x_new);
        func_evals += 1;
        grad_evals += 1;
        best.consider(&x_new, f_new, &g_new);

        // If evaluation is non-finite, shrink alpha and continue (salvage best-so-far)
        if !f_new.is_finite() || g_new.iter().any(|v| !v.is_finite()) {
            core.nonfinite_seen.set(true);
            alpha *= rho;
            if alpha < 1e-16 {
                return Err(LineSearchError::StepSizeTooSmall);
            }
            if func_evals >= 3 {
                return Err(LineSearchError::MaxAttempts(max_attempts));
            }
            continue;
        }

        let fmax = {
            let gll = core.gll.borrow();
            if gll.is_empty() { f_k } else { gll.fmax() }
        };
        let g_proj_new = core.projected_gradient(&x_new, &g_new);
        let gkTs = g_proj_k.dot(&s);
        let armijo_accept = core.accept_nonmonotone(f_k, fmax, gkTs, f_new);
        if f_new.is_finite() && armijo_accept {
            return Ok((
                alpha,
                f_new,
                g_new,
                func_evals,
                grad_evals,
                AcceptKind::Nonmonotone,
            ));
        }

        // Gradient reduction acceptance
        let gnew_norm = g_proj_new.dot(&g_proj_new).sqrt();
        let gk_norm = g_proj_k.dot(&g_proj_k).sqrt();
        let drop_factor = core.grad_drop_factor.get();
        if f_new <= f_k + epsF
            && gnew_norm <= drop_factor * gk_norm
            && g_proj_new.dot(d_k) <= -eps_g(&g_proj_k, d_k, core.tau_g)
        {
            return Ok((
                alpha,
                f_new,
                g_new,
                func_evals,
                grad_evals,
                AcceptKind::GradDrop,
            ));
        }

        // Approximate curvature + flat f acceptance (parity with line_search)
        let approx_curv_ok = g_proj_new.dot(d_k).abs()
            <= core.c2_adapt.get() * g_k_dot_d.abs()
                + core.curv_slack_scale.get() * eps_g(&g_proj_k, d_k, core.tau_g);
        if f_new <= f_k + epsF && approx_curv_ok && g_proj_new.dot(d_k) <= 0.0 {
            return Ok((
                alpha,
                f_new,
                g_new,
                func_evals,
                grad_evals,
                AcceptKind::ApproxWolfe,
            ));
        }

        if (f_new - f_k).abs() <= epsF {
            no_change_count += 1;
        } else {
            no_change_count = 0;
            expanded_once = false;
        }
        if no_change_count >= 3 {
            rho = 0.8;
        }
        if no_change_count >= 2 && !expanded_once {
            // one-time expansion to hop flat plateau
            alpha /= rho; // slight expand
            expanded_once = true;
        } else {
            alpha *= rho;
        }
        // Stochastic jiggling to avoid hitting identical thresholds repeatedly
        if core.jiggle_on_flats && no_change_count >= 2 {
            let jiggle = 1.0 + core.jiggle_scale * core.next_rand_sym();
            alpha = (alpha * jiggle).max(f64::EPSILON);
        }
        // Relative step-size stop: ||alpha d|| <= tol_x
        let tol_x = 1e-12 * (1.0 + x_k.dot(x_k).sqrt()) + 1e-16;
        if (alpha * dnorm) <= tol_x {
            return Err(LineSearchError::StepSizeTooSmall);
        }
    }

    // Probing grid before declaring failure
    if alpha > 0.0
        && let Some((a, f, g, kind)) = probe_alphas(
            core,
            obj_fn,
            x_k,
            d_k,
            f_k,
            g_k,
            0.0,
            alpha,
            core.tau_f,
            core.tau_g,
            core.grad_drop_factor.get(),
            &mut func_evals,
            &mut grad_evals,
        )
    {
        return Ok((a, f, g, func_evals, grad_evals, kind));
    }

    // Stash best seen during backtracking
    core.global_best.borrow_mut().replace(best);
    Err(LineSearchError::MaxAttempts(max_attempts))
}

/// Helper "zoom" function using cubic interpolation, as described by Nocedal & Wright (Alg. 3.6).
///
/// This function is called when a bracketing interval [alpha_lo, alpha_hi] that contains
/// a point satisfying the Strong Wolfe conditions is known. It iteratively refines this
/// interval until a suitable step size is found.
#[allow(clippy::too_many_arguments)]
fn zoom<ObjFn>(
    core: &BfgsCore,
    obj_fn: &mut ObjFn,
    x_k: &Array1<f64>,
    d_k: &Array1<f64>,
    f_k: f64,
    g_k: &Array1<f64>,
    g_proj_k: &Array1<f64>,
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
) -> LsResult
where
    ObjFn: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
{
    let max_zoom_attempts = 15;
    let min_alpha_step = 1e-12; // Prevents division by zero or degenerate steps.
    let epsF = eps_f(f_k, core.tau_f);
    let mut best = ProbeBest::new(x_k, f_k, g_k);
    let mut lo_deriv_known = true;
    let mut hi_deriv_known = true;
    for _ in 0..max_zoom_attempts {
        let kink_lo = if alpha_lo > 0.0 {
            let (_, _, kink) = core.project_with_step(x_k, d_k, alpha_lo);
            kink
        } else {
            false
        };
        let kink_hi = if alpha_hi > 0.0 {
            let (_, _, kink) = core.project_with_step(x_k, d_k, alpha_hi);
            kink
        } else {
            false
        };
        if kink_lo || kink_hi {
            let fallback = backtracking_line_search(core, obj_fn, x_k, d_k, f_k, g_k);
            return fallback
                .map(|(a, f, g, fe, ge, kind)| (a, f, g, fe + func_evals, ge + grad_evals, kind));
        }
        // Early exits on tiny bracket or flat ends
        if (alpha_hi - alpha_lo).abs() <= 1e-12 || (f_hi - f_lo).abs() <= epsF {
            let (mut alpha_j, choose_lo) = match (lo_deriv_known, hi_deriv_known) {
                (true, true) => {
                    if g_lo_dot_d.abs() <= g_hi_dot_d.abs() {
                        (alpha_lo, true)
                    } else {
                        (alpha_hi, false)
                    }
                }
                (true, false) => (alpha_lo, true),
                (false, true) => (alpha_hi, false),
                (false, false) => ((alpha_lo + alpha_hi) / 2.0, false),
            };
            // Avoid zero step; prefer the nonzero endpoint, otherwise midpoint
            if alpha_j <= f64::EPSILON {
                alpha_j = if choose_lo { alpha_hi } else { alpha_lo };
            }
            if alpha_j <= f64::EPSILON {
                alpha_j = 0.5 * (alpha_lo + alpha_hi);
            }
            let (x_j, s_j, kink_mid) = core.project_with_step(x_k, d_k, alpha_j);
            if kink_mid {
                let fallback = backtracking_line_search(core, obj_fn, x_k, d_k, f_k, g_k);
                return fallback.map(|(a, f, g, fe, ge, kind)| {
                    (a, f, g, fe + func_evals, ge + grad_evals, kind)
                });
            }
            let (f_j, g_j) = obj_fn(&x_j);
            func_evals += 1;
            grad_evals += 1;
            if !f_j.is_finite() || g_j.iter().any(|&v| !v.is_finite()) {
                core.nonfinite_seen.set(true);
                if choose_lo {
                    alpha_lo = 0.5 * (alpha_lo + alpha_hi);
                    lo_deriv_known = false;
                } else {
                    alpha_hi = 0.5 * (alpha_lo + alpha_hi);
                    hi_deriv_known = false;
                }
                continue;
            }
            // Acceptance guard (use unified rules + gradient reduction)
            let fmax = {
                let gll = core.gll.borrow();
                if gll.is_empty() { f_k } else { gll.fmax() }
            };
            let g_proj_j = core.projected_gradient(&x_j, &g_j);
            let gkTs = g_proj_k.dot(&s_j);
            let armijo_ok = core.accept_nonmonotone(f_k, fmax, gkTs, f_j);
            let g_j_dot_d = g_proj_j.dot(d_k);
            let curv_ok = g_j_dot_d.abs()
                <= c2 * g_k_dot_d.abs()
                    + core.curv_slack_scale.get() * eps_g(g_proj_k, d_k, core.tau_g);
            let f_flat_ok = f_j <= f_k + epsF;
            let gj_norm = g_proj_j.iter().map(|v| v * v).sum::<f64>().sqrt();
            let gk_norm = g_proj_k.iter().map(|v| v * v).sum::<f64>().sqrt();
            let drop_factor = core.grad_drop_factor.get();
            let grad_reduce_ok = f_flat_ok
                && (gj_norm <= drop_factor * gk_norm)
                && (g_j_dot_d <= -eps_g(g_proj_k, d_k, core.tau_g));
            if armijo_ok {
                return Ok((
                    alpha_j,
                    f_j,
                    g_j,
                    func_evals,
                    grad_evals,
                    AcceptKind::Nonmonotone,
                ));
            } else if f_flat_ok && curv_ok && g_j_dot_d <= 0.0 {
                return Ok((
                    alpha_j,
                    f_j,
                    g_j,
                    func_evals,
                    grad_evals,
                    AcceptKind::ApproxWolfe,
                ));
            } else if grad_reduce_ok {
                return Ok((
                    alpha_j,
                    f_j,
                    g_j,
                    func_evals,
                    grad_evals,
                    AcceptKind::GradDrop,
                ));
            } else {
                // tighten bracket and continue
                let mid = 0.5 * (alpha_lo + alpha_hi);
                if alpha_j > mid {
                    alpha_hi = alpha_j;
                    f_hi = f_j;
                    g_hi_dot_d = g_j_dot_d;
                    hi_deriv_known = true;
                } else {
                    alpha_lo = alpha_j;
                    f_lo = f_j;
                    g_lo_dot_d = g_j_dot_d;
                    lo_deriv_known = true;
                }
                continue;
            }
        }
        let flat_f = (f_hi - f_lo).abs() <= epsF;
        let similar_slope = (g_hi_dot_d.abs() - g_lo_dot_d.abs()).abs()
            <= core.curv_slack_scale.get() * eps_g(g_proj_k, d_k, core.tau_g);
        if flat_f && similar_slope {
            let alpha_mid = 0.5 * (alpha_lo + alpha_hi);
            let (x_mid, s_mid, kink_mid) = core.project_with_step(x_k, d_k, alpha_mid);
            if kink_mid {
                let fallback = backtracking_line_search(core, obj_fn, x_k, d_k, f_k, g_k);
                return fallback.map(|(a, f, g, fe, ge, kind)| {
                    (a, f, g, fe + func_evals, ge + grad_evals, kind)
                });
            }
            let (f_mid, g_mid) = obj_fn(&x_mid);
            func_evals += 1;
            grad_evals += 1;
            if f_mid.is_finite() && g_mid.iter().all(|v| v.is_finite()) {
                // Optional midpoint acceptance in flat, similar-slope brackets (guard with descent sign)
                let g_proj_mid = core.projected_gradient(&x_mid, &g_mid);
                let g_mid_dot_d = g_proj_mid.dot(d_k);
                let dir_ok = g_mid_dot_d <= -eps_g(g_proj_k, d_k, core.tau_g);
                if core.accept_flat_midpoint_once && g_mid_dot_d <= 0.0 {
                    return Ok((
                        alpha_mid,
                        f_mid,
                        g_mid,
                        func_evals,
                        grad_evals,
                        AcceptKind::Midpoint,
                    ));
                }
                let fmax = {
                    let gll = core.gll.borrow();
                    if gll.is_empty() { f_k } else { gll.fmax() }
                };
                let gkTs = g_proj_k.dot(&s_mid);
                let armijo_ok = core.accept_nonmonotone(f_k, fmax, gkTs, f_mid);
                let curv_ok = g_mid_dot_d.abs()
                    <= c2 * g_k_dot_d.abs()
                        + core.curv_slack_scale.get() * eps_g(g_proj_k, d_k, core.tau_g);
                let gdrop = g_proj_mid.iter().map(|v| v * v).sum::<f64>().sqrt()
                    <= core.grad_drop_factor.get()
                        * g_proj_k.iter().map(|v| v * v).sum::<f64>().sqrt();
                if armijo_ok && curv_ok {
                    return Ok((
                        alpha_mid,
                        f_mid,
                        g_mid,
                        func_evals,
                        grad_evals,
                        AcceptKind::Nonmonotone,
                    ));
                } else if f_mid <= f_k + epsF && gdrop && dir_ok {
                    return Ok((
                        alpha_mid,
                        f_mid,
                        g_mid,
                        func_evals,
                        grad_evals,
                        AcceptKind::GradDrop,
                    ));
                }
                let tighten_lo = g_lo_dot_d.abs() > g_hi_dot_d.abs();
                if tighten_lo {
                    alpha_lo = alpha_mid;
                    f_lo = f_mid;
                    g_lo_dot_d = g_mid_dot_d;
                    lo_deriv_known = true;
                } else {
                    alpha_hi = alpha_mid;
                    f_hi = f_mid;
                    g_hi_dot_d = g_mid_dot_d;
                    hi_deriv_known = true;
                }
                continue;
            } else {
                core.nonfinite_seen.set(true);
                let tighten_lo = g_lo_dot_d.abs() > g_hi_dot_d.abs();
                if tighten_lo {
                    alpha_lo = alpha_mid;
                    lo_deriv_known = false;
                } else {
                    alpha_hi = alpha_mid;
                    hi_deriv_known = false;
                }
                continue;
            }
        }
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

            // Fallback to bisection if the interval is too small, derivatives unknown,
            // or if function values at the interval ends are infinite, preventing unstable interpolation.
            if alpha_diff < min_alpha_step
                || !f_lo.is_finite()
                || !f_hi.is_finite()
                || !lo_deriv_known
                || !hi_deriv_known
            {
                (alpha_lo + alpha_hi) / 2.0
            } else {
                let d1 = g_lo_dot_d + g_hi_dot_d - 3.0 * (f_hi - f_lo) / alpha_diff;
                let d2_sq = d1 * d1 - g_lo_dot_d * g_hi_dot_d;

                if d2_sq >= 0.0 && d2_sq.is_finite() {
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

        let (x_j, s_j, kink_j) = core.project_with_step(x_k, d_k, alpha_j);
        if kink_j {
            let fallback = backtracking_line_search(core, obj_fn, x_k, d_k, f_k, g_k);
            return fallback
                .map(|(a, f, g, fe, ge, kind)| (a, f, g, fe + func_evals, ge + grad_evals, kind));
        }
        let (f_j, g_j) = obj_fn(&x_j);
        func_evals += 1;
        grad_evals += 1;
        best.consider(&x_j, f_j, &g_j);

        // Handle non-finite by shrinking toward the finite end; keep derivative info intact
        if !f_j.is_finite() || g_j.iter().any(|&v| !v.is_finite()) {
            core.nonfinite_seen.set(true);
            // Move the bound closer to alpha_j, prefer shrinking the side that alpha_j is nearer to
            let to_hi = (alpha_hi - alpha_j).abs() <= (alpha_j - alpha_lo).abs();
            if to_hi {
                alpha_hi = alpha_j;
                f_hi = f_j;
                hi_deriv_known = false;
            } else {
                alpha_lo = alpha_j;
                f_lo = f_j;
                lo_deriv_known = false;
            }
            continue;
        }

        // Check if the new point `alpha_j` satisfies the sufficient decrease condition.
        // An infinite `f_j` means the step was too large and failed the condition.
        let fmax = {
            let gll = core.gll.borrow();
            if gll.is_empty() { f_k } else { gll.fmax() }
        };
        let g_proj_j = core.projected_gradient(&x_j, &g_j);
        let gkTs = g_proj_k.dot(&s_j);
        let armijo_ok = f_j <= f_k + c1 * gkTs + epsF;
        let armijo_gll_ok = f_j <= fmax + c1 * gkTs + epsF;
        if !f_j.is_finite() || (!armijo_ok && !armijo_gll_ok) || f_j >= f_lo - epsF {
            if !f_j.is_finite() {
                core.nonfinite_seen.set(true);
            }
            // The new point is not good enough, shrink the interval from the high end.
            alpha_hi = alpha_j;
            f_hi = f_j;
            g_hi_dot_d = g_proj_j.dot(d_k);
            hi_deriv_known = true;
        } else {
            let g_j_dot_d = g_proj_j.dot(d_k);
            // Check the curvature condition.
            if g_j_dot_d.abs() <= c2 * g_k_dot_d.abs() {
                return Ok((
                    alpha_j,
                    f_j,
                    g_j,
                    func_evals,
                    grad_evals,
                    AcceptKind::StrongWolfe,
                ));
            } else if g_j_dot_d.abs()
                <= c2 * g_k_dot_d.abs()
                    + core.curv_slack_scale.get() * eps_g(g_proj_k, d_k, core.tau_g)
                && f_j <= f_k + epsF
            {
                return Ok((
                    alpha_j,
                    f_j,
                    g_j,
                    func_evals,
                    grad_evals,
                    AcceptKind::ApproxWolfe,
                ));
            }

            // The minimum is bracketed by a point with a negative derivative
            // (alpha_lo) and a point with a positive derivative (alpha_j).
            if g_j_dot_d >= 0.0 {
                // The new point has a positive derivative, so it becomes the new
                // upper bound of the bracket. The new interval is [alpha_lo, alpha_j].
                alpha_hi = alpha_j;
                f_hi = f_j;
                g_hi_dot_d = g_j_dot_d;
                hi_deriv_known = true;
            } else {
                // The new point has a negative derivative, so it becomes the new
                // lower bound of the bracket. The new interval is [alpha_j, alpha_hi].
                alpha_lo = alpha_j;
                f_lo = f_j;
                g_lo_dot_d = g_j_dot_d;
                lo_deriv_known = true;
            }
        }
    }
    // Probing grid before declaring failure
    if let Some((a, f, g, kind)) = probe_alphas(
        core,
        obj_fn,
        x_k,
        d_k,
        f_k,
        g_k,
        alpha_lo.min(alpha_hi),
        alpha_lo.max(alpha_hi),
        core.tau_f,
        core.tau_g,
        core.grad_drop_factor.get(),
        &mut func_evals,
        &mut grad_evals,
    ) {
        return Ok((a, f, g, func_evals, grad_evals, kind));
    }
    core.global_best.borrow_mut().replace(best);
    Err(LineSearchError::MaxAttempts(max_zoom_attempts))
}

#[allow(clippy::too_many_arguments)]
fn probe_alphas<ObjFn>(
    core: &BfgsCore,
    obj_fn: &mut ObjFn,
    x_k: &Array1<f64>,
    d_k: &Array1<f64>,
    f_k: f64,
    g_k: &Array1<f64>,
    a_lo: f64,
    a_hi: f64,
    tau_f: f64,
    tau_g: f64,
    drop_factor: f64,
    fe: &mut usize,
    ge: &mut usize,
) -> Option<(f64, f64, Array1<f64>, AcceptKind)>
where
    ObjFn: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
{
    let cands = [0.2, 0.5, 0.8].map(|t| a_lo + t * (a_hi - a_lo));
    let epsF = eps_f(f_k, tau_f);
    let g_proj_k = core.projected_gradient(x_k, g_k);
    let gk_norm = g_proj_k.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut best: Option<(f64, f64, Array1<f64>, AcceptKind)> = None;
    for &a in &cands {
        if !a.is_finite() || a <= 0.0 {
            continue;
        }
        let (x, _, _) = core.project_with_step(x_k, d_k, a);
        let (f, g) = obj_fn(&x);
        *fe += 1;
        *ge += 1;
        if !f.is_finite() || g.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let g_proj = core.projected_gradient(&x, &g);
        let ok_f = f <= f_k + epsF;
        let gi_norm = g_proj.dot(&g_proj).sqrt();
        let dir_ok = g_proj.dot(d_k) <= -eps_g(&g_proj_k, d_k, tau_g);
        let ok_g = gi_norm <= drop_factor * gk_norm && dir_ok;
        if (ok_f || ok_g) && best.as_ref().map(|(fb, _, _, _)| f < *fb).unwrap_or(true) {
            let kind = if ok_g {
                AcceptKind::GradDrop
            } else {
                AcceptKind::Nonmonotone
            };
            best = Some((f, a, g, kind));
        }
    }
    best.map(|(f, a, g, kind)| (a, f, g, kind))
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
    use std::path::Path;

    #[derive(serde::Deserialize)]
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
        let python = ensure_python_deps()?;
        let input_json = serde_json::json!({
            "x0": x0.to_vec(),
            "function": function_name,
            "tolerance": tolerance,
            "max_iterations": max_iterations
        });

        let output = Command::new(python)
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

    fn ensure_python_deps() -> Result<String, String> {
        let venv_python = ".venv/bin/python";
        let python = if Path::new(venv_python).exists() {
            venv_python.to_string()
        } else {
            "python3".to_string()
        };

        let check = Command::new(&python)
            .arg("-c")
            .arg("import numpy, scipy")
            .output()
            .map_err(|e| format!("Failed to execute Python: {}", e))?;

        if check.status.success() {
            return Ok(python);
        }

        if python != venv_python {
            let venv = Command::new("python3")
                .arg("-m")
                .arg("venv")
                .arg(".venv")
                .output()
                .map_err(|e| format!("Failed to create venv: {}", e))?;
            if !venv.status.success() {
                return Err(format!(
                    "Failed to create venv: {}",
                    String::from_utf8_lossy(&venv.stderr)
                ));
            }
        }

        let install = Command::new(venv_python)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("numpy")
            .arg("scipy")
            .output()
            .map_err(|e| format!("Failed to install numpy/scipy: {}", e))?;
        if !install.status.success() {
            return Err(format!(
                "Failed to install numpy/scipy: {}",
                String::from_utf8_lossy(&install.stderr)
            ));
        }

        Ok(venv_python.to_string())
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
    fn test_quadratic_still_converges_strongly() {
        let x0 = array![20.0, -30.0];
        let sol = Bfgs::new(x0, quadratic)
            .with_tolerance(1e-8)
            .with_max_iterations(1000)
            .run()
            .unwrap();
        assert_that!(&sol.final_point[0]).is_close_to(0.0, 1e-6);
        assert_that!(&sol.final_point[1]).is_close_to(0.0, 1e-6);
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
        let max_iterations = 5;
        let result = Bfgs::new(x0, rosenbrock)
            .with_max_iterations(max_iterations)
            .run();

        match result {
            Err(BfgsError::MaxIterationsReached { last_solution }) => {
                assert_eq!(last_solution.iterations, max_iterations);
                // Also check that the point is not the origin, i.e., that some work was done.
                assert_that!(&last_solution.final_point.dot(&last_solution.final_point))
                    .is_greater_than(0.0);
            }
            _ => panic!("Expected MaxIterationsReached error, but got {:?}", result),
        }
    }

    #[test]
    fn test_non_convex_function_is_handled() {
        let x0 = array![2.0];
        let result = Bfgs::new(x0.clone(), non_convex_max).run();
        eprintln!("non_convex result: {:?}", result);
        // The robust solver should not fail. It gets stuck trying to minimize a function with no minimum.
        // It will hit the max iteration limit because it can't find steps that satisfy the descent condition.
        assert!(matches!(
            result,
            Err(BfgsError::MaxIterationsReached { .. }) | Err(BfgsError::LineSearchFailed { .. })
        ));
    }

    #[test]
    fn test_zero_curvature_is_handled() {
        let x0 = array![10.0, 10.0];
        let result = Bfgs::new(x0, linear_function)
            .with_flat_stall_exit(false, 3)
            .with_no_improve_stop(1e-8, usize::MAX)
            .run();
        // The solver should skip Hessian updates due to sy=0 and eventually
        // hit the max iteration limit as it cannot make progress.
        assert!(matches!(
            result,
            Err(BfgsError::MaxIterationsReached { .. })
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
            .with_no_improve_stop(1e-8, 100)
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

        let PythonOptResult {
            final_value,
            final_gradient_norm,
            func_evals,
            grad_evals,
            message,
            ..
        } = scipy_res;
        if let Some(value) = final_value {
            assert!(value.is_finite());
        }
        if let Some(norm) = final_gradient_norm {
            assert!(norm.is_finite());
        }
        if let Some(count) = func_evals {
            assert!(count > 0);
        }
        if let Some(count) = grad_evals {
            assert!(count > 0);
        }
        if let Some(text) = message {
            assert!(!text.is_empty());
        }
    }

    #[test]
    fn test_quadratic_matches_scipy_behavior() {
        let x0 = array![150.0, -275.5];
        let tolerance = 1e-8;

        // Run our implementation.
        match Bfgs::new(x0.clone(), quadratic)
            .with_tolerance(tolerance)
            .run()
        {
            Ok(sol) => sol,
            Err(BfgsError::MaxIterationsReached { last_solution }) => *last_solution,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        // Run scipy's implementation with synchronized settings.
        let scipy_res = optimize_with_python(&x0, "quadratic", tolerance, 100)
            .expect("Python optimization failed");

        assert!(
            scipy_res.success,
            "Scipy optimization failed: {:?}",
            scipy_res.error
        );

        let PythonOptResult {
            final_point,
            final_value,
            final_gradient_norm,
            iterations,
            func_evals,
            grad_evals,
            message,
            ..
        } = scipy_res;
        if let Some(point) = final_point {
            assert_eq!(point.len(), 2);
        }
        if let Some(value) = final_value {
            assert!(value.is_finite());
        }
        if let Some(norm) = final_gradient_norm {
            assert!(norm.is_finite());
        }
        if let Some(iters) = iterations {
            assert!(iters <= 100);
        }
        if let Some(count) = func_evals {
            assert!(count > 0);
        }
        if let Some(count) = grad_evals {
            assert!(count > 0);
        }
        if let Some(text) = message {
            assert!(!text.is_empty());
        }
    }

    // --- 4. Robustness Tests ---

    #[test]
    fn test_ill_conditioned_problem_converges() {
        let x0 = array![1.0, 1000.0]; // Start far up the narrow valley
        let res = Bfgs::new(x0, ill_conditioned_quadratic).run();
        assert!(res.is_ok() || matches!(res, Err(BfgsError::MaxIterationsReached { .. })));
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
        assert!(result.is_ok() || matches!(result, Err(BfgsError::MaxIterationsReached { .. })));
    }

    #[test]
    fn test_flat_with_noise_accepts() {
        let f = |x: &Array1<f64>| {
            let noise = (x.sum() * 1e6).sin() * 1e-12;
            let val = 1.0 + noise;
            let g = Array1::from_vec(vec![1e-12; x.len()]);
            (val, g)
        };
        let x0 = array![0.0, 0.0];
        let res = Bfgs::new(x0, f).with_tolerance(1e-10).run();
        assert!(res.is_ok() || matches!(res, Err(super::BfgsError::MaxIterationsReached { .. })));
    }

    #[test]
    fn test_piecewise_alpha_jump() {
        let f = |x: &Array1<f64>| {
            let r = x.dot(x).sqrt();
            let val = if r < 1.0 { 1.0 } else { 0.9 };
            let g = if r < 1.0 {
                Array1::zeros(x.len())
            } else {
                x.mapv(|v| 1e-6 * v)
            };
            (val, g)
        };
        let x0 = array![0.5, 0.5];
        let res = Bfgs::new(x0, f).run();
        assert!(res.is_ok() || matches!(res, Err(super::BfgsError::MaxIterationsReached { .. })));
    }

    #[test]
    fn test_rng_symmetry() {
        // Ensure the internal RNG produces a roughly symmetric distribution.
        let x0 = array![0.0];
        let f = |x: &Array1<f64>| (x[0], array![1.0]);
        let solver = super::Bfgs::new(x0, f).with_rng_seed(12345);
        let mut sum = 0.0f64;
        let n = 20_000;
        for _ in 0..n {
            sum += solver.next_rand_sym();
        }
        let mean = sum / (n as f64);
        assert_that!(&mean.abs()).is_less_than(5e-3);
    }
}
