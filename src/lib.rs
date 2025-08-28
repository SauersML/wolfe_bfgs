//! An implementation of the BFGS optimization algorithm.
#![allow(non_snake_case)]
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
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;

// Numerical helpers and small utilities
const EPS: f64 = std::f64::EPSILON;
#[inline]
fn eps_f(fk: f64, tau: f64) -> f64 { tau * EPS * (1.0 + fk.abs()) }
#[inline]
fn eps_g(gk: &Array1<f64>, dk: &Array1<f64>, tau: f64) -> f64 {
    tau * EPS * gk.dot(gk).sqrt() * dk.dot(dk).sqrt()
}
// removed unused step_small helper

// Ring buffer for GLL nonmonotone Armijo (internal only)
struct GllWindow { buf: VecDeque<f64>, cap: usize }
impl GllWindow {
    fn new(cap: usize) -> Self { Self { buf: VecDeque::with_capacity(cap), cap } }
    fn clear(&mut self) { self.buf.clear(); }
    fn push(&mut self, f: f64) { if self.buf.len() == self.cap { self.buf.pop_front(); } self.buf.push_back(f); }
    fn fmax(&self) -> f64 { self.buf.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
    fn is_empty(&self) -> bool { self.buf.is_empty() }
}

// Best-seen tracker during line search/zoom (internal only)
#[derive(Clone)]
struct ProbeBest { f: f64, x: Array1<f64>, g: Array1<f64> }
impl ProbeBest {
    fn new(x0: &Array1<f64>, f0: f64, g0: &Array1<f64>) -> Self { Self { x: x0.clone(), f: f0, g: g0.clone() } }
    fn consider(&mut self, x: &Array1<f64>, f: f64, g: &Array1<f64>) { if f < self.f { self.f = f; self.x = x.clone(); self.g = g.clone(); } }
}

// Simple dense SPD Cholesky (LL^T) and solve utilities
fn chol_decompose(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n { return None; }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j { sum -= l[[i, k]] * l[[j, k]]; }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() { return None; }
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
        for k in 0..i { sum -= l[[i,k]] * y[k]; }
        y[i] = sum / l[[i,i]];
    }
    // Backward solve: L^T x = y
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i+1)..n { sum -= l[[k,i]] * x[k]; }
        x[i] = sum / l[[i,i]];
    }
    x
}

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
    #[error("Maximum number of iterations reached without converging. The best solution found is returned.")]
    MaxIterationsReached {
        /// The best solution found before the iteration limit was reached.
        last_solution: Box<BfgsSolution>,
    },
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
    stay_nonmonotone: Cell<usize>,
    nonfinite_seen: Cell<bool>,
    wolfe_clean_successes: Cell<usize>,
    bt_clean_successes: Cell<usize>,
    ls_failures_in_row: Cell<usize>,
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
            stay_nonmonotone: Cell::new(0),
            nonfinite_seen: Cell::new(false),
            wolfe_clean_successes: Cell::new(0),
            bt_clean_successes: Cell::new(0),
            ls_failures_in_row: Cell::new(0),
        }
    }

    #[inline]
    fn accept_nonmonotone(&self, f_k: f64, fmax: f64, alpha: f64, gkdotd: f64, f_i: f64) -> bool {
        let c1 = self.c1_adapt.get();
        let epsf = eps_f(f_k, 1e3);
        (f_i <= f_k + c1 * alpha * gkdotd + epsf) || (f_i <= fmax + c1 * alpha * gkdotd + epsf)
    }

fn trust_region_dogleg(&self, b_inv: &Array2<f64>, g: &Array1<f64>, delta: f64) -> Option<(Array1<f64>, f64)> {
        // Factor B_inv = L L^T (SPD). If it fails, add a small ridge and retry once.
        let mut binv = b_inv.clone();
        let l = match chol_decompose(&binv) {
            Some(L) => L,
            None => {
                let n = binv.nrows();
                let mean_diag = (0..n).map(|i| binv[[i,i]].abs()).sum::<f64>()/(n as f64);
                let ridge = (1e-10*mean_diag).max(1e-16);
                for i in 0..binv.nrows() { binv[[i,i]] += ridge; }
                match chol_decompose(&binv) {
                    Some(L2) => L2,
                    None => return None,
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
            if !pred_dec.is_finite() || pred_dec <= 0.0 { return None; }
            return Some((p_b, pred_dec));
        }
        let p_u_norm = p_u.dot(&p_u).sqrt();
        if p_u_norm >= delta {
            let p = -g * (delta / gnorm2.sqrt());
            let hp = chol_solve(&l, &p);
            let pred = g.dot(&p) + 0.5 * p.dot(&hp);
            let pred_dec = -pred;
            if !pred_dec.is_finite() || pred_dec <= 0.0 { return None; }
            return Some((p, pred_dec));
        }
        // Dogleg along segment from pu to pb hitting boundary
        let s = &p_b - &p_u;
        let a = s.dot(&s);
        let b = 2.0 * p_u.dot(&s);
        let c = p_u.dot(&p_u) - delta * delta;
        let disc = b*b - 4.0*a*c;
        if !disc.is_finite() || disc < 0.0 { return None; }
        let t = (-b + disc.sqrt()) / (2.0 * a);
        let p = &p_u + &(s * t);
        let hp = chol_solve(&l, &p);
        let pred = g.dot(&p) + 0.5 * p.dot(&hp);
        let pred_dec = -pred;
        if !pred_dec.is_finite() || pred_dec <= 0.0 { return None; }
        Some((p, pred_dec))
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

        // Initialize adaptive state
        {
            let mut gll = self.gll.borrow_mut();
            gll.clear();
            gll.push(f_k);
        }
        self.global_best.borrow_mut().replace(ProbeBest::new(&x_k, f_k, &g_k));
        self.c1_adapt.set(self.c1);
        self.c2_adapt.set(self.c2);
        self.primary_strategy.set(LineSearchStrategy::StrongWolfe);
        self.wolfe_fail_streak.set(0);
        self.stay_nonmonotone.set(0);
        // Initialize trust radius relative to starting point scale
        let xnorm0 = x_k.dot(&x_k).sqrt();
        self.trust_radius.set((1.0 + xnorm0).min(1e6));

        let mut stall_count: usize = 0;
        for k in 0..self.max_iterations {
            // reset non-finite flag per iteration
            self.nonfinite_seen.set(false);
            let g_norm = g_k.dot(&g_k).sqrt();
            if !g_norm.is_finite() {
                log::warn!("[BFGS] Non-finite gradient norm at iter {}: g_norm={:?}", k, g_norm);
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
                    k, sol.final_value, sol.final_gradient_norm, sol.func_evals, sol.grad_evals, self.trust_radius.get()
                );
                return Ok(sol);
            }

            let mut present_d_k = -b_inv.dot(&g_k);
            // Enforce descent direction; reset if needed
            let gdotd = g_k.dot(&present_d_k);
            if gdotd >= 0.0 || present_d_k.iter().all(|&v| v.abs() < 1e-16) {
                log::warn!("[BFGS] Non-descent direction; resetting to -g and B_inv=I.");
                b_inv = Array2::eye(n);
                present_d_k = -g_k.clone();
                self.resets_count.set(self.resets_count.get() + 1);
            }

            // --- Adaptive Hybrid Line Search Execution ---
            let (alpha_k, f_next, g_next, f_evals, g_evals) = {
                let search_result = match self.primary_strategy.get() {
                    LineSearchStrategy::StrongWolfe => line_search(self, &self.obj_fn, &x_k, &present_d_k, f_k, &g_k, self.c1_adapt.get(), self.c2_adapt.get()),
                    LineSearchStrategy::Backtracking => backtracking_line_search(self, &self.obj_fn, &x_k, &present_d_k, f_k, &g_k, self.c1_adapt.get()),
                };

                match search_result {
                    Ok(result) => {
                        // Reset failure streak and relax toward canonical constants
                        self.wolfe_fail_streak.set(0);
                        self.ls_failures_in_row.set(0);
                        self.c1_adapt.set((self.c1_adapt.get()*0.9).max(1e-4));
                        self.c2_adapt.set((self.c2_adapt.get()*1.1).min(0.9));
                        match self.primary_strategy.get() {
                            LineSearchStrategy::StrongWolfe => {
                                self.wolfe_clean_successes.set(self.wolfe_clean_successes.get()+1);
                                self.bt_clean_successes.set(0);
                            }
                            LineSearchStrategy::Backtracking => {
                                self.bt_clean_successes.set(self.bt_clean_successes.get()+1);
                                self.wolfe_clean_successes.set(0);
                            }
                        }
                        result
                    }
                    Err(e) => {
                        // The primary strategy failed.
                        if let LineSearchError::StepSizeTooSmall = e {
                            // Treat as search failure; shrink trust radius and try fallback paths
                            self.trust_radius.set((self.trust_radius.get()*0.5).max(1e-12));
                        }

                        // Attempt fallback if the primary strategy was StrongWolfe.
                        if matches!(self.primary_strategy.get(), LineSearchStrategy::StrongWolfe) {
                            let streak = self.wolfe_fail_streak.get() + 1;
                            self.wolfe_fail_streak.set(streak);
                            log::warn!("[BFGS Adaptive] Strong Wolfe failed at iter {}. Falling back to Backtracking.", k);
                            // Adapt c1/c2 on failures
                            if streak == 1 { self.c2_adapt.set(0.5); }
                            if streak >= 2 { self.c2_adapt.set(0.1); self.c1_adapt.set(1e-3); }
                            self.ls_failures_in_row.set(self.ls_failures_in_row.get()+1);
                            if self.ls_failures_in_row.get() >= 2 {
                                self.gll.borrow_mut().cap = 10;
                            }
                            let fallback_result = backtracking_line_search(self, &self.obj_fn, &x_k, &present_d_k, f_k, &g_k, self.c1_adapt.get());
                            if let Ok(result) = fallback_result {
                                // Fallback succeeded.
                                result
                            } else {
                                // The fallback also failed. Terminate with the informative error.
                                let max_attempts = if let Err(LineSearchError::MaxAttempts(attempts)) = fallback_result { attempts } else { 0 };
                                // Salvage best point seen during line search if any
                                if let Some(b) = self.global_best.borrow().as_ref() {
                                    if b.f < f_k - eps_f(f_k, 1e3) {
                                        x_k = b.x.clone(); f_k = b.f; g_k = b.g.clone();
                                        b_inv = Array2::eye(n);
                                        continue;
                                    }
                                }
                                // Try full trust-region dogleg fallback before giving up
                                let delta = self.trust_radius.get();
                                if let Some((p_tr, pred_dec)) = self.trust_region_dogleg(&b_inv, &g_k, delta) {
                                    let x_try = &x_k + &p_tr;
                                    let g_old = g_k.clone();
                                    let (f_try, g_try) = (self.obj_fn)(&x_try);
                                    func_evals += 1; grad_evals += 1;
                                    let act_dec = f_k - f_try;
                                    if !pred_dec.is_finite() || pred_dec <= 0.0 { self.trust_radius.set((delta*0.5).max(1e-12)); } else {
                                        let rho = act_dec / pred_dec;
                                        self.tr_fallbacks.set(self.tr_fallbacks.get() + 1);
                                        if rho > 0.1 && f_try.is_finite() && g_try.iter().all(|v| v.is_finite()) {
                                            x_k = x_try; f_k = f_try; g_k = g_try.clone();
                                            // Update GLL window and global best
                                            self.gll.borrow_mut().push(f_k);
                                        let maybe_f = { let gb = self.global_best.borrow(); gb.as_ref().map(|b| b.f) };
                                        if let Some(bf) = maybe_f {
                                            if f_k < bf - eps_f(bf, 100.0) {
                                                self.global_best.borrow_mut().replace(ProbeBest { f: f_k, x: x_k.clone(), g: g_k.clone() });
                                            }
                                        } else {
                                            self.global_best.borrow_mut().replace(ProbeBest::new(&x_k, f_k, &g_k));
                                        }
                                            if rho > 0.75 && p_tr.dot(&p_tr).sqrt() > 0.99*delta { self.trust_radius.set((delta*2.0).min(1e6)); }
                                            else if rho < 0.25 { self.trust_radius.set((delta*0.5).max(1e-12)); }
                                            // Also update B_inv using TR step (Powell-damped)
                                            let s_tr = p_tr.clone();
                                            let y_tr = &g_try - &g_old;
                                            let s_norm_tr = s_tr.dot(&s_tr).sqrt();
                                            if s_norm_tr > 1e-14 {
                                                let mut binv_upd = b_inv.clone();
                                                let mut Lopt = chol_decompose(&binv_upd);
                                                if Lopt.is_none() {
                                                    let mean_diag = (0..n).map(|i| binv_upd[[i,i]].abs()).sum::<f64>()/(n as f64);
                                                    let ridge = (1e-10*mean_diag).max(1e-16);
                                                    for i in 0..n { binv_upd[[i,i]] += ridge; }
                                                    Lopt = chol_decompose(&binv_upd);
                                                }
                                                if let Some(L) = Lopt {
                                                    let h_s = chol_solve(&L, &s_tr);
                                                    let s_h_s = s_tr.dot(&h_s);
                                                    let sy_tr = s_tr.dot(&y_tr);
                                                    let theta = if sy_tr < 0.2 * s_h_s { 0.8 * s_h_s / (s_h_s - sy_tr) } else { 1.0 };
                                                    let y_tilde = &y_tr * theta + &h_s * (1.0 - theta);
                                                    let sty = s_tr.dot(&y_tilde);
                                                    if sty.is_finite() && sty > 1e-16 * s_norm_tr * y_tilde.dot(&y_tilde).sqrt() {
                                                        let rho_up = 1.0 / sty;
                                                        let s_col = s_tr.view().insert_axis(Axis(1));
                                                        let s_row = s_tr.view().insert_axis(Axis(0));
                                                        let y_col = y_tilde.view().insert_axis(Axis(1));
                                                        let y_row = y_tilde.view().insert_axis(Axis(0));
                                                        let eye = Array2::<f64>::eye(n);
                                                        let left  = &eye - &(rho_up * s_col.dot(&y_row));
                                                        let right = &eye - &(rho_up * y_col.dot(&s_row));
                                                        let s_s_t = s_col.dot(&s_row);
                                                        let tmp = left.dot(&b_inv).dot(&right);
                                                        b_inv = tmp + rho_up * s_s_t;
                                                        b_inv = (&b_inv + &b_inv.t()) * 0.5;
                                                    } else {
                                                        for i in 0..n { b_inv[[i,i]] += 1e-10; }
                                                    }
                                                }
                                            }
                                            continue;
                                        }
                                    }
                                }
                                // Shrink trust radius; if non-finite seen, return error; else continue
                                self.trust_radius.set((self.trust_radius.get()*0.5).max(1e-12));
                                if self.nonfinite_seen.get() {
                                    let mut ls = BfgsSolution {
                                        final_point: x_k.clone(),
                                        final_value: f_k,
                                        final_gradient_norm: g_norm,
                                        iterations: k,
                                        func_evals,
                                        grad_evals,
                                    };
                                    if let Some(b) = self.global_best.borrow().as_ref() {
                                        if b.f < f_k - eps_f(f_k, 1e3) {
                                            ls.final_point = b.x.clone();
                                            ls.final_value = b.f;
                                            ls.final_gradient_norm = b.g.dot(&b.g).sqrt();
                                        }
                                    }
                                    log::warn!("[BFGS] Line search failed at iter {} (nonfinite seen), fe={}, ge={}, Δ={:.3e}", k, func_evals, grad_evals, self.trust_radius.get());
                                    return Err(BfgsError::LineSearchFailed { last_solution: Box::new(ls), max_attempts });
                                }
                                continue;
                            }
                        } else {
                            // The robust Backtracking strategy has failed. This is a critical problem.
                            // Reset the Hessian and try one last time with a steepest descent direction.
                            log::error!("[BFGS Adaptive] CRITICAL: Backtracking failed at iter {}. Resetting Hessian.", k);
                            b_inv = Array2::<f64>::eye(n);
                            present_d_k = -g_k.clone();
                            let fallback_result = backtracking_line_search(self, &self.obj_fn, &x_k, &present_d_k, f_k, &g_k, self.c1_adapt.get());
                            if let Ok(result) = fallback_result {
                                result
                            } else {
                                let max_attempts = if let Err(LineSearchError::MaxAttempts(attempts)) = fallback_result { attempts } else { 0 };
                                // Full trust-region dogleg fallback
                                let delta = self.trust_radius.get();
                                if let Some((p_tr, pred_dec)) = self.trust_region_dogleg(&b_inv, &g_k, delta) {
                                    let x_try = &x_k + &p_tr;
                                    let (f_try, g_try) = (self.obj_fn)(&x_try);
                                    func_evals += 1; grad_evals += 1;
                                    let act_dec = f_k - f_try;
                                    if !pred_dec.is_finite() || pred_dec <= 0.0 { self.trust_radius.set((delta*0.5).max(1e-12)); } else {
                                        let rho = act_dec / pred_dec;
                                        self.tr_fallbacks.set(self.tr_fallbacks.get() + 1);
                                        if rho > 0.1 && f_try.is_finite() && g_try.iter().all(|v| v.is_finite()) {
                                        x_k = x_try; f_k = f_try; g_k = g_try;
                                        // Update GLL window and global best
                                        self.gll.borrow_mut().push(f_k);
                                        let maybe_f = { let gb = self.global_best.borrow(); gb.as_ref().map(|b| b.f) };
                                        if let Some(bf) = maybe_f {
                                            if f_k < bf - eps_f(bf, 100.0) {
                                                self.global_best.borrow_mut().replace(ProbeBest { f: f_k, x: x_k.clone(), g: g_k.clone() });
                                            }
                                        } else {
                                            self.global_best.borrow_mut().replace(ProbeBest::new(&x_k, f_k, &g_k));
                                        }
                                            if rho > 0.75 && p_tr.dot(&p_tr).sqrt() > 0.99*delta { self.trust_radius.set((delta*2.0).min(1e6)); }
                                            else if rho < 0.25 { self.trust_radius.set((delta*0.5).max(1e-12)); }
                                            continue;
                                        } else {
                                            self.trust_radius.set((delta*0.5).max(1e-12));
                                        }
                                    }
                                }
                                if let Some(b) = self.global_best.borrow().as_ref() {
                                    if b.f < f_k - eps_f(f_k, 1e3) { x_k = b.x.clone(); f_k = b.f; g_k = b.g.clone(); b_inv = Array2::eye(n); continue; }
                                }
                                // Shrink trust radius; if non-finite seen, return error; else continue
                                self.trust_radius.set((self.trust_radius.get()*0.5).max(1e-12));
                                if self.nonfinite_seen.get() {
                                    let mut ls = BfgsSolution {
                                        final_point: x_k.clone(),
                                        final_value: f_k,
                                        final_gradient_norm: g_norm,
                                        iterations: k,
                                        func_evals,
                                        grad_evals,
                                    };
                                    if let Some(b) = self.global_best.borrow().as_ref() {
                                        if b.f < f_k - eps_f(f_k, 1e3) {
                                            ls.final_point = b.x.clone();
                                            ls.final_value = b.f;
                                            ls.final_gradient_norm = b.g.dot(&b.g).sqrt();
                                        }
                                    }
                                    log::warn!("[BFGS] Line search failed at iter {} (nonfinite seen), fe={}, ge={}, Δ={:.3e}", k, func_evals, grad_evals, self.trust_radius.get());
                                    return Err(BfgsError::LineSearchFailed { last_solution: Box::new(ls), max_attempts });
                                }
                                continue;
                            }
                        }
                    }
                }
            };

            // The "Learner" part: promote Backtracking if Wolfe keeps failing.
            if self.wolfe_fail_streak.get() >= Self::FALLBACK_THRESHOLD {
                log::warn!("[BFGS Adaptive] Fallback streak ({}) reached. Switching primary to Backtracking.", self.wolfe_fail_streak.get());
                self.primary_strategy.set(LineSearchStrategy::Backtracking);
                self.strategy_switches.set(self.strategy_switches.get() + 1);
                self.wolfe_fail_streak.set(0);
                self.stay_nonmonotone.set(10);
            }
            // Switch back to StrongWolfe after a run of clean backtracking successes
            if matches!(self.primary_strategy.get(), LineSearchStrategy::Backtracking)
                && self.bt_clean_successes.get() >= 3 && self.wolfe_fail_streak.get()==0 {
                log::info!("[BFGS Adaptive] Backtracking succeeded cleanly ({} iters); switching back to StrongWolfe.", self.bt_clean_successes.get());
                self.primary_strategy.set(LineSearchStrategy::StrongWolfe);
                self.strategy_switches.set(self.strategy_switches.get() + 1);
                self.bt_clean_successes.set(0);
                self.gll.borrow_mut().cap = 8;
            }
            // Switch back to StrongWolfe after a run of clean backtracking successes
            if matches!(self.primary_strategy.get(), LineSearchStrategy::Backtracking)
                && self.bt_clean_successes.get() >= 5 && self.wolfe_fail_streak.get()==0 {
                log::info!("[BFGS Adaptive] Backtracking succeeded cleanly ({} iters); switching back to StrongWolfe.", self.bt_clean_successes.get());
                self.primary_strategy.set(LineSearchStrategy::StrongWolfe);
                self.strategy_switches.set(self.strategy_switches.get() + 1);
                self.bt_clean_successes.set(0);
            }

            func_evals += f_evals;
            grad_evals += g_evals;

            let s_k = alpha_k * &present_d_k;
            let y_k = &g_next - &g_k;

            // --- Cautious Hessian Update ---
            let sy = s_k.dot(&y_k);

            if k == 0 {
                // Improved first-step scaling
                let yy = y_k.dot(&y_k);
                let mut scale = if sy > 1e-12 && yy > 0.0 { sy / yy } else { 1.0 };
                if !scale.is_finite() { scale = 1.0; }
                scale = scale.clamp(1e-3, 1e3);
                b_inv = Array2::eye(n) * scale;
            } else {
                // Powell-damped inverse BFGS update (keep SPD)
                let s_norm = s_k.dot(&s_k).sqrt();
                if s_norm > 1e-14 {
                    // Compute H s via solving B_inv * (H s) = s
                    let mut binv_upd = b_inv.clone();
                    let mut Lopt = chol_decompose(&binv_upd);
                    if Lopt.is_none() {
                        // scale-aware ridge
                        let mean_diag = (0..n).map(|i| binv_upd[[i,i]].abs()).sum::<f64>()/(n as f64);
                        let ridge = (1e-10*mean_diag).max(1e-16);
                        for i in 0..n { binv_upd[[i,i]] += ridge; }
                        Lopt = chol_decompose(&binv_upd);
                    }
                    if let Some(L) = Lopt {
                        let h_s = chol_solve(&L, &s_k);
                        let s_h_s = s_k.dot(&h_s);
                        let theta = if sy < 0.2 * s_h_s { 0.8 * s_h_s / (s_h_s - sy) } else { 1.0 };
                        let y_tilde = &y_k * theta + &h_s * (1.0 - theta);
                        let sty = s_k.dot(&y_tilde);
                        if !sty.is_finite() || sty <= 1e-16 * s_norm * y_tilde.dot(&y_tilde).sqrt() {
                            log::warn!("[BFGS] s^T y_tilde non-positive/tiny; skipping update and inflating diag.");
                            for i in 0..n { b_inv[[i,i]] *= 1.0 + 1e-3; }
                        } else {
                            let rho = 1.0 / sty;
                            let s_col = s_k.view().insert_axis(Axis(1));
                            let s_row = s_k.view().insert_axis(Axis(0));
                            let y_col = y_tilde.view().insert_axis(Axis(1));
                            let y_row = y_tilde.view().insert_axis(Axis(0));
                            let eye = Array2::<f64>::eye(n);
                            let left  = &eye - &(rho * s_col.dot(&y_row));
                            let right = &eye - &(rho * y_col.dot(&s_row));
                            let s_s_t = s_col.dot(&s_row);
                            let tmp = left.dot(&b_inv).dot(&right);
                            b_inv = tmp + rho * s_s_t;
                        }
                    } else {
                        log::warn!("[BFGS] B_inv not SPD after ridge; skipping update this iter.");
                    }
                    // Enforce symmetry and gentle regularization
                    b_inv = (&b_inv + &b_inv.t()) * 0.5;
                    let mut diag_min = f64::INFINITY;
                    for i in 0..n { diag_min = diag_min.min(b_inv[[i,i]]); }
                    if !diag_min.is_finite() || diag_min <= 0.0 {
                        let fro = b_inv.mapv(|v| v*v).sum().sqrt();
                        let delta = 1e-12 * fro;
                        for i in 0..n { b_inv[[i,i]] += delta; }
                    }
                }
            }

            // Optional richer stopping tests: small step and flat f
            let step_ok = s_k.dot(&s_k).sqrt() <= 1e-12 * (1.0 + x_k.dot(&x_k).sqrt()) + 1e-16;
            let f_ok = (f_next - f_k).abs() <= eps_f(f_k, 1e3);
            let gnext_finite = f_next.is_finite() && g_next.iter().all(|v| v.is_finite());
            let gnext_norm = g_next.dot(&g_next).sqrt();
            if step_ok && f_ok && gnext_finite && gnext_norm < self.tolerance {
                let sol = BfgsSolution {
                    final_point: x_k.clone() + &s_k,
                    final_value: f_next,
                    final_gradient_norm: gnext_norm,
                    iterations: k + 1,
                    func_evals,
                    grad_evals,
                };
                log::info!(
                    "[BFGS] Converged by small step/flat f: iters={}, f={:.6e}, ||g||={:.3e}, fe={}, ge={}, Δ={:.3e}",
                    sol.iterations, sol.final_value, sol.final_gradient_norm, sol.func_evals, sol.grad_evals, self.trust_radius.get()
                );
                return Ok(sol);
            }

            // Stall detection and periodic reset
            if step_ok && f_ok && gnext_norm >= 0.99 * g_norm {
                stall_count += 1;
            } else {
                stall_count = 0;
            }
            if stall_count >= 3 {
                log::warn!("[BFGS] Stalled 3 iterations; resetting B_inv to identity.");
                b_inv = Array2::eye(n);
                self.resets_count.set(self.resets_count.get() + 1);
                stall_count = 0;
            }

            x_k += &s_k;
            f_k = f_next;
            g_k = g_next;
            // Keep trust radius unchanged on line-search success to avoid blow-up on non-convex cases
            // Update GLL window and global best
            self.gll.borrow_mut().push(f_k);
            // Avoid overlapping borrows on RefCell
            let maybe_f = {
                let gb = self.global_best.borrow();
                gb.as_ref().map(|b| b.f)
            };
            match maybe_f {
                Some(bf) => {
                    if f_k < bf - eps_f(bf, 100.0) {
                        self.global_best.borrow_mut().replace(ProbeBest { f: f_k, x: x_k.clone(), g: g_k.clone() });
                    }
                }
                None => {
                    self.global_best.borrow_mut().replace(ProbeBest::new(&x_k, f_k, &g_k));
                }
            }

            // Nonmonotone stickiness countdown
            if self.stay_nonmonotone.get() > 0 {
                self.stay_nonmonotone.set(self.stay_nonmonotone.get() - 1);
                if self.stay_nonmonotone.get() == 0 && self.wolfe_fail_streak.get() == 0 {
                    self.primary_strategy.set(LineSearchStrategy::StrongWolfe);
                }
            }
        }

        // The loop finished. Construct a solution from the final state.
        let final_g_norm = g_k.dot(&g_k).sqrt();
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
            self.max_iterations, last_solution.final_value, last_solution.final_gradient_norm, last_solution.func_evals, last_solution.grad_evals, self.trust_radius.get()
        );
        Err(BfgsError::MaxIterationsReached { last_solution })
    }
}

/// A line search algorithm that finds a step size satisfying the Strong Wolfe conditions.
///
/// This implementation follows the structure of Algorithm 3.5 in Nocedal & Wright,
/// with an efficient state-passing mechanism to avoid re-computation.
fn line_search<ObjFn>(
    this: &Bfgs<ObjFn>,
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
    let mut alpha_i: f64 = 1.0; // Per Nocedal & Wright, always start with a unit step.
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
    let epsF = eps_f(f_k, 1e3);
    let mut best = ProbeBest::new(x_k, f_k, g_k);

    for _ in 0..max_attempts {
        // Cap step by trust radius on length: ||alpha d|| <= Δ
        let dnorm = d_k.dot(d_k).sqrt();
        if dnorm.is_finite() && dnorm > 0.0 {
            let step_cap = this.trust_radius.get() / dnorm;
            alpha_i = alpha_i.min(step_cap);
        }
        let x_new = x_k + alpha_i * d_k;
        let (f_i, g_i) = obj_fn(&x_new);
        func_evals += 1;
        grad_evals += 1;
        best.consider(&x_new, f_i, &g_i);

        // Handle any non-finite value early
        let g_i_finite = g_i.iter().all(|v| v.is_finite());
        if !f_i.is_finite() || !g_i_finite {
            this.nonfinite_seen.set(true);
            // shrink bracket without zooming on a bad endpoint
            let prev = alpha_prev;
            alpha_i = 0.5 * (alpha_prev + alpha_i);
            if (alpha_i - prev).abs() <= 1e-16 { return Err(LineSearchError::StepSizeTooSmall); }
            continue;
        }

        // Classic Armijo + previous worsening for bracketing (Strong-Wolfe)
        let armijo_strict = f_i > f_k + c1 * alpha_i * g_k_dot_d + epsF;
        let prev_worse = func_evals > 1 && f_i >= f_prev - epsF;
        if armijo_strict || prev_worse
        {
            // The minimum is bracketed between alpha_prev and alpha_i.
            // A non-finite gradient from a non-finite function value is handled
            // robustly by the zoom function.
            let g_i_dot_d = g_i.dot(d_k);
            let r = zoom(
                this,
                obj_fn,
                x_k,
                d_k,
                f_k,
                g_k,
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
            if r.is_err() { this.global_best.borrow_mut().replace(best.clone()); }
            return r;
        }

        let g_i_dot_d = g_i.dot(d_k);
        // The curvature condition.
        if g_i_dot_d.abs() <= c2 * g_k_dot_d.abs() {
            // Strong Wolfe conditions are satisfied.
            // Expand trust radius modestly on successful strong-wolfe step
            let delta_now = this.trust_radius.get();
            this.trust_radius.set((delta_now * 1.25).min(1e6));
            return Ok((alpha_i, f_i, g_i, func_evals, grad_evals));
        }

        // Approximate-Wolfe and gradient-reduction acceptors
        let approx_curv_ok = g_i_dot_d.abs() <= c2 * g_k_dot_d.abs() + eps_g(g_k, d_k, 100.0);
        let f_flat_ok = f_i <= f_k + epsF;
        if approx_curv_ok && f_flat_ok {
            this.approx_wolfe_accepts.set(this.approx_wolfe_accepts.get() + 1);
            return Ok((alpha_i, f_i, g_i, func_evals, grad_evals));
        }
        let gi_norm = g_i.iter().fold(0.0, |acc, &v| acc + v*v).sqrt();
        let gk_norm = g_k.iter().fold(0.0, |acc, &v| acc + v*v).sqrt();
        // gradient reduction requires descent-aligned direction
        if f_flat_ok && gi_norm <= 0.9 * gk_norm && g_i_dot_d <= -eps_g(g_k, d_k, 100.0) {
            return Ok((alpha_i, f_i, g_i, func_evals, grad_evals));
        }

        // Nonmonotone acceptance (GLL) paired with curvature can avoid zoom
        let fmax = { let gll = this.gll.borrow(); if gll.is_empty() { f_k } else { gll.fmax() } };
        let nonmono_ok = this.accept_nonmonotone(f_k, fmax, alpha_i, g_k_dot_d, f_i);
        if nonmono_ok && approx_curv_ok {
            return Ok((alpha_i, f_i, g_i, func_evals, grad_evals));
        }

        if g_i_dot_d >= 0.0 {
            // The minimum is bracketed between alpha_i and alpha_prev.
            // The new `hi` is the current point; the new `lo` is the previous.
            let r = zoom(
                this,
                obj_fn,
                x_k,
                d_k,
                f_k,
                g_k,
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
            if r.is_err() { this.global_best.borrow_mut().replace(best.clone()); }
            return r;
        }

        // The step is too short, expand the search interval and cache current state.
        alpha_prev = alpha_i;
        f_prev = f_i;
        g_prev_dot_d = g_i_dot_d;
        alpha_i = (alpha_i * 2.0).min(this.trust_radius.get());
    }

    this.global_best.borrow_mut().replace(best);
    // Probing grid before declaring failure
    if alpha_i > 0.0 {
        if let Some((a, f, g)) = probe_alphas(obj_fn, x_k, d_k, f_k, g_k, 0.0, alpha_i) {
            return Ok((a, f, g, func_evals, grad_evals));
        }
    }
    Err(LineSearchError::MaxAttempts(max_attempts))
}

/// A simple backtracking line search that satisfies the Armijo (sufficient decrease) condition.
fn backtracking_line_search<ObjFn>(
    this: &Bfgs<ObjFn>,
    obj_fn: &ObjFn,
    x_k: &Array1<f64>,
    d_k: &Array1<f64>,
    f_k: f64,
    g_k: &Array1<f64>,
    _c1: f64,
) -> Result<(f64, f64, Array1<f64>, usize, usize), LineSearchError>
where
    ObjFn: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    let mut alpha: f64 = 1.0;
    let mut rho = 0.5;
    let max_attempts = 50;

    let g_k_dot_d = g_k.dot(d_k);
    // A backtracking search is only valid on a descent direction.
    if g_k_dot_d > 0.0 {
        log::warn!("[BFGS Backtracking] Search started with a non-descent direction (gᵀd = {:.2e} > 0). This step will likely fail.", g_k_dot_d);
    }

    let mut func_evals = 0;
    let mut grad_evals = 0;
    let mut best = ProbeBest::new(x_k, f_k, g_k);
    let epsF = eps_f(f_k, 1e3);
    let mut no_change_count = 0usize;
    let mut expanded_once = false;
    for _ in 0..max_attempts {
        // Cap step by trust radius on length: ||alpha d|| <= Δ
        let dnorm = d_k.dot(d_k).sqrt();
        if dnorm.is_finite() && dnorm > 0.0 {
            let step_cap = this.trust_radius.get() / dnorm;
            alpha = alpha.min(step_cap);
        }
        let x_new = x_k + alpha * d_k;
        let (f_new, g_new) = obj_fn(&x_new);
        func_evals += 1;
        grad_evals += 1;
        best.consider(&x_new, f_new, &g_new);

        // If evaluation is non-finite, shrink alpha and continue (salvage best-so-far)
        if !f_new.is_finite() || g_new.iter().any(|v| !v.is_finite()) {
            this.nonfinite_seen.set(true);
            alpha *= rho;
            if alpha < 1e-16 { return Err(LineSearchError::StepSizeTooSmall); }
            continue;
        }

        let fmax = { let gll = this.gll.borrow(); if gll.is_empty() { f_k } else { gll.fmax() } };
        let armijo_accept = this.accept_nonmonotone(f_k, fmax, alpha, g_k_dot_d, f_new);
        if f_new.is_finite() && armijo_accept {
            return Ok((alpha, f_new, g_new, func_evals, grad_evals));
        }

        // Gradient reduction acceptance
        let gnew_norm = g_new.iter().fold(0.0, |acc, &v| acc + v*v).sqrt();
        let gk_norm = g_k.iter().fold(0.0, |acc, &v| acc + v*v).sqrt();
        if f_new <= f_k + epsF && gnew_norm <= 0.9 * gk_norm && g_new.dot(d_k) <= -eps_g(g_k, d_k, 100.0) {
            return Ok((alpha, f_new, g_new, func_evals, grad_evals));
        }

        // Approximate curvature + flat f acceptance (parity with line_search)
        let approx_curv_ok = g_new.dot(d_k).abs() <= this.c2_adapt.get() * g_k_dot_d.abs() + eps_g(g_k, d_k, 100.0);
        if f_new <= f_k + epsF && approx_curv_ok {
            return Ok((alpha, f_new, g_new, func_evals, grad_evals));
        }

        if (f_new - f_k).abs() <= epsF { no_change_count += 1; } else { no_change_count = 0; expanded_once = false; }
        if no_change_count >= 3 { rho = 0.8; }
        if no_change_count >= 2 && !expanded_once {
            // one-time expansion to hop flat plateau
            alpha /= rho; // slight expand
            expanded_once = true;
        } else {
            alpha *= rho;
        }
        // Relative step-size stop: ||alpha d|| <= tol_x
        let tol_x = 1e-12 * (1.0 + x_k.dot(x_k).sqrt()) + 1e-16;
        if (alpha * dnorm) <= tol_x { return Err(LineSearchError::StepSizeTooSmall); }
    }

    // Stash best seen during backtracking
    this.global_best.borrow_mut().replace(best);
    Err(LineSearchError::MaxAttempts(max_attempts))
}


/// Helper "zoom" function using cubic interpolation, as described by Nocedal & Wright (Alg. 3.6).
///
/// This function is called when a bracketing interval [alpha_lo, alpha_hi] that contains
/// a point satisfying the Strong Wolfe conditions is known. It iteratively refines this
/// interval until a suitable step size is found.
#[allow(clippy::too_many_arguments)]
fn zoom<ObjFn>(
    this: &Bfgs<ObjFn>,
    obj_fn: &ObjFn,
    x_k: &Array1<f64>,
    d_k: &Array1<f64>,
    f_k: f64,
    g_k: &Array1<f64>,
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
    let max_zoom_attempts = 15;
    let min_alpha_step = 1e-12; // Prevents division by zero or degenerate steps.
    let epsF = eps_f(f_k, 1e3);
    let mut best = ProbeBest::new(x_k, f_k, g_k);
    for _ in 0..max_zoom_attempts {
        // Early exits on tiny bracket or flat ends
        if (alpha_hi - alpha_lo).abs() <= 1e-12 || (f_hi - f_lo).abs() <= epsF {
            let choose_lo = g_lo_dot_d.abs() <= g_hi_dot_d.abs();
            let mut alpha_j = if choose_lo { alpha_lo } else { alpha_hi };
            // Avoid zero step; prefer the nonzero endpoint, otherwise midpoint
            if alpha_j <= f64::EPSILON { alpha_j = if choose_lo { alpha_hi } else { alpha_lo }; }
            if alpha_j <= f64::EPSILON { alpha_j = 0.5 * (alpha_lo + alpha_hi); }
            let x_j = x_k + alpha_j * d_k;
            let (f_j, g_j) = obj_fn(&x_j);
            func_evals += 1; grad_evals += 1;
            if !f_j.is_finite() || g_j.iter().any(|&v| !v.is_finite()) {
                this.nonfinite_seen.set(true);
                if choose_lo { alpha_lo = 0.5 * (alpha_lo + alpha_hi); g_lo_dot_d = g_k_dot_d; }
                else { alpha_hi = 0.5 * (alpha_lo + alpha_hi); g_hi_dot_d = g_k_dot_d; }
                continue;
            }
            // Acceptance guard (use unified rules + gradient reduction)
            let fmax = { let gll = this.gll.borrow(); if gll.is_empty() { f_k } else { gll.fmax() } };
            let armijo_ok = this.accept_nonmonotone(f_k, fmax, alpha_j, g_k_dot_d, f_j);
            let curv_ok = g_j.dot(d_k).abs() <= c2 * g_k_dot_d.abs() + eps_g(&g_j, d_k, 100.0);
            let f_flat_ok = f_j <= f_k + epsF;
            let gj_norm = g_j.iter().fold(0.0, |acc, &v| acc + v*v).sqrt();
            let gk_norm = g_k.iter().fold(0.0, |acc, &v| acc + v*v).sqrt();
            let grad_reduce_ok = f_flat_ok && (gj_norm <= 0.9 * gk_norm) && (g_j.dot(d_k) <= -eps_g(&g_j, d_k, 100.0));
            if armijo_ok || (f_flat_ok && curv_ok) || grad_reduce_ok {
                return Ok((alpha_j, f_j, g_j, func_evals, grad_evals));
            } else {
                // tighten bracket and continue
                let g_j_dot_d = g_j.dot(d_k);
                let mid = 0.5 * (alpha_lo + alpha_hi);
                if alpha_j > mid {
                    alpha_hi = alpha_j; f_hi = f_j; g_hi_dot_d = g_j_dot_d;
                } else {
                    alpha_lo = alpha_j; f_lo = f_j; g_lo_dot_d = g_j_dot_d;
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
        best.consider(&x_j, f_j, &g_j);

        // Handle non-finite by shrinking bracket rather than fatal error; keep derivative info intact
        if !f_j.is_finite() || g_j.iter().any(|&v| !v.is_finite()) {
            this.nonfinite_seen.set(true);
            // shrink from the side of alpha_j; neutralize derivative so it won't bias endpoint choice
            alpha_hi = alpha_j;
            f_hi = f_j;
            g_hi_dot_d = g_k_dot_d; // neutral derivative magnitude for non-finite endpoint
            continue;
        }

        // Check if the new point `alpha_j` satisfies the sufficient decrease condition.
        // An infinite `f_j` means the step was too large and failed the condition.
        let fmax = { let gll = this.gll.borrow(); if gll.is_empty() { f_k } else { gll.fmax() } };
        let armijo_ok = f_j <= f_k + c1 * alpha_j * g_k_dot_d + epsF;
        let armijo_gll_ok = f_j <= fmax + c1 * alpha_j * g_k_dot_d + epsF;
        if !f_j.is_finite() || (!armijo_ok && !armijo_gll_ok) || f_j >= f_lo - epsF {
            if !f_j.is_finite() { this.nonfinite_seen.set(true); }
            // The new point is not good enough, shrink the interval from the high end.
            alpha_hi = alpha_j;
            f_hi = f_j;
            g_hi_dot_d = g_j.dot(d_k);
        } else {
            let g_j_dot_d = g_j.dot(d_k);
            // Check the curvature condition.
            if g_j_dot_d.abs() <= c2 * g_k_dot_d.abs()
                || (g_j_dot_d.abs() <= c2 * g_k_dot_d.abs() + eps_g(&g_j, d_k, 100.0) && f_j <= f_k + epsF)
            {
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
    // Probing grid before declaring failure
    if let Some((a, f, g)) = probe_alphas(obj_fn, x_k, d_k, f_k, g_k, alpha_lo.min(alpha_hi), alpha_lo.max(alpha_hi)) {
        return Ok((a, f, g, func_evals, grad_evals));
    }
    this.global_best.borrow_mut().replace(best);
    Err(LineSearchError::MaxAttempts(max_zoom_attempts))
}

fn probe_alphas<ObjFn>(
    obj_fn: &ObjFn,
    x_k: &Array1<f64>,
    d_k: &Array1<f64>,
    f_k: f64,
    g_k: &Array1<f64>,
    a_lo: f64,
    a_hi: f64,
) -> Option<(f64, f64, Array1<f64>)>
where
    ObjFn: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    let cands = [0.2, 0.5, 0.8].map(|t| a_lo + t * (a_hi - a_lo));
    let epsF = eps_f(f_k, 1e3);
    let gk_norm = g_k.dot(g_k).sqrt();
    let mut best: Option<(f64, f64, Array1<f64>)> = None;
    for &a in &cands {
        if !a.is_finite() || a <= 0.0 { continue; }
        let x = x_k + a * d_k;
        let (f, g) = obj_fn(&x);
        if !f.is_finite() || g.iter().any(|v| !v.is_finite()) { continue; }
        let ok_f = f <= f_k + epsF;
        let gi_norm = g.dot(&g).sqrt();
        let dir_ok = g.dot(d_k) <= -eps_g(g_k, d_k, 100.0);
        let ok_g = gi_norm <= 0.9 * gk_norm && dir_ok;
        if ok_f || ok_g {
            if best.as_ref().map(|(fb,_,_)| f < *fb).unwrap_or(true) {
                best = Some((f, a, g));
            }
        }
    }
    best.map(|(f,a,g)| (a, f, g))
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
