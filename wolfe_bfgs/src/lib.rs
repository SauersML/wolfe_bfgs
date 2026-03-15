//! Focused BFGS optimization API.

pub use opt::{
    Bfgs, BfgsError, Bounds, BoundsError, ConfigError, FiniteDiffGradient, FirstOrderObjective,
    FirstOrderSample, LineSearchFailureReason, MaxIterations, ObjectiveEvalError, Problem,
    Profile, Solution, Tolerance, ZerothOrderObjective, optimize,
};
