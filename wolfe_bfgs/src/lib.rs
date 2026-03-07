//! Focused BFGS optimization API.

pub use opt::{
    Bfgs, BfgsError, BfgsSolution, Bounds, BoundsError, ConfigError, FirstOrderObjective,
    LineSearchFailureReason, MaxIterations, ObjectiveEvalError, Problem, Profile, Tolerance,
    optimize,
};
