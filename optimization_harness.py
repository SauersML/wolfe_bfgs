"""
Python optimization harness for BFGS comparison testing.
Uses scipy.optimize.minimize with method='BFGS' for robust comparison.
"""

import json
import sys
import numpy as np
from scipy.optimize import minimize


def rosenbrock(x):
    """The Rosenbrock function and its gradient."""
    a = 1.0
    b = 100.0
    f = (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    g = np.array([
        -2.0 * (a - x[0]) - 4.0 * b * (x[1] - x[0]**2) * x[0],
        2.0 * b * (x[1] - x[0]**2)
    ])
    return f, g


def quadratic(x):
    """A simple quadratic function: f(x) = x'x"""
    f = np.dot(x, x)
    g = 2.0 * x
    return f, g


def scipy_objective(x, func_name):
    """Wrapper for scipy that returns only the function value."""
    if func_name == "rosenbrock":
        return rosenbrock(x)[0]
    elif func_name == "quadratic":
        return quadratic(x)[0]
    else:
        raise ValueError(f"Unknown function: {func_name}")


def scipy_gradient(x, func_name):
    """Wrapper for scipy that returns only the gradient."""
    if func_name == "rosenbrock":
        return rosenbrock(x)[1]
    elif func_name == "quadratic":
        return quadratic(x)[1]
    else:
        raise ValueError(f"Unknown function: {func_name}")


def optimize_with_scipy(x0, func_name, tolerance=1e-6, max_iterations=100):
    """
    Optimize using scipy's BFGS implementation.
    
    Args:
        x0: Initial point (list)
        func_name: Function name ('rosenbrock' or 'quadratic')
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
    
    Returns:
        Dictionary with optimization results
    """
    x0 = np.array(x0)
    
    # Set up optimization options
    options = {
        'gtol': tolerance,  # Gradient tolerance
        'maxiter': max_iterations,
        'disp': False
    }
    
    # Run optimization
    result = minimize(
        fun=lambda x: scipy_objective(x, func_name),
        x0=x0,
        method='BFGS',
        jac=lambda x: scipy_gradient(x, func_name),
        options=options
    )
    
    # Extract results
    return {
        'success': bool(result.success),
        'final_point': result.x.tolist(),
        'final_value': float(result.fun),
        'final_gradient_norm': float(np.linalg.norm(result.jac)),
        'iterations': int(result.nit),
        'func_evals': int(result.nfev),
        'grad_evals': int(result.njev),
        'message': str(result.message)
    }


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) != 2:
        print("Usage: python optimization_harness.py '<json_input>'", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Parse input JSON
        input_data = json.loads(sys.argv[1])
        
        x0 = input_data['x0']
        func_name = input_data['function']
        tolerance = input_data.get('tolerance', 1e-6)
        max_iterations = input_data.get('max_iterations', 100)
        
        # Run optimization
        result = optimize_with_scipy(x0, func_name, tolerance, max_iterations)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == '__main__':
    main()
