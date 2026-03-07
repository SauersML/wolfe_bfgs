# Optimization Workspace

This repository contains two Rust crates:

- `opt`: the full nonlinear optimization crate with BFGS, Newton trust-region, and ARC
- `wolfe_bfgs`: the focused BFGS-only crate

Repository layout:

- `opt/`
- `wolfe_bfgs/`

Common workspace commands:

```bash
cargo test --workspace
cargo publish -p opt
cargo publish -p wolfe_bfgs
```

For package-level documentation, see:

- `opt/README.md`
- `wolfe_bfgs/README.md`
