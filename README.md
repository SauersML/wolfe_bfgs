# Optimization Workspace

This repository now contains two Rust crates:

- `opt`: the full nonlinear optimization crate with BFGS, Newton trust-region, and ARC.
- `wolfe_bfgs`: the focused BFGS crate.

Publish each package from the workspace root with:

```bash
cargo publish -p opt
cargo publish -p wolfe_bfgs
```

Or validate the full workspace with:

```bash
cargo test --workspace
```
