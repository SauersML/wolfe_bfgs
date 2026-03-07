# Rust Optimization Workspace

[![Build Status](https://github.com/SauersML/wolfe_bfgs/actions/workflows/test.yml/badge.svg)](https://github.com/SauersML/wolfe_bfgs/actions)

This repository is a Cargo workspace for two related crates:

- [`opt`](./opt): the full nonlinear optimization crate with BFGS, Newton trust-region, and ARC
- [`wolfe_bfgs`](./wolfe_bfgs): the focused BFGS-only crate

## Why two crates

The repository originally centered on `wolfe_bfgs`, but the implementation grew into a broader optimization library. The split keeps:

- `opt` as the primary crate for the full solver toolkit
- `wolfe_bfgs` as the narrow package for users who only want the BFGS surface

Both crates live in the same workspace and share the same implementation base.

## Repository layout

```text
.
├── opt/
├── wolfe_bfgs/
└── .github/workflows/
```

## Package docs

- [`opt/README.md`](./opt/README.md)
- [`wolfe_bfgs/README.md`](./wolfe_bfgs/README.md)

## Common commands

Validate the workspace:

```bash
cargo test --workspace
```

Publish crates in dependency order:

```bash
cargo publish -p opt
cargo publish -p wolfe_bfgs
```

The publish workflow in [`.github/workflows/publish.yml`](./.github/workflows/publish.yml) follows the same order and waits for crates.io indexing before publishing `wolfe_bfgs`.
