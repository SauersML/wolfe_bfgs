# opt

[![Build Status](https://github.com/SauersML/opt/actions/workflows/test.yml/badge.svg)](https://github.com/SauersML/opt/actions)

This repository is the Cargo workspace for two related crates:

- [`opt`](./opt): the full nonlinear optimization crate with BFGS, Newton trust-region, ARC, and fixed-point iteration
- [`wolfe_bfgs`](./wolfe_bfgs): the thin BFGS-only crate that reexports the first-order surface from `opt`

`opt` is the primary crate and the canonical repository identity. `wolfe_bfgs` remains in the same workspace as a narrower package for users who only want the BFGS API.

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
