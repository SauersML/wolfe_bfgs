# opt

[![Build Status](https://github.com/SauersML/opt/actions/workflows/test.yml/badge.svg)](https://github.com/SauersML/opt/actions)

`opt` is a Rust library for dense nonlinear optimization. This repository publishes two crates:

- [`opt`](./opt): the main crate with BFGS, Newton trust-region, ARC, and fixed-point iteration
- [`wolfe_bfgs`](./wolfe_bfgs): the smaller crate that reexports the BFGS-focused API from `opt`

Use `opt` unless you specifically want the narrower `wolfe_bfgs` surface.

## Repository layout

```text
.
├── Cargo.toml
├── README.md
├── opt
│   ├── Cargo.toml
│   ├── README.md
│   ├── optimization_harness.py
│   └── src
│       └── lib.rs
├── wolfe_bfgs
│   ├── Cargo.toml
│   ├── README.md
│   └── src
│       └── lib.rs
└── .github
    └── workflows
```

## Common commands

Run the full workspace tests:

```bash
cargo test --workspace
```

Run only the main crate tests:

```bash
cargo test -p opt
```

Publish crates in dependency order:

```bash
cargo publish -p opt
cargo publish -p wolfe_bfgs
```

The publish workflow in [`.github/workflows/publish.yml`](./.github/workflows/publish.yml) follows the same order because `wolfe_bfgs` depends on `opt`.
