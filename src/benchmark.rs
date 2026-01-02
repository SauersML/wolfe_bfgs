#![cfg(test)]
extern crate test;

use crate::{Bfgs, BfgsSolution};
use ndarray::Array1;
use test::Bencher;

#[bench]
fn test_x_fourth_p_1000(bencher: &mut Bencher) {
    let p = 1_000;
    let x0 = Array1::from_elem(p, 2.0);
    let obj_fn = |x: &Array1<f64>| -> (f64, Array1<f64>) {
        let f: f64 = x.iter().map(|xx| xx.powi(4)).sum();
        let g: Array1<f64> = x.iter().map(|xx| 4.0 * xx.powi(3)).collect();
        (f, g)
    };
    bencher.iter(|| {
        let result = Bfgs::new(x0.clone(), obj_fn).run().unwrap();
        assert!(
            result.final_gradient_norm < 1e-5,
            "Expected small gradient norm, got {}",
            result.final_gradient_norm
        );
    })
}
