//! Micro-benchmarks used to compare this implementation against Python MLX on identical work.
//!
//! These touch no checkpoint: they isolate raw MLX behaviour (per-op cost and kernel
//! throughput) from anything in the model code, which is what makes them useful when a
//! wall-clock difference needs to be attributed to the runtime or to this code.

use std::time::Instant;

use mlx_rs::{nn, ops, Array, Dtype};

fn device_line() -> String {
    match mlx_rs::Device::try_default() {
        Ok(device) => {
            let kind = device
                .get_type()
                .map(|value| format!("{value:?}"))
                .unwrap_or_default();
            let index = device
                .get_index()
                .map(|value| value.to_string())
                .unwrap_or_default();
            format!("{kind}({index})")
        }
        Err(error) => format!("unknown ({error})"),
    }
}

fn inputs() -> Result<(Array, Array), String> {
    let weights = mlx_rs::random::normal::<f32>(&[2048, 1024], None, None, None)
        .map_err(super::model::msg)?
        .as_dtype(Dtype::Float16)
        .map_err(super::model::msg)?;
    let transposed = weights.swap_axes(0, 1).map_err(super::model::msg)?;
    Ok((weights, transposed))
}

/// Raw kernel throughput: one matmul shape, repeated, with the transpose built once.
pub fn run_matmul(reps: i32) -> Result<(), String> {
    let (weights, transposed) = inputs()?;
    let a = mlx_rs::random::normal::<f32>(&[288, 1024], None, None, None)
        .map_err(super::model::msg)?
        .as_dtype(Dtype::Float16)
        .map_err(super::model::msg)?;
    let _ = weights;
    let call =
        |a: &Array| -> Result<Array, String> { a.matmul(&transposed).map_err(super::model::msg) };
    for _ in 0..3 {
        call(&a)?.eval().map_err(super::model::msg)?;
    }
    let started = Instant::now();
    for _ in 0..reps {
        call(&a)?.eval().map_err(super::model::msg)?;
    }
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    println!("matmul device={} reps={reps} ms={ms:.3}", device_line());
    Ok(())
}

/// Eager multi-op chain: matmul, split, gelu, multiply, add, repeated.
pub fn run(layers: i32) -> Result<(), String> {
    let (weights, transposed) = inputs()?;
    let _ = weights;
    let start = mlx_rs::random::normal::<f32>(&[3, 96, 1024], None, None, None)
        .map_err(super::model::msg)?
        .as_dtype(Dtype::Float16)
        .map_err(super::model::msg)?;
    let chain = |a: &Array| -> Result<Array, String> {
        let projected = a.matmul(&transposed).map_err(super::model::msg)?;
        let parts = ops::split_equal(&projected, 2, -1).map_err(super::model::msg)?;
        let gated = ops::multiply(&nn::gelu(&parts[0]).map_err(super::model::msg)?, &parts[1])
            .map_err(super::model::msg)?;
        ops::add(a, &gated).map_err(super::model::msg)
    };
    let mut a = start.clone();
    for _ in 0..2 {
        for _ in 0..layers {
            a = chain(&a)?;
        }
        a.eval().map_err(super::model::msg)?;
    }
    let started = Instant::now();
    for _ in 0..layers {
        a = chain(&a)?;
    }
    a.eval().map_err(super::model::msg)?;
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    println!(
        "eager device={} layers={layers} ops={} ms={ms:.3}",
        device_line(),
        layers * 6
    );
    Ok(())
}

/// The same chain with one shapeless-compiled GELU reused across every layer.
pub fn run_persistent_gelu(layers: i32) -> Result<(), String> {
    use mlx_rs::transforms::compile::compile as mlx_compile;

    let (weights, transposed) = inputs()?;
    let _ = weights;
    let start = mlx_rs::random::normal::<f32>(&[3, 96, 1024], None, None, None)
        .map_err(super::model::msg)?
        .as_dtype(Dtype::Float16)
        .map_err(super::model::msg)?;
    let scalar = |value: f32| -> Result<Array, String> {
        let out = Array::from_slice(&[value], &[])
            .as_dtype(Dtype::Float16)
            .map_err(super::model::msg)?;
        out.eval().map_err(super::model::msg)?;
        Ok(out)
    };
    let one = scalar(1.0)?;
    let two = scalar(2.0)?;
    let root_two = scalar(2f32.sqrt())?;
    let gelu_body = |args: &[Array]| -> Result<Vec<Array>, mlx_rs::error::Exception> {
        let x = &args[0];
        Ok(vec![x
            .multiply(&args[1] + ops::erf(&(x / &args[3]))?)?
            .divide(&args[2])?])
    };
    let mut gelu = mlx_compile(gelu_body, true);
    let mut chain = |a: &Array| -> Result<Array, mlx_rs::error::Exception> {
        let projected = a.matmul(&transposed)?;
        let parts = ops::split_equal(&projected, 2, -1)?;
        let activated =
            gelu(&[parts[0].clone(), one.clone(), two.clone(), root_two.clone()])?.remove(0);
        let gated = ops::multiply(&activated, &parts[1])?;
        ops::add(a, &gated)
    };
    let mut a = start;
    for _ in 0..2 {
        for _ in 0..layers {
            a = chain(&a).map_err(super::model::msg)?;
        }
        a.eval().map_err(super::model::msg)?;
    }
    let started = Instant::now();
    for _ in 0..layers {
        a = chain(&a).map_err(super::model::msg)?;
    }
    a.eval().map_err(super::model::msg)?;
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    println!(
        "persistent-gelu device={} layers={layers} ms={ms:.3}",
        device_line()
    );
    Ok(())
}

/// The same chain, traced once by MLX compile instead of dispatched op by op.
pub fn run_compiled(layers: i32) -> Result<(), String> {
    use mlx_rs::transforms::compile::compile as mlx_compile;

    let (weights, transposed) = inputs()?;
    let _ = weights;
    let start = mlx_rs::random::normal::<f32>(&[3, 96, 1024], None, None, None)
        .map_err(super::model::msg)?
        .as_dtype(Dtype::Float16)
        .map_err(super::model::msg)?;
    let body = move |given: &[Array]| -> Result<Vec<Array>, mlx_rs::error::Exception> {
        let mut a = given[0].clone();
        for _ in 0..layers {
            let projected = a.matmul(&transposed)?;
            let parts = ops::split_equal(&projected, 2, -1)?;
            let gated = ops::multiply(&nn::gelu(&parts[0])?, &parts[1])?;
            a = ops::add(&a, &gated)?;
        }
        Ok(vec![a])
    };
    let mut compiled = mlx_compile(body, None);
    for _ in 0..2 {
        let out = compiled(&[start.clone()]).map_err(super::model::msg)?;
        out[0].eval().map_err(super::model::msg)?;
    }
    let started = Instant::now();
    let out = compiled(&[start.clone()]).map_err(super::model::msg)?;
    out[0].eval().map_err(super::model::msg)?;
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    println!(
        "compiled device={} layers={layers} ops={} ms={ms:.3}",
        device_line(),
        layers * 6
    );
    Ok(())
}
