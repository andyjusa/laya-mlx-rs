//! Laya decision service, MLX from Rust.
//!
//!   laya-rs --serve                                   resident HTTP service
//!   laya-rs <state.json> <questions.json> [sub] [dtype]   single case
//!   laya-rs --batch <cases.json> [sub] [dtype]        many cases, one model load
//!   laya-rs --debug <case.json> [sub] [dtype]         batch and encoder dump for diagnosis

mod agent;
mod lang;
mod micro;
mod model;
mod router;
mod server;
mod text;
mod tokenizer;

use mlx_rs::Dtype;
use serde_json::{Map, Value};

fn dtype_from(value: Option<&String>) -> Dtype {
    match value.map(String::as_str).unwrap_or("float32") {
        "float16" => Dtype::Float16,
        _ => Dtype::Float32,
    }
}

fn subfolder_from(value: Option<&String>) -> Option<&str> {
    value
        .map(String::as_str)
        .filter(|item| !item.is_empty() && *item != "-")
}

fn model_spec() -> String {
    std::env::var("LAYA_MODEL").unwrap_or_else(|_| router::BUNDLE_REPO.to_string())
}

fn serve() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let host = std::env::var("LAYA_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("LAYA_PORT")
        .unwrap_or_else(|_| "8765".to_string())
        .parse()?;
    let token = std::env::var("LAYA_TOKEN")
        .ok()
        .filter(|value| !value.trim().is_empty());
    let dtype_name = std::env::var("LAYA_DTYPE").unwrap_or_else(|_| "float16".to_string());
    let dtype = if dtype_name == "float32" {
        Dtype::Float32
    } else {
        Dtype::Float16
    };
    let names: Vec<String> = std::env::var("LAYA_MODELS")
        .unwrap_or_else(|_| "english,multilingual,typed-decisions".to_string())
        .split(',')
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect();

    let inference =
        server::Inference::spawn(model_spec(), names.clone(), dtype, dtype_name.clone())?;
    eprintln!(
        "loaded {} checkpoint(s) in {:.2}s ({dtype_name})",
        names.len(),
        inference.load_seconds
    );
    let state = server::AppState {
        inference,
        backend: "rust-mlx".to_string(),
        token,
    };
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(server::serve(state, &host, port))?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let spec = model_spec();

    if args.first().map(String::as_str) == Some("--serve") {
        return serve();
    }

    if args.first().map(String::as_str) == Some("--micro-matmul") {
        let reps: i32 = args
            .get(1)
            .and_then(|value| value.parse().ok())
            .unwrap_or(100);
        micro::run_matmul(reps)?;
        return Ok(());
    }

    if args.first().map(String::as_str) == Some("--micro") {
        let layers: i32 = args
            .get(1)
            .and_then(|value| value.parse().ok())
            .unwrap_or(28);
        micro::run(layers)?;
        return Ok(());
    }

    if args.first().map(String::as_str) == Some("--micro-persistent-gelu") {
        let layers: i32 = args
            .get(1)
            .and_then(|value| value.parse().ok())
            .unwrap_or(28);
        micro::run_persistent_gelu(layers)?;
        return Ok(());
    }

    if args.first().map(String::as_str) == Some("--micro-compiled") {
        let layers: i32 = args
            .get(1)
            .and_then(|value| value.parse().ok())
            .unwrap_or(28);
        micro::run_compiled(layers)?;
        return Ok(());
    }

    if args.first().map(String::as_str) == Some("--bench") {
        let path = args.get(1).ok_or("--bench needs a case.json path")?;
        let reps: usize = args
            .get(2)
            .and_then(|value| value.parse().ok())
            .unwrap_or(20);
        let subfolder = subfolder_from(args.get(3));
        let dtype = dtype_from(args.get(4));
        let payload: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let case = payload
            .get("cases")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .ok_or("case.json has no cases")?;
        let state = case.get("state").ok_or("case has no state")?;
        let questions: Map<String, Value> = case
            .get("questions")
            .and_then(Value::as_object)
            .ok_or("case has no questions object")?
            .clone();
        let agent = agent::Agent::load(&spec, subfolder, dtype)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&agent.bench(state, &questions, reps)?)?
        );
        return Ok(());
    }

    if args.first().map(String::as_str) == Some("--debug") {
        let path = args.get(1).ok_or("--debug needs a case.json path")?;
        let payload: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let case = payload
            .get("cases")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .ok_or("case.json has no cases")?;
        let state = case.get("state").ok_or("case has no state")?;
        let questions: Map<String, Value> = case
            .get("questions")
            .and_then(Value::as_object)
            .ok_or("case has no questions object")?
            .clone();
        let agent =
            agent::Agent::load(&spec, subfolder_from(args.get(2)), dtype_from(args.get(3)))?;
        println!(
            "{}",
            serde_json::to_string(&agent.debug(state, &questions)?)?
        );
        return Ok(());
    }

    if args.first().map(String::as_str) == Some("--batch") {
        let path = args.get(1).ok_or("--batch needs a cases.json path")?;
        let payload: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let cases = payload
            .get("cases")
            .and_then(Value::as_array)
            .ok_or("cases.json has no cases array")?;
        let agent =
            agent::Agent::load(&spec, subfolder_from(args.get(2)), dtype_from(args.get(3)))?;
        let mut results = Vec::new();
        for case in cases {
            let name = case
                .get("name")
                .and_then(Value::as_str)
                .ok_or("case has no name")?;
            let state = case.get("state").ok_or("case has no state")?;
            let questions: Map<String, Value> = case
                .get("questions")
                .and_then(Value::as_object)
                .ok_or("case has no questions object")?
                .clone();
            let prediction = agent.predict(state, &questions)?;
            let mut entry = Map::new();
            entry.insert("name".into(), Value::String(name.to_string()));
            entry.insert("result".into(), prediction);
            results.push(Value::Object(entry));
        }
        println!("{}", serde_json::to_string_pretty(&Value::Array(results))?);
        return Ok(());
    }

    if args.len() < 2 {
        eprintln!("usage: laya-rs --serve");
        eprintln!("       laya-rs <state.json> <questions.json> [subfolder] [float16|float32]");
        eprintln!("       laya-rs --batch <cases.json> [subfolder] [float16|float32]");
        std::process::exit(2);
    }
    let state: Value = serde_json::from_str(&std::fs::read_to_string(&args[0])?)?;
    let questions: Map<String, Value> = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let agent = agent::Agent::load(&spec, subfolder_from(args.get(2)), dtype_from(args.get(3)))?;
    let result = agent.predict(&state, &questions)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
