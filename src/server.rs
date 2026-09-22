//! Resident HTTP service over the router, matching the Python service's surface.
//!
//! Inference runs on one dedicated OS thread. MLX arrays belong to the stream of the thread that
//! created them, so a checkpoint loaded on a tokio worker is unusable from another worker ("there
//! is no Stream(cpu, 0) in current thread"). Keeping the model and every forward pass on the same
//! thread also keeps MLX calls off the async scheduler.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use mlx_rs::Dtype;
use serde_json::{json, Map, Value};
use tokio::sync::{mpsc, oneshot};

use crate::router::Router;

pub enum Request {
    Predict {
        state: Value,
        questions: Map<String, Value>,
        model: Option<String>,
        lang: Option<String>,
        reply: oneshot::Sender<Result<Value, String>>,
    },
    Route {
        state: Value,
        model: Option<String>,
        lang: Option<String>,
        reply: oneshot::Sender<Result<Value, String>>,
    },
}

/// Handle to the inference thread plus the facts /health reports.
pub struct Inference {
    pub sender: mpsc::Sender<Request>,
    pub loaded: Vec<String>,
    pub load_seconds: f64,
    pub dtype: String,
    pub started: Instant,
}

impl Inference {
    /// Start the inference thread and wait until its checkpoints are resident.
    pub fn spawn(
        spec: String,
        names: Vec<String>,
        dtype: Dtype,
        dtype_name: String,
    ) -> Result<Self, String> {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (sender, mut receiver) = mpsc::channel::<Request>(8);
        std::thread::Builder::new()
            .name("laya-inference".to_string())
            .spawn(move || {
                let started = Instant::now();
                let router = match Router::preload(&spec, &names, dtype) {
                    Ok(router) => router,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                let load_seconds = started.elapsed().as_secs_f64();
                let _ = ready_tx.send(Ok((router.loaded(), load_seconds)));
                while let Some(request) = receiver.blocking_recv() {
                    match request {
                        Request::Predict {
                            state,
                            questions,
                            model,
                            lang,
                            reply,
                        } => {
                            let _ = reply.send(router.predict(
                                &state,
                                &questions,
                                model.as_deref(),
                                lang.as_deref(),
                            ));
                        }
                        Request::Route {
                            state,
                            model,
                            lang,
                            reply,
                        } => {
                            let _ =
                                reply.send(router.route(&state, model.as_deref(), lang.as_deref()));
                        }
                    }
                }
            })
            .map_err(|error| format!("cannot start the inference thread: {error}"))?;

        let (loaded, load_seconds) = ready_rx
            .recv()
            .map_err(|error| format!("inference thread failed to start: {error}"))??;
        Ok(Self {
            sender,
            loaded,
            load_seconds,
            dtype: dtype_name,
            started: Instant::now(),
        })
    }
}

pub struct AppState {
    pub inference: Inference,
    pub backend: String,
    pub token: Option<String>,
}

type Shared = Arc<AppState>;

/// Bearer check. Returns Err with a 401 response when the request must be rejected.
fn check(headers: &HeaderMap, expected: Option<&str>) -> Result<(), Response> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let presented = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok());
    if presented == Some(format!("Bearer {expected}").as_str()) {
        return Ok(());
    }
    Err((
        StatusCode::UNAUTHORIZED,
        Json(json!({"error": "unauthorized"})),
    )
        .into_response())
}

fn bad_request(error: String) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error": error}))).into_response()
}

async fn health(State(state): State<Shared>) -> Response {
    Json(json!({
        "status": "ok",
        "backend": state.backend,
        "dtype": state.inference.dtype,
        "loaded": state.inference.loaded,
        "load_seconds": (state.inference.load_seconds * 100.0).round() / 100.0,
        "uptime_seconds": (state.inference.started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
    }))
    .into_response()
}

async fn route(
    State(state): State<Shared>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Err(response) = check(&headers, state.token.as_deref()) {
        return response;
    }
    let (reply_tx, reply_rx) = oneshot::channel();
    let request = Request::Route {
        state: body.get("state").cloned().unwrap_or(Value::Null),
        model: body
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
        lang: body.get("lang").and_then(Value::as_str).map(str::to_string),
        reply: reply_tx,
    };
    if state.inference.sender.send(request).await.is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "inference thread is gone"})),
        )
            .into_response();
    }
    match reply_rx.await {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(error)) => bad_request(error),
        Err(error) => bad_request(error.to_string()),
    }
}

async fn predict(
    State(state): State<Shared>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Err(response) = check(&headers, state.token.as_deref()) {
        return response;
    }
    let Some(state_value) = body.get("state") else {
        return bad_request("missing state".to_string());
    };
    let Some(questions) = body.get("questions").and_then(Value::as_object) else {
        return bad_request("missing questions object".to_string());
    };
    let (reply_tx, reply_rx) = oneshot::channel();
    let request = Request::Predict {
        state: state_value.clone(),
        questions: questions.clone(),
        model: body
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
        lang: body.get("lang").and_then(Value::as_str).map(str::to_string),
        reply: reply_tx,
    };
    if state.inference.sender.send(request).await.is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "inference thread is gone"})),
        )
            .into_response();
    }
    match reply_rx.await {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(error)) => bad_request(error),
        Err(error) => bad_request(error.to_string()),
    }
}

pub async fn serve(state: AppState, host: &str, port: u16) -> Result<(), String> {
    let shared: Shared = Arc::new(state);
    let app: AxumRouter = AxumRouter::new()
        .route("/health", get(health))
        .route("/route", post(route))
        .route("/predict", post(predict))
        .with_state(shared);
    let address = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&address)
        .await
        .map_err(|error| format!("cannot bind {address}: {error}"))?;
    eprintln!("laya-rs listening on http://{address}");
    axum::serve(listener, app)
        .await
        .map_err(|error| error.to_string())
}
