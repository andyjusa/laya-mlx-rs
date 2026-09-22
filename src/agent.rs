//! Checkpoint loading, batching, calibration and result formatting.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_rs::{ops, Array, Dtype};
use serde_json::{Map, Value};

use crate::model::{msg, EncoderConfig, LayaModel, Params, Res};
use crate::text::{
    build_sequence, confidence_from_probs, qtype, temp_bucket, to_internal, Question,
};
use crate::tokenizer::Tokenizer;

/// Where the four pieces of a checkpoint live.
pub struct Checkpoint {
    pub weights: PathBuf,
    pub encoder_config: PathBuf,
    pub agent_config: PathBuf,
    pub tokenizer_dir: PathBuf,
}

/// Cache root for downloaded checkpoints, honouring XDG_CACHE_HOME.
fn cache_root() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"));
    base.join("laya-rs")
}

/// Download one checkpoint file on first use, then reuse the local copy.
///
/// A small hand-rolled fetch beats pulling in a Hub client: the four files a checkpoint needs are
/// plain GETs against `resolve/main`, redirects are followed by default, and offline runs after the
/// first fetch. Set `HF_TOKEN` for gated repositories.
fn ensure_file(repo: &str, relative: &str) -> Res<PathBuf> {
    let target = cache_root().join(repo).join(relative);
    if target.exists() {
        return Ok(target);
    }
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(msg)?;
    }
    let url = format!("https://huggingface.co/{repo}/resolve/main/{relative}");
    let mut request = ureq::get(&url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        request = request.header("Authorization", &format!("Bearer {token}"));
    }
    let response = request
        .call()
        .map_err(|error| format!("failed to download {url}: {error}"))?;
    let mut reader = response.into_body().into_reader();
    let mut file = std::fs::File::create(&target).map_err(msg)?;
    std::io::copy(&mut reader, &mut file).map_err(msg)?;
    Ok(target)
}

fn first_existing(candidates: &[PathBuf]) -> Res<PathBuf> {
    candidates
        .iter()
        .find(|path| path.exists())
        .cloned()
        .ok_or_else(|| {
            format!(
                "none of {:?} exists",
                candidates.iter().take(3).collect::<Vec<_>>()
            )
        })
}

pub fn resolve(spec: &str, subfolder: Option<&str>) -> Res<Checkpoint> {
    let local = Path::new(spec);
    if local.is_dir() {
        let inner = match subfolder {
            Some(name) => local.join(name),
            None => local.to_path_buf(),
        };
        let tokenizer_dir = first_existing(&[inner.join("tokenizer"), local.join("tokenizer")])?;
        return Ok(Checkpoint {
            weights: inner.join("model.safetensors"),
            encoder_config: inner.join("encoder").join("config.json"),
            // The bundle keeps rl_agent_config.json at its root; a subfolder copy wins if present.
            agent_config: first_existing(&[
                inner.join("rl_agent_config.json"),
                local.join("rl_agent_config.json"),
            ])?,
            tokenizer_dir,
        });
    }

    let fetch = |relative: &str| -> Res<PathBuf> { ensure_file(spec, relative) };
    let prefix = match subfolder {
        Some(name) => format!("{name}/"),
        None => String::new(),
    };
    let agent_config = match fetch(&format!("{prefix}rl_agent_config.json")) {
        Ok(path) => path,
        Err(_) => fetch("rl_agent_config.json")?,
    };
    let subfolder_tokenizer = fetch(&format!("{prefix}tokenizer/tokenizer.json"))
        .and_then(|path| fetch(&format!("{prefix}tokenizer/tokenizer_config.json")).map(|_| path));
    let tokenizer_file = match subfolder_tokenizer {
        Ok(path) => path,
        Err(_) => fetch("tokenizer/tokenizer.json")?,
    };
    let tokenizer_dir = tokenizer_file
        .parent()
        .ok_or("tokenizer path has no parent")?
        .to_path_buf();
    Ok(Checkpoint {
        weights: fetch(&format!("{prefix}model.safetensors"))?,
        encoder_config: fetch(&format!("{prefix}encoder/config.json"))?,
        agent_config,
        tokenizer_dir,
    })
}

pub struct Batch {
    pub input_ids: Array,
    pub attention_mask: Vec<Vec<u8>>,
    pub marker_pos: Vec<Vec<i32>>,
    pub marker_mask: Vec<Vec<u8>>,
    pub qtype: Vec<i32>,
    pub input_tokens: i64,
    /// Rows are questions; this is the padded marker count shared by every row.
    pub marker_width: usize,
}

pub struct Agent {
    pub model: LayaModel,
    pub tokenizer: Tokenizer,
    max_len: i32,
    head_max_len: i32,
    temperature: Vec<f32>,
    temperature_by_options: HashMap<String, f32>,
}

impl Agent {
    pub fn load(spec: &str, subfolder: Option<&str>, dtype: Dtype) -> Res<Self> {
        let checkpoint = resolve(spec, subfolder)?;
        let agent_config: Value =
            serde_json::from_str(&std::fs::read_to_string(&checkpoint.agent_config).map_err(msg)?)
                .map_err(msg)?;
        let encoder_raw: Value = serde_json::from_str(
            &std::fs::read_to_string(&checkpoint.encoder_config).map_err(msg)?,
        )
        .map_err(msg)?;
        let encoder = EncoderConfig::from_json(&encoder_raw)?;
        let tokenizer = Tokenizer::from_dir(&checkpoint.tokenizer_dir)?;

        let mut params = Params::load(&checkpoint.weights)?;
        params.cast(dtype)?;
        let model = LayaModel::new(params, encoder, &agent_config)?;

        let max_len = agent_config
            .get("max_len")
            .and_then(Value::as_i64)
            .unwrap_or(512) as i32;
        let head_max_len = agent_config
            .get("head_max_len")
            .and_then(Value::as_i64)
            .unwrap_or(192) as i32;
        if !(4 < head_max_len && head_max_len < max_len) {
            return Err(format!(
                "expected 4 < head_max_len < max_len, got {head_max_len} and {max_len}"
            ));
        }
        let temperature = agent_config
            .get("temperature")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_f64)
                    .map(|value| value as f32)
                    .collect::<Vec<f32>>()
            })
            .unwrap_or_else(|| vec![1.0, 1.0, 1.0]);
        if temperature.len() != 3 || temperature.iter().any(|value| *value <= 0.0) {
            return Err("calibration temperatures must be three positive numbers".into());
        }
        let temperature_by_options = agent_config
            .get("temperature_by_options")
            .and_then(Value::as_object)
            .map(|map| {
                map.iter()
                    .filter_map(|(key, value)| {
                        value.as_f64().map(|number| (key.clone(), number as f32))
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            model,
            tokenizer,
            max_len,
            head_max_len,
            temperature,
            temperature_by_options,
        })
    }

    /// Same contract as the Python port: state plus named typed questions, one forward pass.
    pub fn predict(&self, state: &Value, questions: &Map<String, Value>) -> Res<Value> {
        let mut names: Vec<String> = Vec::new();
        let mut internal: Vec<Question> = Vec::new();
        let mut sequences: Vec<(Vec<u32>, Vec<usize>)> = Vec::new();
        for (name, raw) in questions {
            let question = to_internal(raw)?;
            let sequence = build_sequence(
                &self.tokenizer,
                state,
                &question,
                self.max_len as usize,
                self.head_max_len as usize,
            )?;
            names.push(name.clone());
            sequences.push(sequence);
            internal.push(question);
        }
        if names.is_empty() {
            return Err("no questions were provided".into());
        }

        let batch = self.collate(&sequences, &internal)?;
        let width = batch.marker_width;
        let (logits, act) = self.model.forward(
            &batch.input_ids,
            &batch.attention_mask,
            &batch.marker_pos,
            &batch.marker_mask,
            &batch.qtype,
        )?;
        let logits = logits
            .as_dtype(Dtype::Float32)
            .map_err(msg)?
            .as_slice::<f32>()
            .to_vec();
        let act = ops::softmax_axis(&act.as_dtype(Dtype::Float32).map_err(msg)?, -1, None)
            .map_err(msg)?
            .as_slice::<f32>()
            .to_vec();

        let mut answers = Map::new();
        for (row, name) in names.iter().enumerate() {
            let question = &internal[row];
            let count = question.option_count();
            let kind = question.kind.as_str();
            let fallback = qtype(kind).unwrap_or(0) as usize;
            let scale = self
                .temperature_by_options
                .get(&temp_bucket(kind, count))
                .copied()
                .unwrap_or(self.temperature[fallback])
                .max(1e-3);
            let raw: Vec<f32> = logits[row * width..row * width + count].to_vec();
            let peak = raw.iter().copied().fold(f32::MIN, f32::max);
            let mut probs: Vec<f32> = raw
                .iter()
                .map(|value| ((value - peak) / scale).exp())
                .collect();
            let total: f32 = probs.iter().sum();
            for value in probs.iter_mut() {
                *value /= total;
            }

            let mut answer = Map::new();
            answer.insert("type".into(), Value::String(kind.to_string()));
            answer.insert(
                "confidence".into(),
                Value::from(round4(confidence_from_probs(&probs, count))),
            );
            let mut action = Map::new();
            action.insert("act_probability".into(), Value::from(round4(act[row * 2])));
            let _ = count;
            answer.insert("action".into(), Value::Object(action));

            match kind {
                "choice" => {
                    let labels: Vec<String> = question
                        .criteria
                        .iter()
                        .map(|(label, _)| label.clone())
                        .collect();
                    let best = probs
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(index, _)| index)
                        .unwrap_or(0);
                    answer.insert("choice".into(), Value::String(labels[best].clone()));
                    let mut table = Map::new();
                    for (label, probability) in labels.iter().zip(probs.iter()) {
                        table.insert(label.clone(), Value::from(round4(*probability)));
                    }
                    answer.insert("probabilities".into(), Value::Object(table));
                }
                "score" => {
                    let expected: f32 = probs
                        .iter()
                        .enumerate()
                        .map(|(index, value)| index as f32 * value)
                        .sum();
                    answer.insert("score".into(), Value::from(round4(expected)));
                    let mut legend = Map::new();
                    let mut table = Map::new();
                    for (index, level) in question.levels.iter().enumerate() {
                        legend.insert(index.to_string(), Value::String(level.clone()));
                        table.insert(index.to_string(), Value::from(round4(probs[index])));
                    }
                    answer.insert("legend".into(), Value::Object(legend));
                    answer.insert("probabilities".into(), Value::Object(table));
                }
                _ => {
                    let truth = probs[1];
                    answer.insert("noul".into(), Value::from(round4(truth)));
                    answer.insert(
                        "confidence".into(),
                        Value::from(round4(truth.max(1.0 - truth))),
                    );
                }
            }
            answers.insert(name.clone(), Value::Object(answer));
        }

        let mut usage = Map::new();
        usage.insert("input_tokens".into(), Value::from(batch.input_tokens));
        usage.insert("output_tokens".into(), Value::from(0));
        let mut out = Map::new();
        out.insert("model".into(), Value::from("laya-rl-agent"));
        out.insert("answers".into(), Value::Object(answers));
        out.insert("usage".into(), Value::Object(usage));
        Ok(Value::Object(out))
    }

    /// One row per question: every question carries its own prompt, and different questions can
    /// have different option counts, so the marker axis is padded to the widest row.
    /// Stage timings for one case: tokenisation, batching, encoder, head. Env-gated callers only.
    pub fn bench(&self, state: &Value, questions: &Map<String, Value>, reps: usize) -> Res<Value> {
        let median = |mut values: Vec<f64>| -> f64 {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values[values.len() / 2]
        };
        let mut names: Vec<String> = Vec::new();
        let mut internal: Vec<Question> = Vec::new();
        let mut tokenize_ms = Vec::new();
        for (name, raw) in questions {
            let started = std::time::Instant::now();
            let question = to_internal(raw)?;
            let _ = build_sequence(
                &self.tokenizer,
                state,
                &question,
                self.max_len as usize,
                self.head_max_len as usize,
            )?;
            tokenize_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            names.push(name.clone());
            internal.push(question);
        }
        let sequences: Vec<(Vec<u32>, Vec<usize>)> = internal
            .iter()
            .map(|question| {
                build_sequence(
                    &self.tokenizer,
                    state,
                    question,
                    self.max_len as usize,
                    self.head_max_len as usize,
                )
            })
            .collect::<Result<_, _>>()?;
        let batch = self.collate(&sequences, &internal)?;

        let encode_once = || -> Res<f64> {
            let started = std::time::Instant::now();
            let hidden = self.model.encode(&batch.input_ids, &batch.attention_mask)?;
            hidden.eval().map_err(msg)?;
            Ok(started.elapsed().as_secs_f64() * 1000.0)
        };
        let forward_once = || -> Res<f64> {
            let started = std::time::Instant::now();
            let (logits, act) = self.model.forward(
                &batch.input_ids,
                &batch.attention_mask,
                &batch.marker_pos,
                &batch.marker_mask,
                &batch.qtype,
            )?;
            logits.eval().map_err(msg)?;
            act.eval().map_err(msg)?;
            Ok(started.elapsed().as_secs_f64() * 1000.0)
        };
        for _ in 0..3 {
            encode_once()?;
            forward_once()?;
        }
        // Reset after warm-up so the reported peak is the transient cost of one pass, not the
        // weights that were loaded before it.
        mlx_rs::memory::reset_peak_memory().map_err(msg)?;
        let mut encode_ms = Vec::new();
        let mut forward_ms = Vec::new();
        for _ in 0..reps {
            encode_ms.push(encode_once()?);
            forward_ms.push(forward_once()?);
        }
        let mut out = Map::new();
        // Allocator behaviour is a plausible cause of GPU stalls: if MLX reuses fewer buffers on
        // this path it will allocate during the graph and block. Peak versus active shows that.
        out.insert(
            "active_mb".into(),
            Value::from(mlx_rs::memory::active_memory().map_err(msg)? as f64 / 1_048_576.0),
        );
        out.insert(
            "peak_mb".into(),
            Value::from(mlx_rs::memory::peak_memory().map_err(msg)? as f64 / 1_048_576.0),
        );
        out.insert(
            "sequence_length".into(),
            Value::from(batch.input_ids.shape()[1]),
        );
        out.insert("questions".into(), Value::from(names.len()));
        out.insert("tokenize_ms".into(), Value::from(median(tokenize_ms)));
        out.insert("encode_ms".into(), Value::from(median(encode_ms.clone())));
        out.insert("forward_ms".into(), Value::from(median(forward_ms.clone())));
        let head: Vec<f64> = forward_ms
            .iter()
            .zip(encode_ms.iter())
            .map(|(f, e)| f - e)
            .collect();
        out.insert("head_ms".into(), Value::from(median(head)));
        Ok(Value::Object(out))
    }

    /// Diagnostic: dump the collated batch and the encoder output of row 0 as JSON.
    pub fn debug(&self, state: &Value, questions: &Map<String, Value>) -> Res<Value> {
        let mut internal: Vec<Question> = Vec::new();
        let mut sequences: Vec<(Vec<u32>, Vec<usize>)> = Vec::new();
        for (_name, raw) in questions {
            let question = to_internal(raw)?;
            let sequence = build_sequence(
                &self.tokenizer,
                state,
                &question,
                self.max_len as usize,
                self.head_max_len as usize,
            )?;
            sequences.push(sequence);
            internal.push(question);
        }
        let batch = self.collate(&sequences, &internal)?;
        let hidden = self.model.encode(&batch.input_ids, &batch.attention_mask)?;
        let shape = hidden.shape().to_vec();
        let values = hidden
            .as_dtype(Dtype::Float32)
            .map_err(msg)?
            .as_slice::<f32>()
            .to_vec();
        let mut out = Map::new();
        out.insert(
            "input_ids".into(),
            serde_json::to_value(batch.input_ids.as_slice::<i32>()).map_err(msg)?,
        );
        out.insert(
            "attention_mask".into(),
            serde_json::to_value(&batch.attention_mask).map_err(msg)?,
        );
        out.insert(
            "marker_pos".into(),
            serde_json::to_value(&batch.marker_pos).map_err(msg)?,
        );
        out.insert(
            "marker_mask".into(),
            serde_json::to_value(&batch.marker_mask).map_err(msg)?,
        );
        out.insert(
            "qtype".into(),
            serde_json::to_value(&batch.qtype).map_err(msg)?,
        );
        out.insert(
            "encoder_shape".into(),
            serde_json::to_value(&shape).map_err(msg)?,
        );
        out.insert(
            "encoder_row0".into(),
            serde_json::to_value(&values[..shape[2] as usize]).map_err(msg)?,
        );
        Ok(Value::Object(out))
    }

    fn collate(&self, sequences: &[(Vec<u32>, Vec<usize>)], questions: &[Question]) -> Res<Batch> {
        let batch = sequences.len() as i32;
        let length = sequences
            .iter()
            .map(|(ids, _)| ids.len())
            .max()
            .unwrap_or(0) as i32;
        let markers = sequences
            .iter()
            .map(|(_, marks)| marks.len())
            .max()
            .unwrap_or(0) as i32;
        let mut flat_ids = vec![self.tokenizer.pad_token_id; (batch * length) as usize];
        let mut attention_mask = vec![vec![0u8; length as usize]; batch as usize];
        let mut marker_pos = vec![vec![0i32; markers as usize]; batch as usize];
        let mut marker_mask = vec![vec![0u8; markers as usize]; batch as usize];
        let mut input_tokens = 0i64;
        for (row, (ids, marks)) in sequences.iter().enumerate() {
            for (index, id) in ids.iter().enumerate() {
                flat_ids[row * length as usize + index] = *id as i32;
                attention_mask[row][index] = 1;
            }
            input_tokens += ids.len() as i64;
            for (index, marker) in marks.iter().enumerate() {
                marker_pos[row][index] = *marker as i32;
                marker_mask[row][index] = 1;
            }
        }
        let qtype_values: Vec<i32> = questions
            .iter()
            .map(|question| qtype(&question.kind).unwrap_or(0))
            .collect();
        Ok(Batch {
            input_ids: Array::from_slice(&flat_ids, &[batch, length]),
            attention_mask,
            marker_pos,
            marker_mask,
            qtype: qtype_values,
            input_tokens,
            marker_width: markers as usize,
        })
    }
}

/// Match Python's round(float, 4), which also yields the same printed digits.
pub fn round4(value: f32) -> f64 {
    ((value as f64) * 10_000.0).round() / 10_000.0
}
