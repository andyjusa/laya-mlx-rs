//! ModernBERT encoder plus Laya's decision head, on MLX from Rust.
//!
//! Ported from the upstream Apache-2.0 reference (NandhaKishorM/laya) and from this repository's
//! Python MLX port. Parameter keys are the checkpoint's own safetensors names.
//!
//! Three details carried over from the Python port, each one measured rather than assumed:
//!   * layer norm accumulates in f32 - in f16 the variance over 1024 dims overflows fp16 range
//!     and rsqrt(inf) zeroes the whole tensor (every answer becomes uniform);
//!   * the fused attention kernel takes a boolean keep-mask, not an additive f32/f16 one
//!     (f16 turns the -1e9 sentinel into -inf and then NaN);
//!   * RoPE is computed in f32 and cast back.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use mlx_rs::error::Exception;
use mlx_rs::fast::{self, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::transforms::compile::compile as mlx_compile;
use mlx_rs::{nn, ops, Array, Dtype};
use mlx_rs::{with_stream, Stream};
use serde_json::Value;

pub type Res<T> = Result<T, String>;

pub fn msg<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

/// Shape tracing for diagnosing broadcast failures; enabled with LAYA_DEBUG_SHAPES=1.
fn trace(label: &str, array: &Array) {
    if std::env::var_os("LAYA_DEBUG_SHAPES").is_some() {
        eprintln!("  [shape] {label}: {:?}", array.shape());
    }
}

pub struct Params {
    map: HashMap<String, Array>,
}

impl Params {
    pub fn load(path: &Path) -> Res<Self> {
        let map = Array::load_safetensors(path).map_err(msg)?;
        Ok(Self { map })
    }

    /// Read-only access for helpers that take the parameter map directly.
    pub fn map(&self) -> &HashMap<String, Array> {
        &self.map
    }

    /// Store every matrix weight in transposed layout for the matmuls in the forward pass.
    ///
    /// The original is replaced rather than kept, so resident memory is unchanged. This is done
    /// once at load: doing it per call cost one MLX op per linear layer plus a non-contiguous
    /// operand for every matmul.
    pub fn prepare_linear_weights(&mut self) -> Res<()> {
        let keys: Vec<String> = self.map.keys().cloned().collect();
        for key in keys {
            let lookup_only =
                key.ends_with("tok_embeddings.weight") || key.ends_with("type_emb.weight");
            let value = self.map.get(&key).ok_or_else(|| format!("missing {key}"))?;
            if lookup_only || value.ndim() != 2 {
                continue;
            }
            let transposed = value.swap_axes(0, 1).map_err(msg)?;
            transposed.eval().map_err(msg)?;
            self.map.insert(key, transposed);
        }
        Ok(())
    }

    pub fn cast(&mut self, dtype: Dtype) -> Res<()> {
        if dtype == Dtype::Float32 {
            for value in self.map.values_mut() {
                *value = value.as_dtype(Dtype::Float32).map_err(msg)?;
            }
        }
        Ok(())
    }

    pub fn get(&self, key: &str) -> Res<&Array> {
        self.map
            .get(key)
            .ok_or_else(|| format!("checkpoint is missing tensor {key}"))
    }
}

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub hidden_size: i32,
    pub heads: i32,
    pub layers: i32,
    pub intermediate: i32,
    pub vocab: i32,
    pub eps: f32,
    pub sliding_window: i32,
    pub layer_types: Vec<String>,
    pub head_dim: i32,
    pub rope_theta_full: f32,
    pub rope_theta_sliding: f32,
}

impl EncoderConfig {
    pub fn from_json(raw: &Value) -> Res<Self> {
        let int = |key: &str| -> Res<i32> {
            raw.get(key)
                .and_then(Value::as_i64)
                .map(|value| value as i32)
                .ok_or_else(|| format!("encoder config has no integer {key}"))
        };
        let hidden_size = int("hidden_size")?;
        let heads = int("num_attention_heads")?;
        if hidden_size % heads != 0 {
            return Err("hidden_size is not divisible by num_attention_heads".into());
        }
        let layer_types: Vec<String> = raw
            .get("layer_types")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .collect()
            })
            .ok_or("encoder config has no layer_types")?;
        let layers = int("num_hidden_layers")?;
        if layer_types.len() as i32 != layers {
            return Err("layer_types length does not match num_hidden_layers".into());
        }
        let local_attention = raw
            .get("local_attention")
            .and_then(Value::as_i64)
            .unwrap_or(128) as i32;
        let theta = |kind: &str| -> f32 {
            raw.get("rope_parameters")
                .and_then(|rope| rope.get(kind))
                .and_then(|entry| entry.get("rope_theta"))
                .and_then(Value::as_f64)
                .map(|value| value as f32)
                .unwrap_or(10000.0)
        };
        Ok(Self {
            hidden_size,
            heads,
            layers,
            intermediate: int("intermediate_size")?,
            vocab: int("vocab_size")?,
            eps: raw.get("norm_eps").and_then(Value::as_f64).unwrap_or(1e-5) as f32,
            sliding_window: local_attention / 2,
            layer_types,
            head_dim: hidden_size / heads,
            rope_theta_full: theta("full_attention"),
            rope_theta_sliding: theta("sliding_attention"),
        })
    }
}

/// Weights are stored transposed (see Params::prepare_linear_weights), so no per-call
/// swap_axes: each one cost an extra MLX op and handed the kernel a non-contiguous operand.
fn linear(x: &Array, weight: &Array, bias: Option<&Array>) -> Res<Array> {
    let out = x.matmul(weight).map_err(msg)?;
    match bias {
        Some(bias) => ops::add(&out, bias).map_err(msg),
        None => Ok(out),
    }
}

/// Layer normalisation through MLX's fused kernel: one op where the explicit mean/variance form
/// costs nine, and MLX accumulates half precision in float32 internally, which fp16 depends on.
fn layer_norm(x: &Array, weight: &Array, bias: Option<&Array>, eps: f32) -> Res<Array> {
    fast::layer_norm(x, weight, bias, eps).map_err(msg)
}

/// Apply rotary embeddings with MLX's fused kernel. The base differs by layer type in the
/// checkpoint config, so it is passed per call; frequencies are derived inside the kernel, which
/// also keeps the traced graph independent of the sequence length.
fn apply_rope(x: &Array, theta: f32, head_dim: i32, _seq: i32) -> Res<Array> {
    fast::rope(x, head_dim, false, theta, 1.0, 0, None).map_err(msg)
}

/// Boolean keep-masks, built on the CPU: True where a query may attend to a key.
fn build_masks(attention_mask: &[Vec<u8>], seq: i32, window: i32) -> Res<(Array, Array)> {
    let batch = attention_mask.len() as i32;
    let mut full = vec![false; (batch * seq * seq) as usize];
    let mut sliding = vec![false; (batch * seq * seq) as usize];
    for row in 0..batch as usize {
        for query in 0..seq as usize {
            for key in 0..seq as usize {
                let keep = attention_mask[row][key] != 0;
                let offset = (row as i32 * seq * seq + query as i32 * seq + key as i32) as usize;
                full[offset] = keep;
                sliding[offset] = keep && (query as i32 - key as i32).abs() <= window;
            }
        }
    }
    Ok((
        Array::from_slice(&full, &[batch, 1, seq, seq]),
        Array::from_slice(&sliding, &[batch, 1, seq, seq]),
    ))
}

type CompiledGelu = Box<dyn for<'a> FnMut(&'a [Array]) -> Result<Vec<Array>, Exception>>;

struct Gelu {
    compiled: CompiledGelu,
    one: Array,
    two: Array,
    root_two: Array,
}

impl Gelu {
    fn new(dtype: Dtype) -> Res<Self> {
        let scalar = |value: f32| -> Res<Array> {
            let out = Array::from_slice(&[value], &[])
                .as_dtype(dtype)
                .map_err(msg)?;
            out.eval().map_err(msg)?;
            Ok(out)
        };
        let body = |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let x = &args[0];
            Ok(vec![x
                .multiply(&args[1] + ops::erf(&(x / &args[3]))?)?
                .divide(&args[2])?])
        };
        Ok(Self {
            compiled: Box::new(mlx_compile(body, true)),
            one: scalar(1.0)?,
            two: scalar(2.0)?,
            root_two: scalar(2f32.sqrt())?,
        })
    }

    fn apply(&mut self, x: &Array) -> Result<Array, Exception> {
        Ok((self.compiled)(&[
            x.clone(),
            self.one.clone(),
            self.two.clone(),
            self.root_two.clone(),
        ])?
        .remove(0))
    }
}

pub struct LayaModel {
    params: Params,
    gelu: RefCell<Gelu>,
    pub cfg: EncoderConfig,
    pub head_layers: i32,
    pub fused: bool,
    /// Evaluate the encoder every N layers instead of once at the end; 0 disables it.
    eval_every: usize,
    /// Pins one request to one MLX stream; measured performance-neutral, but keeps the
    /// resident model's GPU work on its dedicated inference thread.
    stream: Stream,
}

/// The encoder forward pass as a plain function of (model, inputs), so tracing experiments can
/// reuse it. Masks stay outside because building them is host-side work.
fn encode_core(
    model: &LayaModel,
    input_ids: &Array,
    full_mask: &Array,
    sliding_mask: &Array,
) -> Result<Array, Exception> {
    let cfg = &model.cfg;
    let params = model.params.map();
    let eval_every = model.eval_every;
    let custom = Exception::custom;
    let mut hidden = params
        .get("encoder.embeddings.tok_embeddings.weight")
        .ok_or_else(|| Exception::custom("missing token embeddings"))?
        .take_axis(input_ids, 0)?;
    hidden = layer_norm(
        &hidden,
        params
            .get("encoder.embeddings.norm.weight")
            .ok_or_else(|| Exception::custom("missing embedding norm"))?,
        None,
        cfg.eps,
    )
    .map_err(custom)?;

    for index in 0..cfg.layers {
        let kind = cfg.layer_types[index as usize].as_str();
        let prefix = format!("encoder.layers.{index}");
        let weight = |suffix: &str| -> Result<&Array, Exception> {
            params
                .get(&format!("{prefix}.{suffix}"))
                .ok_or_else(|| Exception::custom(format!("missing {prefix}.{suffix}")))
        };
        let pre = if index == 0 {
            hidden.clone()
        } else {
            layer_norm(&hidden, weight("attn_norm.weight")?, None, cfg.eps).map_err(custom)?
        };
        let mask = if kind == "sliding_attention" {
            sliding_mask
        } else {
            full_mask
        };
        let theta = if kind == "sliding_attention" {
            cfg.rope_theta_sliding
        } else {
            cfg.rope_theta_full
        };
        let attended = model
            .encoder_attention(&format!("{prefix}.attn"), &pre, theta, mask)
            .map_err(custom)?;
        hidden = ops::add(&hidden, &attended)?;

        let pre = layer_norm(&hidden, weight("mlp_norm.weight")?, None, cfg.eps).map_err(custom)?;
        let projected = linear(&pre, weight("mlp.Wi.weight")?, None).map_err(custom)?;
        let pieces = ops::split_equal(&projected, 2, -1)?;
        let gated = ops::multiply(&model.gelu.borrow_mut().apply(&pieces[0])?, &pieces[1])?;
        let mlp = linear(&gated, weight("mlp.Wo.weight")?, None).map_err(custom)?;
        hidden = ops::add(&hidden, &mlp)?;
        // A lazily built graph keeps every layer's intermediates alive until the final evaluate,
        // which measured a 2.1 GB transient peak against Python MLX's 0.3 GB for the same work and
        // correlates with GPU stalls. Evaluating every N layers lets MLX free and reuse buffers.
        if eval_every > 0 && (index + 1) % eval_every as i32 == 0 {
            hidden.eval()?;
        }
    }
    layer_norm(
        &hidden,
        params
            .get("encoder.final_norm.weight")
            .ok_or_else(|| Exception::custom("missing final norm"))?,
        None,
        cfg.eps,
    )
    .map_err(custom)
}

impl LayaModel {
    pub fn new(params: Params, cfg: EncoderConfig, agent_cfg: &Value) -> Res<Self> {
        let head_layers = agent_cfg
            .get("head_layers")
            .and_then(Value::as_i64)
            .ok_or("agent config has no head_layers")? as i32;
        let eval_every = std::env::var("LAYA_EVAL_EVERY")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        let dtype = params
            .get("encoder.embeddings.tok_embeddings.weight")?
            .dtype();
        let mut model = Self {
            params,
            gelu: RefCell::new(Gelu::new(dtype)?),
            cfg,
            head_layers,
            fused: true,
            eval_every,
            stream: Stream::new(),
        };
        model.validate()?;
        model.params.prepare_linear_weights()?;
        Ok(model)
    }

    fn validate(&self) -> Res<()> {
        let mut required = vec![
            "encoder.embeddings.tok_embeddings.weight".to_string(),
            "encoder.embeddings.norm.weight".to_string(),
            "encoder.final_norm.weight".to_string(),
            "type_emb.weight".to_string(),
            "scorer.0.weight".to_string(),
            "scorer.1.weight".to_string(),
            "scorer.3.weight".to_string(),
            "act_head.0.weight".to_string(),
            "act_head.2.weight".to_string(),
        ];
        for index in 0..self.cfg.layers {
            let base = format!("encoder.layers.{index}");
            required.extend([
                format!("{base}.attn.Wqkv.weight"),
                format!("{base}.attn.Wo.weight"),
                format!("{base}.mlp.Wi.weight"),
                format!("{base}.mlp.Wo.weight"),
                format!("{base}.mlp_norm.weight"),
            ]);
            if index > 0 {
                required.push(format!("{base}.attn_norm.weight"));
            }
        }
        for index in 0..self.head_layers {
            let base = format!("head.layers.{index}");
            required.extend([
                format!("{base}.self_attn.in_proj_weight"),
                format!("{base}.self_attn.out_proj.weight"),
                format!("{base}.linear1.weight"),
                format!("{base}.linear2.weight"),
                format!("{base}.norm1.weight"),
                format!("{base}.norm2.weight"),
            ]);
        }
        let missing: Vec<&String> = required
            .iter()
            .filter(|key| self.params.get(key).is_err())
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "checkpoint is missing {} tensor(s): {:?}",
                missing.len(),
                &missing[..missing.len().min(6)]
            ));
        }
        // Key presence is not enough: a checkpoint for another architecture would load and then
        // fail deep inside a forward pass, so the shapes are checked against the config once here.
        let shape_is = |key: &str, expected: &[i32]| -> Res<()> {
            let shape = self.params.get(key)?.shape();
            if shape != expected {
                return Err(format!("{key} has shape {shape:?}, expected {expected:?}"));
            }
            Ok(())
        };
        let hidden = self.cfg.hidden_size;
        let qkv = 3 * self.cfg.heads * self.cfg.head_dim;
        shape_is(
            "encoder.embeddings.tok_embeddings.weight",
            &[self.cfg.vocab, hidden],
        )?;
        shape_is("encoder.layers.0.attn.Wqkv.weight", &[qkv, hidden])?;
        shape_is(
            "encoder.layers.0.mlp.Wi.weight",
            &[2 * self.cfg.intermediate, hidden],
        )?;
        shape_is(
            "encoder.layers.0.mlp.Wo.weight",
            &[hidden, self.cfg.intermediate],
        )?;
        Ok(())
    }

    fn encoder_attention(&self, prefix: &str, x: &Array, theta: f32, mask: &Array) -> Res<Array> {
        let shape = x.shape();
        let (batch, seq) = (shape[0], shape[1]);
        let _ = batch;
        let (heads, head_dim) = (self.cfg.heads, self.cfg.head_dim);
        let qkv = linear(x, self.params.get(&format!("{prefix}.Wqkv.weight"))?, None)?;
        let qkv = qkv
            .reshape(&[batch, seq, 3, heads, head_dim])
            .map_err(msg)?;
        let parts = ops::split_equal(&qkv, 3, 2).map_err(msg)?;
        let to_heads = |part: &Array| -> Res<Array> {
            part.reshape(&[batch, seq, heads, head_dim])
                .map_err(msg)?
                .swap_axes(1, 2)
                .map_err(msg)
        };
        let q = to_heads(&parts[0])?;
        let k = to_heads(&parts[1])?;
        let value = to_heads(&parts[2])?;

        // The base differs by layer type in the checkpoint config, so it is passed per call.
        let q = apply_rope(&q, theta, head_dim, seq)?;
        let k = apply_rope(&k, theta, head_dim, seq)?;

        let scale = (head_dim as f32).powf(-0.5);
        let context = if self.fused {
            scaled_dot_product_attention(
                &q,
                &k,
                &value,
                scale,
                ScaledDotProductAttentionMask::Array(mask),
                None,
            )
            .map_err(msg)?
        } else {
            let scores = ops::add(
                &ops::multiply(
                    &q.matmul(&k.swap_axes(2, 3).map_err(msg)?).map_err(msg)?,
                    &Array::from_f32(scale),
                )
                .map_err(msg)?,
                &ops::select(mask, &Array::from_f32(0.0), &Array::from_f32(-1e9)).map_err(msg)?,
            )
            .map_err(msg)?;
            let probs = ops::softmax_axis(&scores.as_dtype(Dtype::Float32).map_err(msg)?, -1, None)
                .map_err(msg)?
                .as_dtype(q.dtype())
                .map_err(msg)?;
            probs.matmul(&value).map_err(msg)?
        };
        let context = context
            .swap_axes(1, 2)
            .map_err(msg)?
            .reshape(&[batch, seq, self.cfg.hidden_size])
            .map_err(msg)?;
        linear(
            &context,
            self.params.get(&format!("{prefix}.Wo.weight"))?,
            None,
        )
    }

    /// Build the attention masks, then run the encoder.
    ///
    /// An MLX compile transform was tried here and removed: it saved only 3.6% on the encoder, the
    /// shapeless form cannot infer sizes through Split, and per-shape tracing costs about 1.7 s per
    /// new sequence length, which is a loss for a service that sees varying input lengths.
    pub fn encode(&self, input_ids: &Array, attention_mask: &[Vec<u8>]) -> Res<Array> {
        let seq = input_ids.shape()[1];
        let (full_mask, sliding_mask) = build_masks(attention_mask, seq, self.cfg.sliding_window)?;
        with_stream(&self.stream, || {
            encode_core(self, input_ids, &full_mask, &sliding_mask).map_err(msg)
        })
    }

    fn head_attention(&self, prefix: &str, x: &Array, key_keep: &Array) -> Res<Array> {
        let shape = x.shape();
        let (batch, seq, width) = (shape[0], shape[1], shape[2]);
        let heads = (width / 64).max(1);
        let head_dim = width / heads;
        let qkv = linear(
            x,
            self.params
                .get(&format!("{prefix}.self_attn.in_proj_weight"))?,
            self.params
                .get(&format!("{prefix}.self_attn.in_proj_bias"))
                .ok(),
        )?;
        let parts = ops::split_equal(
            &qkv.reshape(&[batch, seq, 3, heads, head_dim])
                .map_err(msg)?,
            3,
            2,
        )
        .map_err(msg)?;
        let to_heads = |part: &Array| -> Res<Array> {
            part.reshape(&[batch, seq, heads, head_dim])
                .map_err(msg)?
                .swap_axes(1, 2)
                .map_err(msg)
        };
        let q = to_heads(&parts[0])?;
        let k = to_heads(&parts[1])?;
        let value = to_heads(&parts[2])?;
        let scale = (head_dim as f32).powf(-0.5);
        let context = if self.fused {
            scaled_dot_product_attention(
                &q,
                &k,
                &value,
                scale,
                ScaledDotProductAttentionMask::Array(key_keep),
                None,
            )
            .map_err(msg)?
        } else {
            let scores = ops::add(
                &ops::multiply(
                    &q.matmul(&k.swap_axes(2, 3).map_err(msg)?).map_err(msg)?,
                    &Array::from_f32(scale),
                )
                .map_err(msg)?,
                &ops::select(key_keep, &Array::from_f32(0.0), &Array::from_f32(-1e9))
                    .map_err(msg)?,
            )
            .map_err(msg)?;
            let probs = ops::softmax_axis(&scores.as_dtype(Dtype::Float32).map_err(msg)?, -1, None)
                .map_err(msg)?
                .as_dtype(q.dtype())
                .map_err(msg)?;
            probs.matmul(&value).map_err(msg)?
        };
        let context = context
            .swap_axes(1, 2)
            .map_err(msg)?
            .reshape(&[batch, seq, width])
            .map_err(msg)?;
        linear(
            &context,
            self.params
                .get(&format!("{prefix}.self_attn.out_proj.weight"))?,
            self.params
                .get(&format!("{prefix}.self_attn.out_proj.bias"))
                .ok(),
        )
    }

    fn head_layer(&self, index: i32, x: &Array, key_keep: &Array) -> Res<Array> {
        let prefix = format!("head.layers.{index}");
        let normed = layer_norm(
            x,
            self.params.get(&format!("{prefix}.norm1.weight"))?,
            self.params.get(&format!("{prefix}.norm1.bias")).ok(),
            1e-5,
        )?;
        let attended = self.head_attention(&prefix, &normed, key_keep)?;
        let mut hidden = ops::add(x, &attended).map_err(msg)?;
        let normed = layer_norm(
            &hidden,
            self.params.get(&format!("{prefix}.norm2.weight"))?,
            self.params.get(&format!("{prefix}.norm2.bias")).ok(),
            1e-5,
        )?;
        let inner = linear(
            &normed,
            self.params.get(&format!("{prefix}.linear1.weight"))?,
            self.params.get(&format!("{prefix}.linear1.bias")).ok(),
        )?;
        let activated = nn::relu(&inner).map_err(msg)?;
        let residual = linear(
            &activated,
            self.params.get(&format!("{prefix}.linear2.weight"))?,
            self.params.get(&format!("{prefix}.linear2.bias")).ok(),
        )?;
        hidden = ops::add(&hidden, &residual).map_err(msg)?;
        Ok(hidden)
    }

    /// Returns (logits, action logits) with shapes (batch, questions) and (batch, 2).
    pub fn forward(
        &self,
        input_ids: &Array,
        attention_mask: &[Vec<u8>],
        marker_pos: &[Vec<i32>],
        marker_mask: &[Vec<u8>],
        qtype: &[i32],
    ) -> Res<(Array, Array)> {
        let batch = input_ids.shape()[0];
        let questions = marker_pos[0].len() as i32;
        let mut hidden = self.encode(input_ids, attention_mask)?;
        trace("encoded", &hidden);
        let mut type_index = Vec::with_capacity((batch * self.cfg.hidden_size) as usize);
        for code in qtype {
            for _ in 0..self.cfg.hidden_size {
                type_index.push(*code);
            }
        }
        let type_index = Array::from_slice(&type_index, &[batch, self.cfg.hidden_size]);
        let type_row = self
            .params
            .get("type_emb.weight")?
            .take_along_axis(&type_index, 0)
            .map_err(msg)?
            .reshape(&[batch, 1, self.cfg.hidden_size])
            .map_err(msg)?;
        hidden = ops::add(&hidden, &type_row).map_err(msg)?;

        // MLX boolean attention masks mean "attend here", matching the encoder masks above.
        // Marking padding as True would invert the head attention.
        let mut key_keep_values = vec![false; (batch * input_ids.shape()[1]) as usize];
        for row in 0..batch as usize {
            for key in 0..input_ids.shape()[1] as usize {
                key_keep_values[row * input_ids.shape()[1] as usize + key] =
                    attention_mask[row][key] != 0;
            }
        }
        let key_padding = Array::from_slice(&key_keep_values, &[batch, 1, 1, input_ids.shape()[1]]);
        trace("after type_emb", &hidden);
        for index in 0..self.head_layers {
            hidden = self.head_layer(index, &hidden, &key_padding)?;
            trace(&format!("head{index}.out"), &hidden);
        }

        let mut flat_positions = Vec::with_capacity((batch * questions) as usize);
        for row in marker_pos.iter() {
            flat_positions.extend(row.iter().map(|value| (*value).max(0)));
        }
        let gather = ops::broadcast_to(
            &Array::from_slice(&flat_positions, &[batch, questions, 1]),
            &[batch, questions, hidden.shape()[2]],
        )
        .map_err(msg)?;
        let markers = hidden.take_along_axis(&gather, 1).map_err(msg)?;

        let scored = layer_norm(
            &markers,
            self.params.get("scorer.0.weight")?,
            self.params.get("scorer.0.bias").ok(),
            1e-5,
        )?;
        let scored = self
            .gelu
            .borrow_mut()
            .apply(&linear(
                &scored,
                self.params.get("scorer.1.weight")?,
                self.params.get("scorer.1.bias").ok(),
            )?)
            .map_err(msg)?;
        let logits = linear(
            &scored,
            self.params.get("scorer.3.weight")?,
            self.params.get("scorer.3.bias").ok(),
        )?
        .squeeze_axes(&[-1])
        .map_err(msg)?;

        let mut flat_marker_mask = Vec::with_capacity((batch * questions) as usize);
        for row in marker_mask.iter() {
            flat_marker_mask.extend(row.iter().map(|flag| *flag != 0));
        }
        let marker_mask = Array::from_slice(&flat_marker_mask, &[batch, questions]);
        let logits = ops::select(&marker_mask, &logits, &Array::from_f32(-1e4)).map_err(msg)?;

        let probs = ops::softmax_axis(&logits.as_dtype(Dtype::Float32).map_err(msg)?, -1, None)
            .map_err(msg)?;
        let counts = ops::maximum(
            &ops::sum_axis(
                &marker_mask.as_dtype(Dtype::Float32).map_err(msg)?,
                -1,
                true,
            )
            .map_err(msg)?,
            &Array::from_f32(2.0),
        )
        .map_err(msg)?;
        let negative_sum = ops::multiply(
            &ops::sum_axis(
                &ops::multiply(
                    &probs,
                    &ops::log(&ops::maximum(&probs, &Array::from_f32(1e-9)).map_err(msg)?)
                        .map_err(msg)?,
                )
                .map_err(msg)?,
                -1,
                true,
            )
            .map_err(msg)?,
            &Array::from_f32(-1.0),
        )
        .map_err(msg)?;
        let entropy = ops::divide(&negative_sum, &ops::log(&counts).map_err(msg)?).map_err(msg)?;

        let sorted = ops::sort_axis(&probs, -1).map_err(msg)?;
        let index_first = Array::from_slice(&vec![questions - 1; batch as usize], &[batch, 1]);
        let index_second =
            Array::from_slice(&vec![(questions - 2).max(0); batch as usize], &[batch, 1]);
        let first = sorted.take_along_axis(&index_first, -1).map_err(msg)?;
        let second = sorted.take_along_axis(&index_second, -1).map_err(msg)?;
        let features = ops::concatenate(
            &[
                &first,
                &ops::subtract(&first, &second).map_err(msg)?,
                &entropy,
                &ops::divide(&counts, &Array::from_f32(255.0)).map_err(msg)?,
            ],
            -1,
        )
        .map_err(msg)?;

        let pooled_index = ops::broadcast_to(
            &Array::from_slice(&vec![0i32; batch as usize], &[batch, 1, 1]),
            &[batch, 1, hidden.shape()[2]],
        )
        .map_err(msg)?;
        let pooled = hidden
            .take_along_axis(&pooled_index, 1)
            .map_err(msg)?
            .reshape(&[batch, hidden.shape()[2]])
            .map_err(msg)?
            .as_dtype(Dtype::Float32)
            .map_err(msg)?;
        let joined = ops::concatenate(&[&pooled, &features], -1).map_err(msg)?;
        let act = self
            .gelu
            .borrow_mut()
            .apply(&linear(
                &joined,
                self.params.get("act_head.0.weight")?,
                self.params.get("act_head.0.bias").ok(),
            )?)
            .map_err(msg)?;
        let act = linear(
            &act,
            self.params.get("act_head.2.weight")?,
            self.params.get("act_head.2.bias").ok(),
        )?;
        // No explicit eval: the caller reads both arrays, which evaluates the shared graph.
        Ok((logits, act))
    }
}
