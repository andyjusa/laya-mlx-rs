//! Checkpoint routing: keep the needed checkpoints resident and pick the right one per request.

use std::collections::HashMap;

use mlx_rs::Dtype;
use serde_json::{Map, Value};

use crate::agent::Agent;
use crate::lang::analyse;
use crate::model::Res;

pub const BUNDLE_REPO: &str = "convaiinnovations/laya";

/// The bundle carries all three checkpoints; only the requested subfolder is downloaded.
pub fn default_models() -> Vec<(String, Option<String>)> {
    vec![
        ("english".to_string(), None),
        ("multilingual".to_string(), Some("multilingual".to_string())),
        (
            "typed-decisions".to_string(),
            Some("typed-decisions".to_string()),
        ),
    ]
}

pub fn normalise(name: &str) -> Res<String> {
    let key = name.trim().to_lowercase().replace(' ', "-");
    let key = match key.as_str() {
        "en" | "eng" => "english",
        "ml" => "multilingual",
        "td" | "typed_decisions" => "typed-decisions",
        other => other,
    };
    if !default_models().iter().any(|(known, _)| known == key) {
        return Err(format!("unknown checkpoint {name:?}"));
    }
    Ok(key.to_string())
}

pub struct Router {
    agents: HashMap<String, Agent>,
    models: Vec<(String, Option<String>)>,
    default: String,
}

impl Router {
    /// Load only the requested checkpoints; the LAYA_MODELS environment variable decides which.
    pub fn preload(spec: &str, names: &[String], dtype: Dtype) -> Res<Self> {
        let all = default_models();
        let mut models = Vec::new();
        for name in names {
            let key = normalise(name)?;
            let entry = all
                .iter()
                .find(|(known, _)| *known == key)
                .cloned()
                .ok_or_else(|| format!("unknown checkpoint {name}"))?;
            models.push(entry);
        }
        let mut agents = HashMap::new();
        for (name, subfolder) in &models {
            let agent = Agent::load(spec, subfolder.as_deref(), dtype)?;
            agents.insert(name.clone(), agent);
        }
        Ok(Self {
            agents,
            models,
            default: "english".to_string(),
        })
    }

    pub fn loaded(&self) -> Vec<String> {
        let mut names: Vec<String> = self.models.iter().map(|(name, _)| name.clone()).collect();
        names.sort();
        names
    }

    /// Which checkpoint should answer, and why. Does not run a forward pass.
    pub fn route(&self, state: &Value, model: Option<&str>, lang: Option<&str>) -> Res<Value> {
        let (key, reason, detection) = if let Some(explicit) = model {
            let key = normalise(explicit)?;
            (key, format!("explicit model={explicit:?}"), Value::Null)
        } else if let Some(explicit) = lang {
            let code = explicit.to_lowercase();
            let base = code.split('-').next().unwrap_or("");
            let key = if matches!(base, "en" | "eng" | "english") {
                "english"
            } else {
                "multilingual"
            };
            (
                key.to_string(),
                format!("explicit lang={explicit:?}"),
                Value::Null,
            )
        } else {
            let analysis = analyse(state);
            let mut detection = Map::new();
            detection.insert("script".into(), Value::String(analysis.script.clone()));
            detection.insert(
                "language".into(),
                analysis
                    .language
                    .clone()
                    .map(Value::String)
                    .unwrap_or(Value::Null),
            );
            detection.insert("is_english".into(), Value::Bool(analysis.is_english));
            detection.insert(
                "non_latin_fraction".into(),
                Value::from(analysis.non_latin_fraction),
            );
            if analysis.script == "unknown" {
                let key = self.default.clone();
                (
                    key.clone(),
                    format!("no letters detected in state; using default ({key})"),
                    Value::Object(detection),
                )
            } else if analysis.script != "latin" {
                (
                    "multilingual".to_string(),
                    format!(
                        "non-Latin script ({}, {:.0}% of letters); the English checkpoint cannot read it",
                        analysis.script,
                        analysis.non_latin_fraction * 100.0
                    ),
                    Value::Object(detection),
                )
            } else if !analysis.is_english {
                (
                    "multilingual".to_string(),
                    format!(
                        "Latin script but language looks like {:?}, not English",
                        analysis.language
                    ),
                    Value::Object(detection),
                )
            } else {
                (
                    "english".to_string(),
                    "English Latin text".to_string(),
                    Value::Object(detection),
                )
            }
        };
        if !self.agents.contains_key(&key) {
            return Err(format!(
                "checkpoint {key} is not resident; add it to LAYA_MODELS"
            ));
        }
        let subfolder = self
            .models
            .iter()
            .find(|(name, _)| *name == key)
            .and_then(|(_, subfolder)| subfolder.clone());
        let repo = match subfolder {
            Some(folder) => format!("{BUNDLE_REPO}/{folder}"),
            None => BUNDLE_REPO.to_string(),
        };
        let mut out = Map::new();
        out.insert("model".into(), Value::String(key));
        out.insert("repo".into(), Value::String(repo));
        out.insert("reason".into(), Value::String(reason));
        out.insert("detection".into(), detection);
        Ok(Value::Object(out))
    }

    pub fn predict(
        &self,
        state: &Value,
        questions: &Map<String, Value>,
        model: Option<&str>,
        lang: Option<&str>,
    ) -> Res<Value> {
        let decision = self.route(state, model, lang)?;
        let key = decision
            .get("model")
            .and_then(Value::as_str)
            .ok_or("routing produced no model")?
            .to_string();
        let agent = self
            .agents
            .get(&key)
            .ok_or_else(|| format!("checkpoint {key} is not resident"))?;
        let mut result = agent.predict(state, questions)?;
        if let Value::Object(map) = &mut result {
            map.insert("routing".into(), decision);
        }
        Ok(result)
    }
}
