//! Question normalisation, prompt construction and decision post-processing.
//!
//! Ported from the upstream Apache-2.0 reference (NandhaKishorM/laya, laya/common.py) and from
//! this repository's MLX port. The prompt format decides what the model sees, so it is kept
//! identical; any drift here changes answers.

use serde::Serialize;
use serde_json::Value;
use std::io::Write;

/// Python's `json.dumps` default separators: `", "` between items and `": "` after a key.
///
/// This is load-bearing, not cosmetic: the model reads the state as JSON, so compact output turns
/// `{"body": "x"}` into different tokens and silently changes every answer.
struct PythonFormatter;

impl serde_json::ser::Formatter for PythonFormatter {
    fn begin_array_value<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_key<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_value<W: Write + ?Sized>(&mut self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

pub fn py_json<T: Serialize>(value: &T) -> String {
    let mut buffer = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buffer, PythonFormatter);
    if value.serialize(&mut serializer).is_err() {
        return String::from("null");
    }
    String::from_utf8(buffer).unwrap_or_else(|_| String::from("null"))
}

pub const QTYPES_CHOICE: i32 = 0;
pub const QTYPES_SCORE: i32 = 1;
pub const QTYPES_NOUL: i32 = 2;

pub fn qtype(kind: &str) -> Option<i32> {
    match kind {
        "choice" => Some(QTYPES_CHOICE),
        "score" => Some(QTYPES_SCORE),
        "noul" => Some(QTYPES_NOUL),
        _ => None,
    }
}

/// One question, normalised the way the model expects it.
#[derive(Debug, Clone)]
pub struct Question {
    pub kind: String,
    pub instructions: String,
    /// Choice: label -> optional description. Score: one entry per level. Noul: optional keys.
    pub criteria: Vec<(String, Option<Value>)>,
    /// Score levels, in order (the criteria values).
    pub levels: Vec<String>,
}

impl Question {
    /// How many options the scorer must produce for this question.
    pub fn option_count(&self) -> usize {
        match self.kind.as_str() {
            "choice" => self.criteria.len(),
            "score" => self.levels.len(),
            _ => 2,
        }
    }
}

pub fn to_internal(raw: &Value) -> Result<Question, String> {
    let kind = raw
        .get("type")
        .and_then(Value::as_str)
        .ok_or("question is missing a string 'type'")?
        .to_string();
    if qtype(&kind).is_none() {
        return Err(format!(
            "unsupported question type {kind:?}; expected choice, score or noul"
        ));
    }
    let instructions = match raw.get("instructions") {
        Some(Value::String(text)) => text.clone(),
        Some(other) => py_json(other),
        None => return Err("question is missing 'instructions'".to_string()),
    };
    let criteria = raw.get("criteria");
    let mut pairs: Vec<(String, Option<Value>)> = Vec::new();
    let mut levels: Vec<String> = Vec::new();

    match kind.as_str() {
        "choice" => match criteria {
            Some(Value::Array(items)) => {
                for item in items {
                    let label = item
                        .as_str()
                        .ok_or("choice criteria array must hold strings")?
                        .to_string();
                    pairs.push((label, None));
                }
            }
            Some(Value::Object(map)) => {
                for (key, value) in map {
                    let description = match value {
                        Value::Null => None,
                        Value::String(text) if text.is_empty() => None,
                        other => Some(other.clone()),
                    };
                    pairs.push((key.clone(), description));
                }
            }
            Some(_) => return Err("choice criteria must be an object or an array".to_string()),
            None => return Err("choice question is missing 'criteria'".to_string()),
        },
        "score" => match criteria {
            Some(Value::Array(items)) => {
                for item in items {
                    levels.push(render_criterion(item));
                }
            }
            _ => return Err("score criteria must be an array".to_string()),
        },
        _ => {
            if let Some(Value::Object(map)) = criteria {
                for key in ["false", "true"] {
                    let value = map.get(key).and_then(|entry| match entry {
                        Value::Null => None,
                        Value::String(text) if text.is_empty() => None,
                        other => Some(other.clone()),
                    });
                    pairs.push((key.to_string(), value));
                }
            }
        }
    }
    Ok(Question {
        kind,
        instructions,
        criteria: pairs,
        levels,
    })
}

/// Render one criterion value: strings pass through, structured values become compact JSON.
pub fn render_criterion(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => py_json(other),
    }
}

/// Option texts in label-index order. Noul is always [false, true].
pub fn render_options(question: &Question) -> Vec<String> {
    match question.kind.as_str() {
        "choice" => question
            .criteria
            .iter()
            .map(|(label, description)| match description {
                Some(value) => format!("{label}: {}", render_criterion(value)),
                None => label.clone(),
            })
            .collect(),
        "score" => question
            .levels
            .iter()
            .enumerate()
            .map(|(index, level)| format!("level {index}: {level}"))
            .collect(),
        _ => {
            let find = |key: &str| {
                question
                    .criteria
                    .iter()
                    .find(|(label, _)| label == key)
                    .and_then(|(_, value)| value.clone())
            };
            let false_text = match find("false") {
                Some(value) => render_criterion(&value),
                None => "no, the statement does not hold".to_string(),
            };
            let true_text = match find("true") {
                Some(value) => render_criterion(&value),
                None => "yes, the statement holds".to_string(),
            };
            vec![format!("false: {false_text}"), format!("true: {true_text}")]
        }
    }
}

pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(text) => text.clone(),
        other => py_json(other),
    }
}

/// Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
pub fn build_sequence(
    tok: &crate::tokenizer::Tokenizer,
    state: &Value,
    question: &Question,
    max_len: usize,
    head_max_len: usize,
) -> Result<(Vec<u32>, Vec<usize>), String> {
    let mask_token = tok.mask_token.as_str();
    let options = render_options(question);
    let instructions = question.instructions.replace(mask_token, " ");
    let head_text = format!("{} question: {}", question.kind, instructions);
    let mut head_ids = tok.encode(&head_text)?;

    let mut option_ids: Vec<Vec<u32>> = Vec::with_capacity(options.len());
    for option in &options {
        let cleaned = option.replace(mask_token, " ");
        let mut ids = vec![tok.mask_token_id];
        ids.extend(tok.encode(&format!(" {cleaned}"))?.into_iter().take(48));
        option_ids.push(ids);
    }

    let mut budget = head_max_len as i64 - option_ids.iter().map(Vec::len).sum::<usize>() as i64;
    if budget < 16 {
        let per = std::cmp::max(
            4,
            (head_max_len as i64 - 16) / std::cmp::max(1, option_ids.len()) as i64,
        );
        option_ids = option_ids
            .into_iter()
            .map(|ids| ids[..ids.len().min(per as usize)].to_vec())
            .collect();
        budget = head_max_len as i64 - option_ids.iter().map(Vec::len).sum::<usize>() as i64;
    }
    head_ids.truncate(head_ids.len().min(std::cmp::max(8, budget) as usize));

    let mut ids: Vec<u32> = Vec::with_capacity(max_len);
    ids.push(tok.cls_token_id);
    ids.extend(head_ids);
    ids.push(tok.sep_token_id);
    let mut markers = Vec::with_capacity(option_ids.len());
    for option in option_ids {
        markers.push(ids.len());
        ids.extend(option);
    }
    ids.push(tok.sep_token_id);

    let room = max_len.saturating_sub(ids.len() + 1);
    let state_ids = tok.encode(&serialize_state(state).replace(mask_token, " "))?;
    let start = state_ids.len().saturating_sub(room);
    ids.extend(state_ids[start..].iter().take(room).copied());
    ids.push(tok.sep_token_id);

    ids.truncate(max_len);
    markers.retain(|marker| *marker < max_len);
    let expected = render_options(question).len();
    if markers.len() != expected {
        return Err(format!(
            "question options exceed head_max_len={head_max_len}"
        ));
    }
    Ok((ids, markers))
}

/// Normalised Shannon entropy confidence: 1 - H(p) / log(k).
pub fn confidence_from_probs(probs: &[f32], count: usize) -> f32 {
    if count < 2 {
        return 1.0;
    }
    let entropy: f32 = probs[..count].iter().map(|p| -p * p.max(1e-12).ln()).sum();
    (1.0 - entropy / (count as f32).ln()).clamp(0.0, 1.0)
}

pub fn temp_bucket(kind: &str, count: usize) -> String {
    let size = match count {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{}", kind, size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The state is read by the model as text, so compact JSON changes the tokens and every answer.
    #[test]
    fn state_serialisation_matches_python_separators() {
        let state = json!({"from": "user@acme.com", "body": "hi", "tags": ["a", "b"]});
        assert_eq!(
            serialize_state(&state),
            "{\"from\": \"user@acme.com\", \"body\": \"hi\", \"tags\": [\"a\", \"b\"]}"
        );
    }

    #[test]
    fn string_states_pass_through_unchanged() {
        assert_eq!(serialize_state(&json!("plain text")), "plain text");
    }

    #[test]
    fn structured_criteria_use_python_separators() {
        assert_eq!(
            render_criterion(&json!({"desc": "urgent", "weight": 2})),
            "{\"desc\": \"urgent\", \"weight\": 2}"
        );
        assert_eq!(render_criterion(&json!("already text")), "already text");
    }

    #[test]
    fn option_counts_match_the_answer_primitives() {
        let choice = to_internal(
            &json!({"type": "choice", "instructions": "pick", "criteria": {"a": null, "b": null}}),
        )
        .unwrap();
        assert_eq!(choice.option_count(), 2);
        let score = to_internal(
            &json!({"type": "score", "instructions": "rate", "criteria": ["one", "two", "three"]}),
        )
        .unwrap();
        assert_eq!(score.option_count(), 3);
        let noul = to_internal(&json!({"type": "noul", "instructions": "yes?"})).unwrap();
        assert_eq!(noul.option_count(), 2);
    }
}
