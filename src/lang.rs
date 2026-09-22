//! Script and language detection used to route between checkpoints.
//!
//! Ported from the upstream Apache-2.0 reference (NandhaKishorM/laya, laya/lang.py). The routing
//! decision has to match the published benchmarks, so the ranges, stopwords and margins are the
//! reference's, not invented here. Only the string handling is idiomatic Rust: char::is_alphabetic
//! supplies the Unicode letter test that Python's re module provided.

use serde_json::Value;

/// Unicode blocks the English checkpoint cannot read.
const SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    (
        "cyrillic",
        &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    ),
    ("hebrew", &[(0x0590, 0x05FF)]),
    (
        "arabic",
        &[
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ],
    ),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("oriya", &[(0x0B00, 0x0B7F)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0C7F)]),
    ("kannada", &[(0x0C80, 0x0CFF)]),
    ("malayalam", &[(0x0D00, 0x0D7F)]),
    ("sinhala", &[(0x0D80, 0x0DFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("lao", &[(0x0E80, 0x0EFF)]),
    ("tibetan", &[(0x0F00, 0x0FFF)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("georgian", &[(0x10A0, 0x10FF)]),
    ("ethiopic", &[(0x1200, 0x137F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    (
        "hangul",
        &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    ),
    (
        "kana",
        &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    ),
    (
        "han",
        &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    ),
];

/// Function words per language, in the reference's order (ties keep the earlier language).
const STOPWORDS: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
            "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
            "will", "can", "would", "there", "their", "what", "which", "please", "we", "i",
        ],
    ),
    (
        "fr",
        &[
            "le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
            "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
        ],
    ),
    (
        "de",
        &[
            "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
            "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
        ],
    ),
    (
        "es",
        &[
            "el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
            "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus",
        ],
    ),
    (
        "pt",
        &[
            "os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
            "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela",
        ],
    ),
    (
        "it",
        &[
            "il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
            "questo", "questa", "anche", "come", "più", "nella", "alla",
        ],
    ),
    (
        "nl",
        &[
            "het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
            "door", "maar", "ook", "worden", "deze", "naar", "wordt",
        ],
    ),
];

const NON_EN_DIACRITICS: &str = "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö";

fn is_latin(cp: u32) -> bool {
    cp < 0x0250 || (0x1E00..=0x1EFF).contains(&cp)
}

fn script_of(cp: u32) -> Option<&'static str> {
    for (name, ranges) in SCRIPT_RANGES {
        if ranges
            .iter()
            .any(|(low, high)| (*low..=*high).contains(&cp))
        {
            return Some(name);
        }
    }
    None
}

fn leaves(state: &Value, depth: usize, out: &mut Vec<String>) {
    if depth > 6 {
        return;
    }
    match state {
        Value::String(text) => out.push(text.clone()),
        Value::Object(map) => {
            for value in map.values() {
                leaves(value, depth + 1, out);
            }
        }
        Value::Array(items) => {
            for item in items {
                leaves(item, depth + 1, out);
            }
        }
        _ => {}
    }
}

/// Flatten a state into the text used for detection. Keys are ignored: they are usually English.
pub fn state_text(state: &Value, max_chars: usize) -> String {
    let mut parts = Vec::new();
    leaves(state, 0, &mut parts);
    parts.join(" ").chars().take(max_chars).collect()
}

fn tally(text: &str) -> Vec<(&'static str, usize)> {
    let mut counts: Vec<(&'static str, usize)> = vec![("latin", 0)];
    for ch in text.chars().filter(|c| c.is_alphabetic()) {
        let cp = ch as u32;
        let name = if is_latin(cp) {
            Some("latin")
        } else {
            script_of(cp)
        };
        if let Some(name) = name {
            match counts.iter_mut().find(|(key, _)| *key == name) {
                Some(entry) => entry.1 += 1,
                None => counts.push((name, 1)),
            }
        }
    }
    counts
}

pub fn detect_script(text: &str) -> &'static str {
    let counts = tally(text);
    let total: usize = counts.iter().map(|(_, count)| count).sum();
    if total == 0 {
        return "unknown";
    }
    let mut best = ("latin", 0usize);
    for (name, count) in counts {
        if count > best.1 {
            best = (name, count);
        }
    }
    best.0
}

pub fn script_profile(text: &str) -> Vec<(&'static str, f64)> {
    let counts = tally(text);
    let total: usize = counts.iter().map(|(_, count)| count).sum();
    if total == 0 {
        return Vec::new();
    }
    counts
        .into_iter()
        .filter(|(_, count)| *count > 0)
        .map(|(name, count)| (name, count as f64 / total as f64))
        .collect()
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphabetic())
        .filter(|part| !part.is_empty())
        .map(|part| part.to_lowercase())
        .collect()
}

/// Best-effort language code for Latin-script text, or None when undecided.
pub fn guess_latin_language(text: &str) -> Option<&'static str> {
    let tokens = words(text);
    if tokens.len() < 4 {
        return None;
    }
    let score = |language: &str| -> usize {
        let list = STOPWORDS
            .iter()
            .find(|(key, _)| *key == language)
            .map(|(_, list)| *list)
            .unwrap_or(&[]);
        tokens
            .iter()
            .filter(|token| list.contains(&token.as_str()))
            .count()
    };
    let english = score("en");
    let lowered = text.to_lowercase();
    let characters = lowered.chars().count().max(1);
    let diacritics = lowered
        .chars()
        .filter(|ch| NON_EN_DIACRITICS.contains(*ch))
        .count();
    let diacritic_rate = diacritics as f64 / characters as f64;

    // Strictly greater keeps the reference's behaviour of preferring the earlier language on ties.
    let mut best = (None, 0usize);
    for (language, _) in STOPWORDS.iter().filter(|(key, _)| *key != "en") {
        let value = score(language);
        if value > best.1 {
            best = (Some(*language), value);
        }
    }
    let (best_language, best_score) = best;
    if best_score == 0 && diacritic_rate < 0.02 {
        return if english > 0 { Some("en") } else { None };
    }
    if let Some(language) = best_language {
        if best_score >= 2.max(english + 2) {
            return Some(language);
        }
        if diacritic_rate >= 0.04 && best_score >= english {
            return Some(language);
        }
    }
    if english > 0 {
        Some("en")
    } else {
        None
    }
}

#[derive(Debug, Clone)]
pub struct Analysis {
    pub script: String,
    pub language: Option<String>,
    pub is_english: bool,
    pub non_latin_fraction: f64,
}

pub fn analyse(state: &Value) -> Analysis {
    let text = state_text(state, 4000);
    let profile = script_profile(&text);
    let script = detect_script(&text).to_string();
    let latin = profile
        .iter()
        .find(|(name, _)| *name == "latin")
        .map(|(_, value)| *value)
        .unwrap_or(0.0);
    let non_latin_fraction = if profile.is_empty() { 0.0 } else { 1.0 - latin };
    if script == "unknown" {
        return Analysis {
            script,
            language: None,
            is_english: true,
            non_latin_fraction: 0.0,
        };
    }
    if script != "latin" {
        return Analysis {
            script,
            language: None,
            is_english: false,
            non_latin_fraction,
        };
    }
    let language = guess_latin_language(&text);
    Analysis {
        script,
        language: language.map(str::to_string),
        is_english: language.is_none() || language == Some("en"),
        non_latin_fraction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scripts_are_detected_from_the_state_text() {
        assert_eq!(detect_script("I was charged twice"), "latin");
        assert_eq!(
            detect_script("4411번 청구서가 두 번 결제되었습니다"),
            "hangul"
        );
        assert_eq!(detect_script("发票被重复扣款"), "han");
        assert_eq!(detect_script(""), "unknown");
    }

    #[test]
    fn english_and_other_latin_languages_are_separated() {
        assert!(analyse(&json!({"body": "I was charged twice and I want the duplicate refunded today, please"})).is_english);
        let german = analyse(
            &json!({"body": "Ich wurde zweimal für Rechnung belastet, bitte erstatten Sie den Betrag"}),
        );
        assert!(!german.is_english, "German was treated as English");
        assert!(!analyse(&json!({"body": "4411번 청구서가 두 번 결제되었습니다"})).is_english);
    }
}
