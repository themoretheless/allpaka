//! Prompt-lookup draft configuration, ghost-text continuation, and follow-up suggestions.

use super::messages::content_parts;
use allpaka_model::Tokenizer;
use serde_json::Value;

/// Extract predicted text from an OpenAI `prediction` parameter:
/// supports `"string"`, `{"type": "content", "content": "string"}`, or
/// `{"type": "content", "content": [{"type": "text", "text": "..."}]}`.
pub(crate) fn extract_prediction_text(prediction: &Value) -> String {
    if let Some(s) = prediction.as_str() {
        return s.to_string();
    }
    if prediction.is_object() {
        return content_parts(&prediction["content"]).0;
    }
    String::new()
}

/// Resolve the number of prompt-lookup draft tokens per round (`0` disables PLD).
pub(crate) fn parse_prompt_lookup_k(val: &Value, default_enabled: bool) -> usize {
    if let Some(b) = val.as_bool() {
        return if b { 4 } else { 0 };
    }
    if let Some(n) = val.as_u64() {
        return (n as usize).min(16);
    }
    if default_enabled {
        4
    } else {
        0
    }
}

/// Predict inline ghost-text continuation for `draft` using conversation token
/// N-grams and message text prefixes.
pub(crate) fn predict_prompt_continuation(
    tok: &Tokenizer,
    context_tokens: &[u32],
    raw_messages: &[Value],
    draft: &str,
    max_k: usize,
) -> String {
    if draft.is_empty() || max_k == 0 {
        return String::new();
    }

    // 1. Token-level N-gram lookup against context_tokens + encoded messages.
    if let Ok(draft_tokens) = tok.encode(draft) {
        if let Some(&last_tok) = draft_tokens.last() {
            let prefix_history = &draft_tokens[..draft_tokens.len() - 1];
            let mut pool = Vec::with_capacity(context_tokens.len() + draft_tokens.len() + 256);
            pool.extend_from_slice(context_tokens);
            for m in raw_messages {
                let text = content_parts(&m["content"]).0;
                if !text.is_empty() {
                    if let Ok(t) = tok.encode(&text) {
                        pool.extend_from_slice(&t);
                    }
                }
            }
            // Append prefix_history at the tail of `pool` so `find_prompt_drafts`
            // sees the draft's own preceding tokens as the suffix context.
            let search_len = pool.len();
            pool.extend_from_slice(prefix_history);
            let min_n = if prefix_history.is_empty() { 1 } else { 2 };
            let drafts = allpaka_model::speculate::find_prompt_drafts(
                &pool,
                &pool[..search_len],
                last_tok,
                max_k,
                min_n,
                4,
            );
            if !drafts.is_empty() {
                let decoded = tok.decode(&drafts);
                let clean = decoded
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim_end_matches('\u{FFFD}');
                if !clean.is_empty() {
                    return clean.to_string();
                }
            }
        }
    }

    // 2. Fallback: case-insensitive substring/word-prefix continuation from raw messages.
    let draft_lower = draft.to_lowercase();
    let tail = draft_lower
        .split_whitespace()
        .rev()
        .take(2)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join(" ");
    if tail.len() >= 3 {
        for m in raw_messages.iter().rev() {
            let text = content_parts(&m["content"]).0;
            let text_lower = text.to_lowercase();
            if let Some(pos) = text_lower.rfind(&tail) {
                let after_byte = pos + tail.len();
                if text.is_char_boundary(after_byte) && after_byte < text.len() {
                    let rest = &text[after_byte..];
                    let line = rest.lines().next().unwrap_or("");
                    let words: Vec<&str> =
                        line.split_inclusive(char::is_whitespace).take(6).collect();
                    let cont = words.concat();
                    if !cont.trim().is_empty() {
                        return cont;
                    }
                }
            }
        }
    }

    String::new()
}

/// Generate context-aware follow-up prompt suggestions for the chat UI.
pub(crate) fn predict_followup_prompts(raw_messages: &[Value]) -> Vec<String> {
    let Some(last) = raw_messages.last() else {
        return vec![
            "Объясни архитектуру этого проекта".into(),
            "Напиши быстрый код на Rust с тестами".into(),
            "Найди узкие места и точки оптимизации".into(),
        ];
    };
    let text = content_parts(&last["content"]).0;
    let lower = text.to_lowercase();
    let mut out = Vec::with_capacity(4);
    if text.contains("```") || lower.contains("fn ") || lower.contains("impl ") {
        out.push("Напиши модульные тесты для этого кода".into());
        out.push("Оптимизируй производительность и убери лишние аллокации".into());
        out.push("Разбери возможные граничные случаи и ошибки".into());
    } else if lower.contains("ошибк") || lower.contains("error") || lower.contains("bug") {
        out.push("Предложи минимальный фикс и проверь тестами".into());
        out.push("Почему возникла эта проблема?".into());
        out.push("Как предотвратить подобные ошибки в будущем?".into());
    } else {
        out.push("Приведи конкретный пример кода".into());
        out.push("Расскажи подробнее про внутреннее устройство".into());
        out.push("Сравни плюсы и минусы этого подхода".into());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_prediction_supports_string_and_openai_object() {
        assert_eq!(extract_prediction_text(&json!("fast draft")), "fast draft");
        assert_eq!(
            extract_prediction_text(&json!({
                "type": "content",
                "content": "predicted output"
            })),
            "predicted output"
        );
        assert_eq!(
            extract_prediction_text(&json!({
                "type": "content",
                "content": [{"type": "text", "text": "part1 "}, {"type": "text", "text": "part2"}]
            })),
            "part1 part2"
        );
        assert_eq!(extract_prediction_text(&json!(null)), "");
    }

    #[test]
    fn prompt_lookup_k_parses_bool_int_and_defaults() {
        assert_eq!(parse_prompt_lookup_k(&json!(null), true), 4);
        assert_eq!(parse_prompt_lookup_k(&json!(null), false), 0);
        assert_eq!(parse_prompt_lookup_k(&json!(false), true), 0);
        assert_eq!(parse_prompt_lookup_k(&json!(true), false), 4);
        assert_eq!(parse_prompt_lookup_k(&json!(8), false), 8);
        assert_eq!(parse_prompt_lookup_k(&json!(100), true), 16);
    }

    #[test]
    fn followup_suggestions_adapt_to_conversation_context() {
        let initial = predict_followup_prompts(&[]);
        assert_eq!(initial.len(), 3);
        let code_turn = vec![json!({"role": "assistant", "content": "```rust\nfn main() {}\n```"})];
        let after_code = predict_followup_prompts(&code_turn);
        assert!(after_code[0].contains("тесты"));
    }
}
