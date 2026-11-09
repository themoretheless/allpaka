//! OpenAI message rendering, Hermes tool-call parsing, and incremental UTF-8 streaming.

use serde_json::{json, Value};

/// OpenAI messages carry `content` either as a plain string or as an array of
/// typed parts. The engine is text-only today, so a parts array is flattened to
/// its text and the images are only *counted*: the handler refuses them
/// explicitly. Reading `content` with `as_str()` made a whole vision message -
/// picture and text together - collapse into an empty turn.
pub(crate) fn content_parts(content: &Value) -> (String, usize) {
    if let Some(text) = content.as_str() {
        return (text.to_string(), 0);
    }
    let mut text = String::new();
    let mut images = 0usize;
    for part in content.as_array().into_iter().flatten() {
        match part["type"].as_str().unwrap_or_default() {
            "text" | "input_text" => {
                if let Some(chunk) = part["text"].as_str() {
                    text.push_str(chunk);
                }
            }
            "image_url" | "input_image" | "image" => images += 1,
            _ => {}
        }
    }
    (text, images)
}

/// Image parts anywhere in the request, across all messages.
pub(crate) fn image_part_count(raw: &[Value]) -> usize {
    raw.iter()
        .map(|m| {
            m["content"]
                .as_array()
                .map(|parts| {
                    parts
                        .iter()
                        .filter(|part| {
                            matches!(
                                part["type"].as_str(),
                                Some("image_url" | "input_image" | "image")
                            )
                        })
                        .count()
                })
                .unwrap_or(0)
        })
        .sum()
}

/// Convert an OpenAI-shape message list (+ optional `tools`) into plain
/// (role, content) turns in the Hermes convention Qwen3 was trained on:
/// tool schemas inside `<tools>...</tools>` in the system turn, assistant
/// calls as `<tool_call>{json}</tool_call>` blocks, tool results as user
/// turns wrapped in `<tool_response>...</tool_response>` (consecutive
/// results merged into one turn, as the reference chat template does).
pub(crate) fn render_messages(raw: &[Value], tools: Option<&[Value]>) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut pending_tool: Vec<String> = Vec::new();
    let flush_tools = |out: &mut Vec<(String, String)>, pending: &mut Vec<String>| {
        if !pending.is_empty() {
            let body = pending
                .iter()
                .map(|c| format!("<tool_response>\n{c}\n</tool_response>"))
                .collect::<Vec<_>>()
                .join("\n");
            out.push(("user".into(), body));
            pending.clear();
        }
    };
    for m in raw {
        let role = m["role"].as_str().unwrap_or("user");
        let content = content_parts(&m["content"]).0;
        if role == "tool" {
            pending_tool.push(content);
            continue;
        }
        flush_tools(&mut out, &mut pending_tool);
        if role == "assistant" {
            let mut text = content;
            if let Some(calls) = m["tool_calls"].as_array() {
                for c in calls {
                    let name = c["function"]["name"].as_str().unwrap_or_default();
                    let name_json =
                        serde_json::to_string(name).unwrap_or_else(|_| "\"\"".to_string());
                    // Arguments arrive as a JSON-encoded string per the
                    // OpenAI shape; embed them as the object Hermes expects.
                    let args = c["function"]["arguments"].as_str().unwrap_or("{}");
                    text.push_str(&format!(
                        "\n<tool_call>\n{{\"name\": {name_json}, \"arguments\": {args}}}\n</tool_call>"
                    ));
                }
            }
            out.push(("assistant".into(), text));
        } else {
            out.push((role.to_string(), content));
        }
    }
    flush_tools(&mut out, &mut pending_tool);

    if let Some(tools) = tools {
        let schemas = tools
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let section = format!(
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\
             <tools>\n{schemas}\n</tools>\n\n\
             For each function call, return a json object with function name and arguments \
             within <tool_call></tool_call> XML tags:\n\
             <tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
        );
        match out.iter_mut().find(|(role, _)| role == "system") {
            Some((_, content)) => content.push_str(&section),
            None => out.insert(
                0,
                (
                    "system".into(),
                    format!("You are a helpful assistant.{section}"),
                ),
            ),
        }
    }
    out
}

/// Split generated text into visible content and Hermes tool-call blocks,
/// returned in the OpenAI `tool_calls` shape (arguments re-encoded as a JSON
/// string). Malformed blocks stay in the content untouched - the client sees
/// what the model actually wrote instead of a silent drop.
pub(crate) fn parse_tool_calls(text: &str) -> (String, Vec<Value>) {
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        let Some(end) = after.find("</tool_call>") else {
            break;
        };
        let inner = after[..end].trim();
        let parsed: Option<Value> = serde_json::from_str(inner).ok();
        let valid = parsed
            .as_ref()
            .is_some_and(|v| v["name"].is_string() && !v["arguments"].is_null());
        if valid {
            let v = parsed.unwrap();
            calls.push(json!({
                "id": format!("call_{}", calls.len()),
                "type": "function",
                "function": {
                    "name": v["name"],
                    "arguments": v["arguments"].to_string(),
                },
            }));
            content.push_str(&rest[..start]);
        } else {
            content.push_str(&rest[..start + "<tool_call>".len() + end + "</tool_call>".len()]);
        }
        rest = &after[end + "</tool_call>".len()..];
    }
    content.push_str(rest);
    (content.trim().to_string(), calls)
}

/// The longest prefix of `text` that is safe to stream when tool calls may
/// follow: everything up to the first (possibly still incomplete) opening
/// `<tool_call>` tag. A trailing partial match of the tag is held back too.
pub(crate) fn stream_safe_len(text: &str) -> usize {
    const TAG: &str = "<tool_call>";
    if let Some(i) = text.find(TAG) {
        return i;
    }
    let bytes = text.as_bytes();
    let max_tail = (TAG.len() - 1).min(bytes.len());
    for tail in (1..=max_tail).rev() {
        if text.is_char_boundary(bytes.len() - tail)
            && TAG.as_bytes().starts_with(&bytes[bytes.len() - tail..])
        {
            return bytes.len() - tail;
        }
    }
    bytes.len()
}

/// Incrementally decode newly appended UTF-8 bytes from `raw_bytes[*decoded_bytes..]`
/// into `complete_buf`, holding back incomplete multi-byte trailing sequences.
pub(crate) fn append_valid_utf8(
    raw_bytes: &[u8],
    decoded_bytes: &mut usize,
    complete_buf: &mut String,
    base_prefix_len: usize,
) {
    match std::str::from_utf8(&raw_bytes[*decoded_bytes..]) {
        Ok(valid) => {
            complete_buf.push_str(valid);
            *decoded_bytes = raw_bytes.len();
        }
        Err(e) if e.error_len().is_none() => {
            let valid_len = e.valid_up_to();
            if valid_len > 0 {
                let valid =
                    std::str::from_utf8(&raw_bytes[*decoded_bytes..*decoded_bytes + valid_len])
                        .unwrap_or("");
                complete_buf.push_str(valid);
                *decoded_bytes += valid_len;
            }
        }
        Err(_) => {
            let full = String::from_utf8_lossy(raw_bytes);
            let trimmed = full.strip_suffix('\u{FFFD}').unwrap_or(&full);
            complete_buf.truncate(base_prefix_len);
            complete_buf.push_str(trimmed);
            *decoded_bytes = if full.ends_with('\u{FFFD}') {
                raw_bytes.len().saturating_sub(1)
            } else {
                raw_bytes.len()
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_content_keeps_its_text_and_counts_images() {
        let parts = json!([
            {"type": "text", "text": "what is on this picture?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]);
        let (text, images) = content_parts(&parts);
        assert_eq!(text, "what is on this picture?");
        assert_eq!(images, 1);

        let msgs = vec![json!({"role": "user", "content": parts})];
        assert_eq!(image_part_count(&msgs), 1);
        let out = render_messages(&msgs, None);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0],
            ("user".to_string(), "what is on this picture?".to_string())
        );
    }

    #[test]
    fn string_content_and_unknown_parts_are_handled() {
        let (text, images) = content_parts(&json!("plain"));
        assert_eq!((text.as_str(), images), ("plain", 0));
        let (text, images) = content_parts(&json!([
            {"type": "text", "text": "a"},
            {"type": "audio", "audio": {"data": "x"}},
            {"type": "text", "text": "b"},
        ]));
        assert_eq!((text.as_str(), images), ("ab", 0));
        assert_eq!(
            image_part_count(&[json!({"role": "user", "content": "hi"})]),
            0
        );
    }

    #[test]
    fn tools_section_lands_in_the_system_turn() {
        let msgs = vec![json!({"role": "user", "content": "hi"})];
        let tools = vec![json!({"type":"function","function":{"name":"search","parameters":{}}})];
        let out = render_messages(&msgs, Some(&tools));
        assert_eq!(out[0].0, "system");
        assert!(out[0].1.contains("<tools>"));
        assert!(out[0].1.contains("\"search\""));
        assert_eq!(out[1], ("user".into(), "hi".into()));
        let msgs2 = vec![
            json!({"role":"system","content":"be brief"}),
            msgs[0].clone(),
        ];
        let out2 = render_messages(&msgs2, Some(&tools));
        assert_eq!(out2.len(), 2);
        assert!(out2[0].1.starts_with("be brief"));
        assert!(out2[0].1.contains("<tools>"));
    }

    #[test]
    fn assistant_calls_and_tool_results_round_trip_into_hermes_text() {
        let msgs = vec![
            json!({"role":"user","content":"weather?"}),
            json!({"role":"assistant","content":"","tool_calls":[{"id":"call_0","type":"function",
                "function":{"name":"get_weather","arguments":"{\"city\":\"Oslo\"}"}}]}),
            json!({"role":"tool","tool_call_id":"call_0","content":"{\"temp\":-3}"}),
            json!({"role":"tool","tool_call_id":"call_1","content":"{\"wind\":5}"}),
        ];
        let out = render_messages(&msgs, None);
        assert!(out[1].1.contains("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"Oslo\"}}\n</tool_call>"));
        assert_eq!(out[2].0, "user");
        assert_eq!(out[2].1.matches("<tool_response>").count(), 2);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn tool_call_blocks_parse_and_malformed_ones_stay_in_content() {
        let text = "Let me check.\n<tool_call>\n{\"name\": \"rag_search\", \"arguments\": {\"q\": \"kv cache\"}}\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, "Let me check.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "rag_search");
        assert_eq!(calls[0]["function"]["arguments"], "{\"q\":\"kv cache\"}");
        let (content2, calls2) = parse_tool_calls("<tool_call>not json</tool_call>done");
        assert!(calls2.is_empty());
        assert_eq!(content2, "<tool_call>not json</tool_call>done");
    }

    #[test]
    fn stream_hold_back_covers_partial_opening_tags() {
        assert_eq!(stream_safe_len("hello"), 5);
        assert_eq!(stream_safe_len("hello <tool_call>{"), 6);
        assert_eq!(stream_safe_len("hello <tool_c"), 6);
        assert_eq!(stream_safe_len("<"), 0);
        assert_eq!(stream_safe_len("a < b"), 5);
    }
}
