use crate::types::{Message, Settings};
use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde::Serialize;
use serde_json::{json, Value};
use std::{collections::BTreeMap, time::Duration};

#[derive(Clone)]
pub struct Provider {
    pub id: String,
    pub name: String,
    pub base: String,
    pub key: String,
    pub env: String,
    pub anthropic: bool,
    pub custom: bool,
    pub key_source: String,
    pub key_error: Option<String>,
    pub saved: bool,
}
#[derive(Serialize)]
pub struct PublicProvider {
    pub id: String,
    pub name: String,
    pub configured: bool,
    pub key_env: String,
    pub custom: bool,
    pub key_source: String,
    pub key_error: Option<String>,
    pub saved: bool,
}
impl Provider {
    pub fn public(&self) -> PublicProvider {
        PublicProvider {
            id: self.id.clone(),
            name: self.name.clone(),
            configured: self.id == "local" || self.custom || !self.key.is_empty(),
            key_env: self.env.clone(),
            custom: self.custom,
            key_source: self.key_source.clone(),
            key_error: self.key_error.clone(),
            saved: self.saved,
        }
    }
}
pub fn defaults() -> Vec<Provider> {
    [
        (
            "openai",
            "OpenAI / ChatGPT",
            "https://api.openai.com/v1",
            "OPENAI_API_KEY",
            false,
        ),
        (
            "anthropic",
            "Claude",
            "https://api.anthropic.com/v1",
            "ANTHROPIC_API_KEY",
            true,
        ),
        (
            "deepseek",
            "DeepSeek",
            "https://api.deepseek.com/v1",
            "DEEPSEEK_API_KEY",
            false,
        ),
        (
            "kimi",
            "Kimi",
            "https://api.moonshot.ai/v1",
            "MOONSHOT_API_KEY",
            false,
        ),
        ("xai", "Grok", "https://api.x.ai/v1", "XAI_API_KEY", false),
        (
            "gemini",
            "Gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "GEMINI_API_KEY",
            false,
        ),
        (
            "openrouter",
            "OpenRouter",
            "https://openrouter.ai/api/v1",
            "OPENROUTER_API_KEY",
            false,
        ),
        (
            "local",
            "allpaka local",
            "http://127.0.0.1:8099/v1",
            "ALLPAKA_LOCAL_API_KEY",
            false,
        ),
    ]
    .into_iter()
    .map(|(id, name, base, env, anthropic)| Provider {
        id: id.into(),
        name: name.into(),
        base: if id == "local" {
            std::env::var("ALLPAKA_LOCAL_BASE_URL").unwrap_or(base.into())
        } else {
            base.into()
        },
        key: std::env::var(env).unwrap_or_default(),
        key_source: if std::env::var(env).is_ok_and(|v| !v.is_empty()) {
            "environment"
        } else {
            "none"
        }
        .into(),
        key_error: None,
        saved: false,
        env: env.into(),
        anthropic,
        custom: false,
    })
    .collect()
}
pub fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(300))
        .redirect(reqwest::redirect::Policy::none())
        .build()?)
}
fn request(
    client: &reqwest::Client,
    p: &Provider,
    method: reqwest::Method,
    route: &str,
) -> reqwest::RequestBuilder {
    let r = client.request(method, format!("{}{route}", p.base.trim_end_matches('/')));
    let r = if p.id == "openrouter" {
        r.header("X-OpenRouter-Title", "allpaka Studio")
    } else {
        r
    };
    if p.anthropic {
        r.header("x-api-key", &p.key)
            .header("anthropic-version", "2023-06-01")
    } else if !p.key.is_empty() {
        r.bearer_auth(&p.key)
    } else {
        r
    }
}
pub async fn models(client: &reqwest::Client, p: &Provider) -> Result<Vec<Value>> {
    let r = request(client, p, reqwest::Method::GET, "/models")
        .send()
        .await?;
    if !r.status().is_success() {
        bail!("{}: HTTP {} while listing models", p.name, r.status());
    }
    let value: Value = r.json().await?;
    let mut models: Vec<Value> = value["data"].as_array().context("Provider returned no model catalog")?.iter()
        .filter(|m|m["id"].is_string()).map(|m|json!({"id":m["id"],"name":m["name"],"context_length":m["context_length"],"pricing":m["pricing"],"architecture":m["architecture"],"supported_parameters":m["supported_parameters"],"top_provider":m["top_provider"]})).collect();
    models.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    Ok(models)
}

pub fn body(
    p: &Provider,
    model: &str,
    system: &str,
    messages: &[Message],
    tools: &[Value],
    max_output_tokens: u32,
) -> Value {
    if p.anthropic {
        let mut wire: Vec<Value> = vec![];
        for m in messages {
            let (role, blocks) = match m.role.as_str() {
                "tool" => (
                    "user",
                    vec![
                        json!({"type":"tool_result", "tool_use_id":m.tool_call_id, "content":m.content}),
                    ],
                ),
                "assistant" => {
                    let mut b = vec![];
                    if !m.content.is_empty() {
                        b.push(json!({"type":"text","text":m.content}));
                    }
                    for c in &m.tool_calls {
                        b.push(json!({"type":"tool_use","id":c["id"],"name":c["function"]["name"],"input":serde_json::from_str::<Value>(c["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({}))}));
                    }
                    ("assistant", b)
                }
                _ => {
                    let mut b = vec![];
                    for i in &m.images {
                        b.push(json!({"type":"image","source":{"type":"base64","media_type":i.mime,"data":i.data}}));
                    }
                    if !m.content.is_empty() {
                        b.push(json!({"type":"text","text":m.content}));
                    }
                    ("user", b)
                }
            };
            if blocks.is_empty() {
                continue;
            }
            if let Some(last) = wire.last_mut().filter(|last| last["role"] == role) {
                last["content"].as_array_mut().unwrap().extend(blocks);
            } else {
                wire.push(json!({"role":role,"content":blocks}));
            }
        }
        let mut body = json!({"model":model,"system":system,"messages":wire,"max_tokens":max_output_tokens,"stream":true});
        if !tools.is_empty() {
            body["tools"] = json!(tools.iter().map(|t| json!({"name":t["function"]["name"],"description":t["function"]["description"],"input_schema":t["function"]["parameters"]})).collect::<Vec<_>>());
        }
        body
    } else {
        let mut wire = vec![json!({"role":"system","content":system})];
        for m in messages {
            if m.content.is_empty() && m.tool_calls.is_empty() && m.images.is_empty() {
                continue;
            }
            let mut item = serde_json::to_value(m).unwrap();
            item.as_object_mut().unwrap().remove("images");
            item.as_object_mut().unwrap().remove("truncated");
            item.as_object_mut()
                .unwrap()
                .remove("incomplete_tool_calls");
            item.as_object_mut().unwrap().remove("provider");
            item.as_object_mut().unwrap().remove("model");
            // Per-agent swarm reports are local transcript data, never a wire field.
            item.as_object_mut().unwrap().remove("swarm");
            let same_router = p.id == "openrouter"
                && m.provider.as_deref() == Some(p.id.as_str())
                && m.model.as_deref() == Some(model)
                && !m.truncated;
            if !same_router {
                item.as_object_mut().unwrap().remove("reasoning_details");
            } else if m.reasoning_details.is_empty() {
                if let Some(reasoning) = &m.reasoning_content {
                    item["reasoning"] = json!(reasoning);
                }
            }
            if !["deepseek", "kimi"].contains(&p.id.as_str()) || m.provider.as_deref() != Some(p.id.as_str()) {
                item.as_object_mut().unwrap().remove("reasoning_content");
            }
            if p.id != "gemini" || m.provider.as_deref() != Some(p.id.as_str()) {
                if let Some(calls) = item.get_mut("tool_calls").and_then(Value::as_array_mut) {
                    for c in calls {
                        c.as_object_mut().unwrap().remove("extra_content");
                    }
                }
            }
            if !m.images.is_empty() {
                let mut content = vec![json!({"type":"text","text":m.content})];
                for i in &m.images {
                    content.push(json!({"type":"image_url","image_url":{"url":format!("data:{};base64,{}",i.mime,i.data)}}));
                }
                item["content"] = json!(content);
            }
            wire.push(item);
        }
        let mut body = json!({"model":model,"messages":wire,"stream":true});
        // Avoid unsupported temperature/reasoning defaults across model families.
        if p.id == "openai" {
            body["max_completion_tokens"] = json!(max_output_tokens);
        } else {
            body["max_tokens"] = json!(max_output_tokens);
        }
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        if p.id == "deepseek" {
            body["stream_options"] = json!({"include_usage": true});
        }
        body
    }
}

#[derive(Default)]
pub struct Accumulator {
    pub text: String,
    pub reasoning: String,
    pub reasoning_details: Vec<Value>,
    reasoning_bytes: usize,
    pub calls: BTreeMap<usize, Value>,
    pub usage: Value,
    pub done: bool,
    pub finish: String,
}
impl Accumulator {
    pub fn event(&mut self, data: &str, anthropic: bool) -> Result<String> {
        if data == "[DONE]" {
            self.done = true;
            return Ok(String::new());
        }
        let v: Value = serde_json::from_str(data).context("Invalid provider stream JSON")?;
        if v.get("error").is_some() || v["type"] == "error" {
            bail!("Provider returned a streaming error");
        }
        let mut text = String::new();
        if anthropic {
            let index = v["index"].as_u64().unwrap_or(0) as usize;
            match v["type"].as_str().unwrap_or("") {
                "content_block_start" if v["content_block"]["type"] == "tool_use" => {
                    let b = &v["content_block"];
                    self.calls.insert(index,json!({"id":b["id"],"type":"function","function":{"name":b["name"],"arguments":""}}));
                }
                "content_block_start" if v["content_block"]["type"] == "text" => {
                    text = v["content_block"]["text"].as_str().unwrap_or("").into();
                }
                "content_block_delta" => match v["delta"]["type"].as_str().unwrap_or("") {
                    "text_delta" => text = v["delta"]["text"].as_str().unwrap_or("").into(),
                    "input_json_delta" => {
                        if let Some(c) = self.calls.get_mut(&index) {
                            append(&mut c["function"]["arguments"], &v["delta"]["partial_json"]);
                        }
                    }
                    "thinking_delta" => self
                        .reasoning
                        .push_str(v["delta"]["thinking"].as_str().unwrap_or("")),
                    _ => {}
                },
                "message_start" => self.usage = v["message"]["usage"].clone(),
                "message_delta" => {
                    self.finish = v["delta"]["stop_reason"].as_str().unwrap_or("").into();
                    if let Some(u) = v["usage"].as_object() {
                        for (k, val) in u {
                            self.usage[k] = val.clone();
                        }
                    }
                }
                "message_stop" => self.done = true,
                _ => {}
            }
        } else {
            if v.get("usage").is_some_and(|u| !u.is_null()) {
                self.usage = v["usage"].clone();
            }
            if let Some(choice) = v["choices"].as_array().and_then(|c| c.first()) {
                text = choice["delta"]["content"].as_str().unwrap_or("").into();
                self.reasoning.push_str(
                    choice["delta"]["reasoning_content"]
                        .as_str()
                        .or_else(|| choice["delta"]["reasoning"].as_str())
                        .unwrap_or(""),
                );
                if let Some(details) = choice["delta"]
                    .get("reasoning_details")
                    .filter(|v| !v.is_null())
                {
                    let details = details.as_array().context("Invalid reasoning_details")?;
                    self.reasoning_bytes += serde_json::to_vec(details)?.len();
                    if self.reasoning_bytes > 2_000_000
                        || self.reasoning_details.len() + details.len() > 16384
                    {
                        bail!("Reasoning output exceeded limit");
                    }
                    if details.iter().any(|d| !d.is_object()) {
                        bail!("Invalid reasoning detail");
                    }
                    self.reasoning_details.extend(details.iter().cloned());
                }
                if let Some(reason) = choice["finish_reason"].as_str() {
                    self.finish = reason.into();
                }
                if let Some(calls) = choice["delta"]["tool_calls"].as_array() {
                    for delta in calls {
                        let i =
                            delta["index"].as_u64().context("Tool delta has no index")? as usize;
                        let c = self.calls.entry(i).or_insert_with(|| json!({"id":"","type":"function","function":{"name":"","arguments":""}}));
                        append(&mut c["id"], &delta["id"]);
                        append(&mut c["function"]["name"], &delta["function"]["name"]);
                        append(
                            &mut c["function"]["arguments"],
                            &delta["function"]["arguments"],
                        );
                        if let Some(extra) = delta.get("extra_content") {
                            c["extra_content"] = extra.clone();
                        }
                    }
                }
            }
        }
        self.text.push_str(&text);
        if self.text.len() + self.reasoning.len() > 2_000_000
            || self.calls.len() > 64
            || self.calls.values().any(|v| v.to_string().len() > 1_000_000)
        {
            bail!("Provider output exceeded limit");
        }
        Ok(text)
    }
    pub fn finish(self) -> Result<(Message, Value)> {
        if !self.done {
            bail!("Provider stream disconnected before completion; partial text was retained");
        }
        if ["content_filter", "refusal"].contains(&self.finish.as_str()) {
            bail!("Provider stopped: {}", self.finish);
        }
        let truncated = ["length", "max_tokens"].contains(&self.finish.as_str());
        let mut calls: Vec<Value> = self.calls.into_values().collect();
        // Even syntactically complete calls from a truncated turn must not execute.
        let incomplete_tool_calls = if truncated {
            std::mem::take(&mut calls)
        } else {
            vec![]
        };
        for c in &mut calls {
            if c["id"].as_str().unwrap_or("").is_empty()
                || c["function"]["name"].as_str().unwrap_or("").is_empty()
            {
                bail!("Incomplete tool call");
            }
            if c["function"]["arguments"] == "" {
                c["function"]["arguments"] = json!("{}");
            }
            serde_json::from_str::<Value>(
                c["function"]["arguments"]
                    .as_str()
                    .context("Invalid tool arguments")?,
            )
            .context("Malformed tool arguments")?;
        }
        Ok((
            Message {
                role: "assistant".into(),
                truncated,
                incomplete_tool_calls,
                content: self.text,
                reasoning_details: self.reasoning_details,
                tool_calls: calls,
                reasoning_content: (!self.reasoning.is_empty()).then_some(self.reasoning),
                ..Message::default()
            },
            self.usage,
        ))
    }
}
fn append(target: &mut Value, delta: &Value) {
    if let Some(s) = delta.as_str() {
        *target = Value::String(format!("{}{s}", target.as_str().unwrap_or("")));
    }
}

pub async fn generate<F>(
    client: &reqwest::Client,
    provider: &Provider,
    settings: &Settings,
    system: &str,
    messages: &[Message],
    tools: &[Value],
    mut delta: F,
) -> Result<(Message, Value)>
where
    F: FnMut(String, String) + Send,
{
    let route = if provider.anthropic {
        "/messages"
    } else {
        "/chat/completions"
    };
    let mut request_body = body(
        provider,
        &settings.model,
        system,
        messages,
        tools,
        settings.max_output_tokens,
    );
    if settings.json_mode
        && provider.id == "deepseek"
        && !settings.model.starts_with("deepseek-reasoner")
    {
        request_body["response_format"] = json!({"type":"json_object"});
    }
    let response = request(client, provider, reqwest::Method::POST, route)
        .json(&request_body)
        .send()
        .await
        .context("Provider connection failed")?;
    if !response.status().is_success() {
        // Do not echo upstream bodies: a proxy may reflect credentials or request content.
        bail!(
            "{}: HTTP {}. Check API key, model access, quota and provider limits.",
            provider.name,
            response.status()
        );
    }
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    let mut acc = Accumulator::default();
    let mut event_data: Vec<String> = vec![];
    while let Some(chunk) = stream.next().await {
        buffer.extend_from_slice(&chunk?);
        if buffer.len() > 2_000_000 {
            bail!("Provider stream frame too large");
        }
        while let Some(end) = buffer.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = buffer.drain(..=end).collect();
            let line = std::str::from_utf8(&line)?.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                if !event_data.is_empty() {
                    let reasoning_start = acc.reasoning.len();
                    let text = acc.event(&event_data.join("\n"), provider.anthropic)?;
                    event_data.clear();
                    let reasoning = acc.reasoning[reasoning_start..].to_owned();
                    if !text.is_empty() || !reasoning.is_empty() {
                        delta(text, reasoning);
                    }
                    if acc.done {
                        return acc.finish();
                    }
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                if event_data.iter().map(String::len).sum::<usize>() + data.len() > 2_000_000 {
                    bail!("Provider event too large");
                }
                event_data.push(data.strip_prefix(' ').unwrap_or(data).to_owned());
            }
        }
    }
    acc.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fragmented_tool_arguments_and_reasoning_survive() {
        let mut a = Accumulator::default();
        a.event(r#"{"choices":[{"delta":{"reasoning_content":"reason", "tool_calls":[{"index":0,"id":"c1","function":{"name":"read_file","arguments":"{\"path\":"}}]}}]}"#,false).unwrap();
        a.event(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.txt\"}"},"extra_content":{"google":{"thought_signature":"sig"}}}]},"finish_reason":"tool_calls"}]}"#,false).unwrap();
        a.event("[DONE]", false).unwrap();
        let (m, _) = a.finish().unwrap();
        assert_eq!(
            m.tool_calls[0]["function"]["arguments"],
            r#"{"path":"a.txt"}"#
        );
        assert_eq!(
            m.tool_calls[0]["extra_content"]["google"]["thought_signature"],
            "sig"
        );
        assert_eq!(m.reasoning_content.as_deref(), Some("reason"));
    }
    #[test]
    fn router_reasoning_sequence_is_preserved_only_for_same_model() {
        let mut a = Accumulator::default();
        let first = json!({"type":"reasoning.text","index":0,"text":"first","signature":null});
        let second =
            json!({"type":"reasoning.encrypted","index":1,"data":"opaque","format":"unknown"});
        for block in [&first, &second] {
            a.event(
                &json!({"choices":[{"delta":{"reasoning_details":[block]}}]}).to_string(),
                false,
            )
            .unwrap();
        }
        a.event(r#"{"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}],"usage":{"cost":0.01}}"#,false).unwrap();
        a.event("[DONE]", false).unwrap();
        let (mut m, u) = a.finish().unwrap();
        m.provider = Some("openrouter".into());
        m.model = Some("test/model".into());
        assert_eq!(m.reasoning_details, vec![first, second]);
        assert_eq!(u["cost"], 0.01);
        let router = defaults()
            .into_iter()
            .find(|p| p.id == "openrouter")
            .unwrap();
        let same = body(&router, "test/model", "sys", &[m.clone()], &[], 1024);
        assert_eq!(
            same["messages"][1]["reasoning_details"],
            json!(m.reasoning_details)
        );
        assert!(same["messages"][1].get("model").is_none());
        let other = body(&router, "other/model", "sys", &[m.clone()], &[], 1024);
        assert!(other["messages"][1].get("reasoning_details").is_none());
        let openai = defaults().into_iter().find(|p| p.id == "openai").unwrap();
        assert!(
            body(&openai, "test/model", "sys", &[m.clone()], &[], 1024)["messages"][1]
                .get("reasoning_details")
                .is_none()
        );
        m.truncated = true;
        assert!(
            body(&router, "test/model", "sys", &[m], &[], 1024)["messages"][1]
                .get("reasoning_details")
                .is_none()
        );
    }
    #[test]
    fn router_request_uses_official_endpoint_and_bearer_auth() {
        let mut p = defaults()
            .into_iter()
            .find(|p| p.id == "openrouter")
            .unwrap();
        p.key = "test-secret".into();
        let r = request(
            &client().unwrap(),
            &p,
            reqwest::Method::POST,
            "/chat/completions",
        )
        .build()
        .unwrap();
        assert_eq!(
            r.url().as_str(),
            "https://openrouter.ai/api/v1/chat/completions"
        );
        assert_eq!(r.headers()["authorization"], "Bearer test-secret");
        assert_eq!(r.headers()["x-openrouter-title"], "allpaka Studio");
    }
    #[test]
    fn disconnected_stream_is_not_success() {
        let mut a = Accumulator::default();
        a.event(r#"{"choices":[{"delta":{"content":"partial"}}]}"#, false)
            .unwrap();
        assert!(a.finish().is_err());
    }
    #[test]
    fn claude_tools_convert_without_losing_result_link() {
        let p = defaults().into_iter().find(|p| p.anthropic).unwrap();
        let messages = vec![
            Message::text("user", "read"),
            Message {
                role: "assistant".into(),
                tool_calls: vec![
                    json!({"id":"c","function":{"name":"read_file","arguments":"{}"}}),
                ],
                ..Message::default()
            },
            Message {
                role: "tool".into(),
                tool_call_id: Some("c".into()),
                content: "ok".into(),
                ..Message::default()
            },
        ];
        let b = body(&p, "model", "sys", &messages, &[], 4096);
        assert_eq!(b["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(b["messages"][2]["content"][0]["tool_use_id"], "c");
        assert_eq!(b["system"], "sys");
    }
    #[test]
    fn native_image_blocks_are_not_plain_text() {
        let mut m = Message::text("user", "describe");
        m.images.push(crate::context::Image {
            name: "x.png".into(),
            mime: "image/png".into(),
            data: "iVBORw0KGgo=".into(),
        });
        let providers = defaults();
        let claude = body(
            providers.iter().find(|p| p.id == "anthropic").unwrap(),
            "model",
            "sys",
            &[m.clone()],
            &[],
            4096,
        );
        assert_eq!(claude["messages"][0]["content"][0]["type"], "image");
        assert_eq!(
            claude["messages"][0]["content"][0]["source"]["data"],
            "iVBORw0KGgo="
        );
        let openai = body(
            providers.iter().find(|p| p.id == "openai").unwrap(),
            "model",
            "sys",
            &[m],
            &[],
            4096,
        );
        assert_eq!(openai["messages"][1]["content"][1]["type"], "image_url");
        assert!(openai["messages"][1].get("images").is_none());
    }
    #[test]
    fn foreign_reasoning_is_not_sent_to_openai() {
        let m = Message {
            role: "assistant".into(),
            content: "done".into(),
            provider: Some("deepseek".into()),
            reasoning_content: Some("private reasoning".into()),
            ..Message::default()
        };
        let b = body(
            &defaults().into_iter().find(|p| p.id == "openai").unwrap(),
            "model",
            "sys",
            &[m],
            &[],
            4096,
        );
        assert!(b["messages"][1].get("reasoning_content").is_none());
        assert!(b["messages"][1].get("provider").is_none());
    }
    #[test]
    fn claude_stream_collects_tools_and_usage() {
        let mut a = Accumulator::default();
        for event in [
            json!({"type":"message_start","message":{"usage":{"input_tokens":20}}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"c1","name":"read_file","input":{}}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"workspace/a\"}"}}),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}),
            json!({"type":"message_stop"}),
        ] {
            a.event(&event.to_string(), true).unwrap();
        }
        let (m, u) = a.finish().unwrap();
        assert_eq!(
            m.tool_calls[0]["function"]["arguments"],
            r#"{"path":"workspace/a"}"#
        );
        assert_eq!(u["input_tokens"], 20);
        assert_eq!(u["output_tokens"], 5);
    }
    #[test]
    fn token_limit_preserves_text_reasoning_usage_without_executing_partial_tools() {
        for reason in ["length", "max_tokens"] {
            let mut a = Accumulator {
                done: true,
                finish: reason.into(),
                text: "partial answer".into(),
                reasoning: "reasoning".into(),
                usage: json!({"output_tokens":4096}),
                ..Accumulator::default()
            };
            a.calls.insert(
                0,
                json!({"id":"partial","function":{"name":"write_file","arguments":"{\"path\":"}}),
            );
            let (m, usage) = a.finish().unwrap();
            assert!(m.truncated);
            assert_eq!(m.content, "partial answer");
            assert_eq!(m.reasoning_content.as_deref(), Some("reasoning"));
            assert!(m.tool_calls.is_empty());
            assert_eq!(m.incomplete_tool_calls.len(), 1);
            assert_eq!(usage["output_tokens"], 4096);
            let b = body(
                &defaults().into_iter().find(|p| p.id == "openai").unwrap(),
                "model",
                "sys",
                &[m],
                &[],
                16384,
            );
            assert_eq!(b["max_completion_tokens"], 16384);
            assert!(b["messages"][1].get("incomplete_tool_calls").is_none());
            assert!(b["messages"][1].get("truncated").is_none());
            assert!(b["messages"][1].get("tool_calls").is_none());
        }
    }
    #[test]
    fn configurable_output_budget_reaches_each_protocol() {
        for p in defaults() {
            let b = body(
                &p,
                "model",
                "sys",
                &[Message::text("user", "hi")],
                &[],
                12288,
            );
            let field = if p.id == "openai" {
                "max_completion_tokens"
            } else {
                "max_tokens"
            };
            assert_eq!(b[field], 12288, "{}", p.id);
        }
    }
}
