//! Check a model's tool call against the tools the request declared.
//!
//! A call that names an undeclared tool, or whose arguments do not fit the
//! tool's `parameters` schema, is not handed to the client as a tool call: the
//! model wrote something the caller never asked to execute. The supported
//! JSON Schema keywords are the ones tool declarations use in practice:
//! `type`, `properties`, `required`, `additionalProperties: false`, `items`
//! and `enum`. Any other keyword is ignored, never treated as a failure.

use serde_json::Value;

/// Nesting beyond this is refused rather than walked.
const MAX_DEPTH: usize = 32;

/// `Ok` when `name` is one of `tools` and `arguments` fits its schema;
/// otherwise the reason, phrased for the caller.
pub(crate) fn check_call(tools: &[Value], name: &str, arguments: &Value) -> Result<(), String> {
    let Some(tool) = tools
        .iter()
        .find(|tool| tool["function"]["name"].as_str() == Some(name))
    else {
        return Err(format!("tool `{name}` was not declared in this request"));
    };
    // Models sometimes emit the arguments as a JSON string, the shape the
    // OpenAI response uses; accept it when the string itself is an object.
    let decoded;
    let arguments = match arguments {
        Value::String(text) => {
            decoded = serde_json::from_str::<Value>(text)
                .map_err(|_| "arguments are a string that is not JSON".to_string())?;
            &decoded
        }
        other => other,
    };
    if !arguments.is_object() {
        return Err("arguments are not a JSON object".to_string());
    }
    let schema = &tool["function"]["parameters"];
    if schema.is_null() {
        return Ok(());
    }
    check(schema, arguments, "arguments", 0)
}

fn type_matches(expected: &str, value: &Value) -> bool {
    match expected {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "number" => value.is_number(),
        // 3.0 is an integer in JSON Schema; 3.5 is not.
        "integer" => {
            value.is_i64() || value.is_u64() || value.as_f64().is_some_and(|f| f.fract() == 0.0)
        }
        // An unknown type name cannot be checked, so it does not fail the call.
        _ => true,
    }
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn check(schema: &Value, value: &Value, path: &str, depth: usize) -> Result<(), String> {
    if depth > MAX_DEPTH {
        return Err(format!("{path}: nested too deeply to check"));
    }
    let Some(schema) = schema.as_object() else {
        // `true`, or anything that is not a schema object, accepts everything.
        return if schema == &Value::Bool(false) {
            Err(format!("{path}: no value is allowed here"))
        } else {
            Ok(())
        };
    };

    let fits = match schema.get("type") {
        Some(Value::String(expected)) => type_matches(expected, value),
        Some(Value::Array(any)) => any
            .iter()
            .any(|expected| expected.as_str().is_none_or(|e| type_matches(e, value))),
        _ => true,
    };
    if !fits {
        return Err(format!(
            "{path}: expected {}, got {}",
            schema["type"],
            type_name(value)
        ));
    }

    if let Some(allowed) = schema.get("enum").and_then(Value::as_array) {
        if !allowed.contains(value) {
            return Err(format!("{path}: {value} is not one of the allowed values"));
        }
    }

    if let Some(object) = value.as_object() {
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for key in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(key) {
                    return Err(format!("{path}: missing required `{key}`"));
                }
            }
        }
        let properties = schema.get("properties").and_then(Value::as_object);
        for (key, item) in object {
            match properties.and_then(|p| p.get(key)) {
                Some(sub) => check(sub, item, &format!("{path}.{key}"), depth + 1)?,
                None => match schema.get("additionalProperties") {
                    Some(Value::Bool(false)) => {
                        return Err(format!("{path}: unexpected `{key}`"));
                    }
                    Some(extra @ Value::Object(_)) => {
                        check(extra, item, &format!("{path}.{key}"), depth + 1)?
                    }
                    _ => {}
                },
            }
        }
    }

    if let (Some(items), Some(array)) = (schema.get("items"), value.as_array()) {
        for (index, item) in array.iter().enumerate() {
            check(items, item, &format!("{path}[{index}]"), depth + 1)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_call;
    use serde_json::{json, Value};

    fn tools() -> Vec<Value> {
        vec![json!({
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "mode": {"enum": ["exact", "fuzzy"]},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "filter": {
                            "type": "object",
                            "properties": {"lang": {"type": ["string", "null"]}},
                            "additionalProperties": false
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }
            }
        })]
    }

    #[test]
    fn accepts_calls_that_fit_the_schema() {
        let t = tools();
        for args in [
            json!({"query": "rust"}),
            json!({"query": "rust", "limit": 3, "mode": "fuzzy", "tags": ["a", "b"]}),
            json!({"query": "rust", "limit": 3.0, "filter": {"lang": null}}),
            // The OpenAI response shape: arguments as a JSON string.
            json!("{\"query\": \"rust\"}"),
        ] {
            assert_eq!(check_call(&t, "search", &args), Ok(()), "{args}");
        }
    }

    #[test]
    fn refuses_what_the_caller_did_not_declare() {
        let t = tools();
        let cases = [
            ("delete_everything", json!({"query": "x"}), "not declared"),
            ("search", json!({}), "missing required `query`"),
            ("search", json!({"query": 7}), "arguments.query: expected"),
            (
                "search",
                json!({"query": "x", "limit": 2.5}),
                "arguments.limit",
            ),
            (
                "search",
                json!({"query": "x", "mode": "regex"}),
                "allowed values",
            ),
            (
                "search",
                json!({"query": "x", "tags": ["a", 1]}),
                "arguments.tags[1]",
            ),
            (
                "search",
                json!({"query": "x", "shell": "rm"}),
                "unexpected `shell`",
            ),
            (
                "search",
                json!({"query": "x", "filter": {"lang": "en", "x": 1}}),
                "arguments.filter: unexpected `x`",
            ),
            ("search", json!(["query"]), "not a JSON object"),
            ("search", json!("not json at all"), "not JSON"),
            ("search", json!("[1, 2]"), "not a JSON object"),
        ];
        for (name, args, expected) in cases {
            let reason = check_call(&t, name, &args).expect_err(&format!("{name} {args}"));
            assert!(reason.contains(expected), "{args}: {reason}");
        }
    }

    #[test]
    fn a_tool_without_a_schema_takes_any_object() {
        let t = vec![json!({"type": "function", "function": {"name": "ping"}})];
        assert_eq!(check_call(&t, "ping", &json!({"anything": [1, 2]})), Ok(()));
        assert!(check_call(&t, "ping", &json!(3)).is_err());
    }

    #[test]
    fn unknown_keywords_and_types_do_not_fail_a_call() {
        let t = vec![
            json!({"type": "function", "function": {"name": "f", "parameters": {
                "type": "object",
                "properties": {"a": {"type": "decimal", "pattern": "^x", "minimum": 5}},
            }}}),
        ];
        assert_eq!(check_call(&t, "f", &json!({"a": 1, "b": "extra"})), Ok(()));
    }

    #[test]
    fn runaway_nesting_is_refused_not_walked() {
        let mut schema = json!({"type": "array"});
        let mut value = json!([]);
        for _ in 0..40 {
            schema = json!({"type": "array", "items": schema});
            value = json!([value]);
        }
        let t = vec![
            json!({"type": "function", "function": {"name": "f", "parameters": {
                "type": "object", "properties": {"deep": schema},
            }}}),
        ];
        let reason = check_call(&t, "f", &json!({"deep": value})).unwrap_err();
        assert!(reason.contains("too deeply"), "{reason}");
    }
}
