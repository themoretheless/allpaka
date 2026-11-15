//! Keep a tool call the model is writing inside what the request declared.
//!
//! `tool_schema` refuses a finished call that does not fit; this decides,
//! byte by byte, whether a call still can. The sampler asks [`ToolGrammar`]
//! before committing a token, so inside a `<tool_call>` block the model can
//! only produce
//!
//! ```text
//! <tool_call>
//! {"name": "<a declared tool>", "arguments": <a value of its schema>}
//! </tool_call>
//! ```
//!
//! which is the Hermes form Qwen was trained on, with one fixed spelling of
//! the envelope. The arguments follow the same JSON Schema keywords
//! `tool_schema` checks (`type`, `properties`, `required`,
//! `additionalProperties: false`, `items`, and `enum` of strings); anything
//! else in a schema leaves that value free-form JSON.

use serde_json::Value;

const OPEN: &[u8] = b"<tool_call>";
/// Deeper nesting than this is not continued.
const MAX_DEPTH: usize = 32;
/// Unbroken whitespace a model may write between JSON tokens: enough for an
/// indented line, not enough to spend the token budget on spaces.
const MAX_SPACES: u8 = 24;

/// What the request said about calling tools.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ToolChoice {
    /// The model decides; a call it starts is held to the declarations.
    Auto,
    /// The reply must be exactly one call to any declared tool.
    Required,
    /// The reply must be exactly one call to this tool.
    Named(String),
}

impl ToolChoice {
    /// The OpenAI `tool_choice` field. Unknown shapes mean `auto`.
    pub(crate) fn parse(value: &Value) -> Self {
        match value {
            Value::String(s) if s == "required" => Self::Required,
            Value::Object(_) => match value["function"]["name"].as_str() {
                Some(name) => Self::Named(name.to_owned()),
                None => Self::Auto,
            },
            _ => Self::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum NumState {
    /// Just a minus sign.
    Minus,
    /// A leading zero: only a fraction or exponent may follow.
    Zero,
    Int,
    /// After the point, before any fraction digit.
    Point,
    Frac,
    /// After `e`/`E` and an optional sign, before any exponent digit.
    Exp {
        signed: bool,
    },
    ExpDigits,
}

impl NumState {
    /// Whether a delimiter may end the number here.
    fn complete(self) -> bool {
        matches!(self, Self::Zero | Self::Int | Self::Frac | Self::ExpDigits)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ObjState {
    /// After `{`: a key or `}`.
    First,
    /// After `,`: a key.
    Key,
    Colon,
    /// After a member's value: `,` or `}`.
    Next,
}

#[derive(Debug, Clone)]
enum Frame<'a> {
    /// These exact bytes.
    Fixed(&'static [u8]),
    /// A value of this schema (`None`: any JSON) is about to start.
    Value(Option<&'a Value>),
    /// A string that must be exactly one of `candidates`: an enum value, a key
    /// of a closed property list, or (`tool_name`) the name of the tool, whose
    /// closing quote belongs to the fixed text after it.
    Exact {
        candidates: Vec<&'a str>,
        written: Vec<u8>,
        tool_name: bool,
    },
    /// Any JSON string. `key` keeps the contents, to record an object key.
    Text {
        key: Option<Vec<u8>>,
        /// Pending escape: 0 none, 1 after a backslash, 2..=5 hex digits owed.
        escape: u8,
    },
    Num {
        state: NumState,
        integer: bool,
    },
    Obj {
        schema: Option<&'a Value>,
        state: ObjState,
        seen: Vec<String>,
    },
    /// `first`: right after `[`, where `]` or a value may come.
    Arr {
        items: Option<&'a Value>,
        first: bool,
    },
    /// The arguments object, whose schema the tool name decided.
    Arguments,
}

/// One `<tool_call>` block being written.
#[derive(Debug, Clone)]
struct Call<'a> {
    /// Frames still to satisfy, innermost last.
    stack: Vec<Frame<'a>>,
    tool: Option<&'a Value>,
    spaces: u8,
}

fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\n' | b'\r' | b'\t')
}

fn allows_type(schema: Option<&Value>, name: &str) -> bool {
    const KNOWN: [&str; 7] = [
        "object", "array", "string", "number", "integer", "boolean", "null",
    ];
    let fits = |t: &str| {
        t == name
            || (t == "number" && name == "integer")
            // An unknown type name constrains nothing.
            || !KNOWN.contains(&t)
    };
    match schema.and_then(|s| s.get("type")) {
        Some(Value::String(one)) => fits(one),
        Some(Value::Array(many)) => many.iter().filter_map(Value::as_str).any(fits),
        _ => true,
    }
}

/// The members of an `enum` whose members are all strings.
fn string_enum(schema: Option<&Value>) -> Option<Vec<&str>> {
    let members = schema?.get("enum")?.as_array()?;
    members.iter().map(Value::as_str).collect()
}

impl<'a> Call<'a> {
    /// A call to one of `names`; `opened` when `<tool_call>` is already written.
    fn new(names: Vec<&'a str>, opened: bool) -> Self {
        // In reverse: the last frame is the next to satisfy.
        let mut stack = vec![
            Frame::Fixed(b"\n</tool_call>"),
            Frame::Fixed(b"}"),
            Frame::Arguments,
            Frame::Fixed(b"\", \"arguments\": "),
            Frame::Exact {
                candidates: names,
                written: Vec::new(),
                tool_name: true,
            },
            Frame::Fixed(b"{\"name\": \""),
            Frame::Fixed(b"\n"),
        ];
        if !opened {
            stack.push(Frame::Fixed(OPEN));
        }
        Self {
            stack,
            tool: None,
            spaces: 0,
        }
    }

    fn done(&self) -> bool {
        self.stack.is_empty()
    }

    /// Push the frame of the value of `schema` that starts with `byte`.
    fn start_value(&mut self, schema: Option<&'a Value>, byte: u8) -> bool {
        if self.stack.len() >= MAX_DEPTH {
            return false;
        }
        let strings = string_enum(schema);
        // Under a string enum nothing but one of its members may be written.
        let open = strings.is_none();
        let frame = match byte {
            b'"' if strings.is_some() => Frame::Exact {
                candidates: strings.unwrap_or_default(),
                written: Vec::new(),
                tool_name: false,
            },
            b'"' if allows_type(schema, "string") => Frame::Text {
                key: None,
                escape: 0,
            },
            b'{' if open && allows_type(schema, "object") => Frame::Obj {
                schema,
                state: ObjState::First,
                seen: Vec::new(),
            },
            b'[' if open && allows_type(schema, "array") => Frame::Arr {
                items: schema.and_then(|s| s.get("items")),
                first: true,
            },
            b'-' | b'0'..=b'9' if open && allows_type(schema, "integer") => Frame::Num {
                state: match byte {
                    b'-' => NumState::Minus,
                    b'0' => NumState::Zero,
                    _ => NumState::Int,
                },
                integer: !allows_type(schema, "number"),
            },
            b't' if open && allows_type(schema, "boolean") => Frame::Fixed(b"rue"),
            b'f' if open && allows_type(schema, "boolean") => Frame::Fixed(b"alse"),
            b'n' if open && allows_type(schema, "null") => Frame::Fixed(b"ull"),
            _ => return false,
        };
        self.stack.push(frame);
        true
    }

    /// Advance by one byte; `false` when the call cannot continue with it.
    fn step(&mut self, tools: &'a [Value], byte: u8) -> bool {
        loop {
            let Some(top) = self.stack.pop() else {
                return false;
            };
            // Whitespace goes where JSON allows it: before a value and around
            // the punctuation of objects and arrays.
            let takes_space = matches!(
                top,
                Frame::Value(_) | Frame::Obj { .. } | Frame::Arr { .. } | Frame::Arguments
            );
            if takes_space && is_space(byte) {
                self.stack.push(top);
                self.spaces += 1;
                return self.spaces <= MAX_SPACES;
            }
            self.spaces = 0;
            match top {
                Frame::Fixed(text) => {
                    if text.first() != Some(&byte) {
                        return false;
                    }
                    if text.len() > 1 {
                        self.stack.push(Frame::Fixed(&text[1..]));
                    }
                }
                Frame::Value(schema) => {
                    if !self.start_value(schema, byte) {
                        return false;
                    }
                }
                Frame::Arguments => {
                    // Arguments are an object, whatever the schema says.
                    if byte != b'{' || self.stack.len() >= MAX_DEPTH {
                        return false;
                    }
                    let schema = self
                        .tool
                        .and_then(|tool| tool["function"].get("parameters"))
                        .filter(|schema| schema.is_object());
                    self.stack.push(Frame::Obj {
                        schema,
                        state: ObjState::First,
                        seen: Vec::new(),
                    });
                }
                Frame::Exact {
                    candidates,
                    mut written,
                    tool_name,
                } => {
                    if byte == b'"' {
                        // The closing quote, only once a candidate is whole.
                        if !candidates.iter().any(|c| c.as_bytes() == written) {
                            return false;
                        }
                        if tool_name {
                            self.tool = tools.iter().find(|tool| {
                                tool["function"]["name"]
                                    .as_str()
                                    .is_some_and(|name| name.as_bytes() == written)
                            });
                            // This quote opens the fixed text after the name.
                            continue;
                        }
                        // An object key is noted on its object; for an enum
                        // value there is no object waiting for a colon.
                        self.record_key(&written);
                    } else {
                        written.push(byte);
                        let candidates: Vec<&str> = candidates
                            .into_iter()
                            .filter(|c| c.as_bytes().starts_with(&written))
                            .collect();
                        if candidates.is_empty() {
                            return false;
                        }
                        self.stack.push(Frame::Exact {
                            candidates,
                            written,
                            tool_name,
                        });
                    }
                }
                Frame::Text { mut key, escape } => {
                    let escape = match (escape, byte) {
                        (0, b'"') => {
                            if let Some(key) = key {
                                self.record_key(&key);
                            }
                            return true;
                        }
                        (0, b'\\') => 1,
                        (0, 0x00..=0x1f) => return false,
                        (0, _) => 0,
                        (1, b'u') => 5,
                        (1, b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => 0,
                        (2, _) if byte.is_ascii_hexdigit() => 0,
                        (3..=5, _) if byte.is_ascii_hexdigit() => escape - 1,
                        _ => return false,
                    };
                    if let Some(key) = key.as_mut() {
                        key.push(byte);
                    }
                    self.stack.push(Frame::Text { key, escape });
                }
                Frame::Num { state, integer } => {
                    let next = match (state, byte) {
                        (NumState::Minus, b'0') => Some(NumState::Zero),
                        (NumState::Minus | NumState::Int, b'1'..=b'9') => Some(NumState::Int),
                        (NumState::Int, b'0') => Some(NumState::Int),
                        (NumState::Zero | NumState::Int, b'.') if !integer => Some(NumState::Point),
                        (NumState::Point | NumState::Frac, b'0'..=b'9') => Some(NumState::Frac),
                        (NumState::Zero | NumState::Int | NumState::Frac, b'e' | b'E')
                            if !integer =>
                        {
                            Some(NumState::Exp { signed: false })
                        }
                        (NumState::Exp { signed: false }, b'+' | b'-') => {
                            Some(NumState::Exp { signed: true })
                        }
                        (NumState::Exp { .. } | NumState::ExpDigits, b'0'..=b'9') => {
                            Some(NumState::ExpDigits)
                        }
                        _ => None,
                    };
                    match next {
                        Some(state) => self.stack.push(Frame::Num { state, integer }),
                        // Not part of the number: it ends here if it can, and
                        // the byte belongs to whatever encloses it.
                        None if state.complete() => continue,
                        None => return false,
                    }
                }
                Frame::Obj {
                    schema,
                    state,
                    seen,
                } => {
                    let properties = schema
                        .and_then(|s| s.get("properties"))
                        .and_then(Value::as_object);
                    let extra = schema.and_then(|s| s.get("additionalProperties"));
                    let required_met = schema
                        .and_then(|s| s.get("required"))
                        .and_then(Value::as_array)
                        .is_none_or(|required| {
                            required
                                .iter()
                                .filter_map(Value::as_str)
                                .all(|key| seen.iter().any(|s| s == key))
                        });
                    match (state, byte) {
                        (ObjState::First | ObjState::Next, b'}') if required_met => {}
                        (ObjState::First | ObjState::Key, b'"') => {
                            // With a closed property list the key is one of the
                            // properties not written yet; otherwise any key.
                            let key = match properties {
                                Some(properties) if extra == Some(&Value::Bool(false)) => {
                                    let left: Vec<&str> = properties
                                        .keys()
                                        .map(String::as_str)
                                        .filter(|key| !seen.iter().any(|s| s == key))
                                        .collect();
                                    if left.is_empty() {
                                        return false;
                                    }
                                    Frame::Exact {
                                        candidates: left,
                                        written: Vec::new(),
                                        tool_name: false,
                                    }
                                }
                                _ => Frame::Text {
                                    key: Some(Vec::new()),
                                    escape: 0,
                                },
                            };
                            self.stack.push(Frame::Obj {
                                schema,
                                state: ObjState::Colon,
                                seen,
                            });
                            self.stack.push(key);
                        }
                        (ObjState::Colon, b':') => {
                            let key = seen.last().map(String::as_str).unwrap_or_default();
                            let member = match properties.and_then(|p| p.get(key)) {
                                Some(sub) => Some(sub),
                                None => extra.filter(|extra| extra.is_object()),
                            };
                            self.stack.push(Frame::Obj {
                                schema,
                                state: ObjState::Next,
                                seen,
                            });
                            self.stack.push(Frame::Value(member));
                        }
                        (ObjState::Next, b',') => self.stack.push(Frame::Obj {
                            schema,
                            state: ObjState::Key,
                            seen,
                        }),
                        _ => return false,
                    }
                }
                Frame::Arr { items, first } => match byte {
                    b']' => {}
                    b',' if !first => {
                        self.stack.push(Frame::Arr {
                            items,
                            first: false,
                        });
                        self.stack.push(Frame::Value(items));
                    }
                    _ if first => {
                        self.stack.push(Frame::Arr {
                            items,
                            first: false,
                        });
                        if !self.start_value(items, byte) {
                            return false;
                        }
                    }
                    _ => return false,
                },
            }
            return true;
        }
    }

    /// Note `key` on the object waiting for its colon.
    fn record_key(&mut self, key: &[u8]) {
        if let Some(Frame::Obj {
            state: ObjState::Colon,
            seen,
            ..
        }) = self.stack.last_mut()
        {
            seen.push(String::from_utf8_lossy(key).into_owned());
        }
    }
}

/// Follows the reply as it is generated and says which continuations a tool
/// call allows.
#[derive(Debug, Clone)]
pub(crate) struct ToolGrammar<'a> {
    tools: &'a [Value],
    choice: ToolChoice,
    /// The block being written, if any.
    call: Option<Call<'a>>,
    /// The end of the free text, to notice `<tool_call>` across tokens.
    tail: Vec<u8>,
    /// Calls completed so far.
    finished: usize,
    /// Bytes were committed that this grammar would not have allowed: stop
    /// constraining rather than block the model on a guess.
    lost: bool,
}

impl<'a> ToolGrammar<'a> {
    pub(crate) fn new(tools: &'a [Value], choice: ToolChoice) -> Self {
        let mut grammar = Self {
            tools,
            choice,
            call: None,
            tail: Vec::new(),
            finished: 0,
            lost: false,
        };
        if grammar.choice != ToolChoice::Auto {
            grammar.call = Some(Call::new(grammar.names(), false));
        }
        grammar
    }

    fn names(&self) -> Vec<&'a str> {
        self.tools
            .iter()
            .filter_map(|tool| tool["function"]["name"].as_str())
            .filter(|name| match &self.choice {
                ToolChoice::Named(only) => name == only,
                _ => true,
            })
            .collect()
    }

    /// Whether the next token is under constraint at all.
    pub(crate) fn constraining(&self) -> bool {
        !self.lost && (self.call.is_some() || self.must_stop())
    }

    /// A forced call has been written: nothing but the end may follow.
    pub(crate) fn must_stop(&self) -> bool {
        !self.lost && self.choice != ToolChoice::Auto && self.finished > 0 && self.call.is_none()
    }

    /// Whether the generation may end here.
    pub(crate) fn may_stop(&self) -> bool {
        self.lost
            || match self.choice {
                ToolChoice::Auto => self.call.is_none(),
                _ => self.must_stop(),
            }
    }

    /// Whether these bytes may come next.
    pub(crate) fn allows(&self, bytes: &[u8]) -> bool {
        if self.lost {
            return true;
        }
        if self.must_stop() {
            return false;
        }
        let Some(call) = &self.call else {
            return true;
        };
        // A token that says nothing cannot advance a call.
        if bytes.is_empty() {
            return false;
        }
        let mut probe = call.clone();
        for &byte in bytes {
            if probe.done() {
                // The call closed inside this token; what follows it is free
                // text again, or nothing at all when the call was forced.
                return self.choice == ToolChoice::Auto;
            }
            if !probe.step(self.tools, byte) {
                return false;
            }
        }
        true
    }

    /// Record bytes the model has committed to.
    pub(crate) fn feed(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            if self.lost {
                return;
            }
            match self.call.take() {
                Some(mut call) => {
                    if !call.step(self.tools, byte) {
                        self.lost = true;
                    } else if call.done() {
                        self.finished += 1;
                    } else {
                        self.call = Some(call);
                    }
                }
                None if self.choice == ToolChoice::Auto => {
                    self.tail.push(byte);
                    if self.tail.len() > OPEN.len() {
                        self.tail.remove(0);
                    }
                    if self.tail == OPEN {
                        self.call = Some(Call::new(self.names(), true));
                        self.tail.clear();
                    }
                }
                // Text after a forced call: not something the sampler chose.
                None => self.lost = true,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Vec<Value> {
        vec![
            json!({"type": "function", "function": {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "mode": {"enum": ["exact", "fuzzy"]},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "score": {"type": "number"},
                        "deep": {"type": "boolean"}
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }
            }}),
            json!({"type": "function", "function": {"name": "ping"}}),
        ]
    }

    /// Offer `text` byte by byte; the index of the first byte the grammar
    /// refuses, or `None` when all of it is allowed.
    fn first_refusal(grammar: &mut ToolGrammar<'_>, text: &str) -> Option<usize> {
        for (i, byte) in text.bytes().enumerate() {
            if !grammar.allows(&[byte]) {
                return Some(i);
            }
            grammar.feed(&[byte]);
        }
        None
    }

    fn call(body: &str) -> String {
        format!("<tool_call>\n{body}\n</tool_call>")
    }

    #[test]
    fn a_call_that_fits_is_allowed_to_the_end() {
        let t = tools();
        for body in [
            r#"{"name": "search", "arguments": {"query": "rust"}}"#,
            r#"{"name": "search", "arguments": {"query": "a \"b\" é \\", "limit": -12, "mode": "fuzzy"}}"#,
            r#"{"name": "search", "arguments": {"tags": ["a", "b"], "query": "x", "score": 1.5e-3, "deep": true}}"#,
            "{\"name\": \"search\", \"arguments\": {\n    \"query\": \"indented\",\n    \"tags\": []\n}}",
            r#"{"name": "search", "arguments": {"query": "x", "limit": 0, "score": -0.25E+2}}"#,
            r#"{"name": "ping", "arguments": {}}"#,
            r#"{"name": "ping", "arguments": {"anything": [1, {"x": null}], "y": false}}"#,
        ] {
            let mut grammar = ToolGrammar::new(&t, ToolChoice::Required);
            assert_eq!(first_refusal(&mut grammar, &call(body)), None, "{body}");
            assert!(grammar.must_stop() && grammar.may_stop(), "{body}");
            assert!(!grammar.allows(b"x"), "nothing follows a forced call");
            // The finished call passes the after-the-fact check too.
            let parsed: Value = serde_json::from_str(body).unwrap();
            let name = parsed["name"].as_str().unwrap();
            assert_eq!(
                crate::serve::tool_schema::check_call(&t, name, &parsed["arguments"]),
                Ok(()),
                "{body}"
            );
        }
    }

    #[test]
    fn a_call_that_cannot_fit_is_stopped_at_the_byte_that_breaks_it() {
        let t = tools();
        // (body, the text right before the refused byte)
        let cases = [
            (r#"{"name": "delete", "arguments": {}}"#, r#"{"name": ""#),
            (r#"{"name": "sea", "arguments": {}}"#, r#"{"name": "sea"#),
            (
                r#"{"name": "search", "arguments": {}}"#,
                r#""arguments": {"#,
            ),
            (r#"{"name": "search", "arguments": {"q": 1}}"#, r#"{"q"#),
            (
                r#"{"name": "search", "arguments": {"query": 7}}"#,
                r#""query": "#,
            ),
            (
                r#"{"name": "search", "arguments": {"query": "x", "limit": 2.5}}"#,
                r#""limit": 2"#,
            ),
            (
                r#"{"name": "search", "arguments": {"query": "x", "mode": "regex"}}"#,
                r#""mode": ""#,
            ),
            (
                r#"{"name": "search", "arguments": {"query": "x", "mode": 3}}"#,
                r#""mode": "#,
            ),
            (
                r#"{"name": "search", "arguments": {"query": "x", "tags": ["a", 1]}}"#,
                r#"["a", "#,
            ),
            (
                r#"{"name": "search", "arguments": {"query": "x", "tags": ["a",]}}"#,
                r#"["a","#,
            ),
            (
                r#"{"name": "search", "arguments": {"query": "x", "query": "y"}}"#,
                r#""x", ""#,
            ),
            (
                r#"{"name": "search", "arguments": {"query": "x",}}"#,
                r#""x","#,
            ),
            (
                r#"{"name": "search", "arguments": {"query": "x", "deep": "yes"}}"#,
                r#""deep": "#,
            ),
            (
                r#"{"name": "search", "arguments": ["query"]}"#,
                r#""arguments": "#,
            ),
            (
                "{\"name\": \"search\", \"arguments\": {\"query\": \"tab\tin\"}}",
                "\"tab",
            ),
            (
                r#"{"name": "search", "arguments": {"query": "bad \x escape"}}"#,
                r#"bad \"#,
            ),
            (
                r#"{"name": "search", "arguments": {"query": "x", "limit": 01}}"#,
                r#""limit": 0"#,
            ),
            (
                r#"{"name": "search", "arguments": {"query": "x", "score": 1.}}"#,
                r#""score": 1."#,
            ),
        ];
        for (body, before) in cases {
            let mut grammar = ToolGrammar::new(&t, ToolChoice::Required);
            let text = call(body);
            let at =
                first_refusal(&mut grammar, &text).unwrap_or_else(|| panic!("allowed: {body}"));
            assert!(
                text[..at].ends_with(before),
                "{body}: refused after {:?}",
                &text[..at]
            );
            assert!(
                !grammar.may_stop(),
                "a forced call cannot end unfinished: {body}"
            );
        }
    }

    #[test]
    fn auto_leaves_text_alone_and_takes_over_inside_a_block() {
        let t = tools();
        let mut grammar = ToolGrammar::new(&t, ToolChoice::Auto);
        assert!(!grammar.constraining());
        assert_eq!(
            first_refusal(&mut grammar, "Sure, {not json} at all. "),
            None
        );
        assert!(grammar.may_stop());
        // The tag arrives split across tokens.
        grammar.feed(b"<tool_");
        grammar.feed(b"call>");
        assert!(grammar.constraining() && !grammar.may_stop());
        assert!(!grammar.allows(b"{"), "the envelope starts on a new line");
        assert!(grammar.allows(b"\n{\"name\": \"se"));
        assert!(!grammar.allows(b"\n{\"name\": \"x"));
        let rest = "\n{\"name\": \"ping\", \"arguments\": {}}\n</tool_call>";
        assert_eq!(first_refusal(&mut grammar, rest), None);
        // Free text again, and a second call is held to the rules as well.
        assert!(!grammar.constraining() && grammar.may_stop());
        assert_eq!(first_refusal(&mut grammar, " Then: <tool_call>"), None);
        assert!(!grammar.allows(b"\n{\"name\": \"nope"));
    }

    #[test]
    fn a_named_choice_allows_only_that_tool() {
        let t = tools();
        let mut grammar = ToolGrammar::new(&t, ToolChoice::Named("ping".into()));
        assert!(grammar.allows(b"<tool_call>\n{\"name\": \"ping\""));
        assert!(!grammar.allows(b"<tool_call>\n{\"name\": \"search\""));
        assert!(
            !grammar.allows(b"Hello"),
            "a forced call starts at the first byte"
        );
        assert_eq!(
            first_refusal(&mut grammar, &call(r#"{"name": "ping", "arguments": {}}"#)),
            None
        );
        assert_eq!(ToolChoice::parse(&json!("required")), ToolChoice::Required);
        assert_eq!(
            ToolChoice::parse(&json!({"type": "function", "function": {"name": "ping"}})),
            ToolChoice::Named("ping".into())
        );
        for auto in [
            json!("auto"),
            json!(null),
            json!("none"),
            json!({"type": "function"}),
        ] {
            assert_eq!(ToolChoice::parse(&auto), ToolChoice::Auto);
        }
    }

    #[test]
    fn tokens_are_judged_whole_and_runaway_output_is_cut() {
        let t = tools();
        let mut grammar = ToolGrammar::new(&t, ToolChoice::Required);
        grammar.feed(b"<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query\": \"x\"");
        // One token may close the object, the envelope and the tag.
        assert!(grammar.allows(b"}}\n</tool_call>"));
        assert!(!grammar.allows(b"}}\n</tool_call> and more"));
        assert!(!grammar.allows(b""), "an empty token advances nothing");
        // Whitespace cannot be used to run out the token budget.
        assert!(grammar.allows(&[b' '; 24]));
        assert!(!grammar.allows(&[b' '; 25]));
        // Nesting has a floor to stand on.
        let mut deep = ToolGrammar::new(&t, ToolChoice::Named("ping".into()));
        deep.feed(b"<tool_call>\n{\"name\": \"ping\", \"arguments\": {\"a\": ");
        assert!(deep.allows(&[b'['; 20]));
        assert!(!deep.allows(&[b'['; 40]));
        // Bytes committed behind the grammar's back end the constraint
        // instead of deadlocking the sampler.
        let mut lost = ToolGrammar::new(&t, ToolChoice::Required);
        lost.feed(b"plain text");
        assert!(!lost.constraining() && lost.allows(b"anything") && lost.may_stop());
    }
}
