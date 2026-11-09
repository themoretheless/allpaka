//! Stateful, bounded TCP execution of contiguous transformer stages.
//! A connection owns its KV state. A failed exchange invalidates the pipeline;
//! callers must start a fresh connection and replay their prompt.
use crate::{Model, Session};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    io::{Read, Write},
    net::{TcpStream, ToSocketAddrs},
    time::Duration,
};

const MAX_FRAME: usize = 64 * 1024 * 1024;
const WIRE_MAGIC: [u8; 4] = *b"ALP\x01";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageInfo {
    pub model_id: String,
    pub fingerprint: String,
    pub first: usize,
    pub end: usize,
    pub layers: usize,
    pub hidden: usize,
    pub vocab: u32,
}

#[derive(Debug, Clone, PartialEq)]
enum Request {
    Open {
        key: String,
        context: usize,
    },
    Token {
        position: usize,
        token: u32,
    },
    Hidden {
        position: usize,
        values: Vec<f32>,
    },
    TokenBatch {
        position: usize,
        tokens: Vec<u32>,
    },
    HiddenBatch {
        position: usize,
        m: usize,
        values: Vec<f32>,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum Response {
    Info(StageInfo),
    Values(Vec<f32>),
    Error(String),
}

trait WireFrame: Sized {
    fn encode(&self, out: &mut Vec<u8>);
    fn decode(bytes: &[u8]) -> Result<Self>;
}

fn write_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_str(out: &mut Vec<u8>, s: &str) {
    write_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

fn write_u32_slice(out: &mut Vec<u8>, values: &[u32]) {
    write_u32(out, values.len() as u32);
    out.reserve(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn write_f32_slice(out: &mut Vec<u8>, values: &[f32]) {
    write_u32(out, values.len() as u32);
    out.reserve(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn read_u8(cursor: &mut &[u8]) -> Result<u8> {
    let (&b, rest) = cursor
        .split_first()
        .context("unexpected EOF in pipeline frame")?;
    *cursor = rest;
    Ok(b)
}

fn read_u32(cursor: &mut &[u8]) -> Result<u32> {
    if cursor.len() < 4 {
        bail!("unexpected EOF reading u32 in pipeline frame");
    }
    let (head, rest) = cursor.split_at(4);
    *cursor = rest;
    Ok(u32::from_le_bytes([head[0], head[1], head[2], head[3]]))
}

fn read_u64(cursor: &mut &[u8]) -> Result<u64> {
    if cursor.len() < 8 {
        bail!("unexpected EOF reading u64 in pipeline frame");
    }
    let (head, rest) = cursor.split_at(8);
    *cursor = rest;
    Ok(u64::from_le_bytes([
        head[0], head[1], head[2], head[3], head[4], head[5], head[6], head[7],
    ]))
}

fn read_str(cursor: &mut &[u8]) -> Result<String> {
    let len = read_u32(cursor)? as usize;
    if cursor.len() < len {
        bail!("unexpected EOF reading string in pipeline frame");
    }
    let (raw, rest) = cursor.split_at(len);
    *cursor = rest;
    Ok(std::str::from_utf8(raw)
        .context("invalid UTF-8 in pipeline frame")?
        .to_owned())
}

fn read_u32_vec(cursor: &mut &[u8]) -> Result<Vec<u32>> {
    let count = read_u32(cursor)? as usize;
    let byte_len = count.checked_mul(4).context("u32 slice length overflow")?;
    if cursor.len() < byte_len {
        bail!("unexpected EOF reading u32 slice in pipeline frame");
    }
    let (raw, rest) = cursor.split_at(byte_len);
    *cursor = rest;
    let mut out = Vec::with_capacity(count);
    for chunk in raw.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn read_f32_vec(cursor: &mut &[u8]) -> Result<Vec<f32>> {
    let count = read_f32_count(cursor)?;
    let byte_len = count.checked_mul(4).context("f32 slice length overflow")?;
    if cursor.len() < byte_len {
        bail!("unexpected EOF reading f32 slice in pipeline frame");
    }
    let (raw, rest) = cursor.split_at(byte_len);
    *cursor = rest;
    let mut out = Vec::with_capacity(count);
    for chunk in raw.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

#[inline]
fn read_f32_count(cursor: &mut &[u8]) -> Result<usize> {
    Ok(read_u32(cursor)? as usize)
}

fn check_magic(cursor: &mut &[u8]) -> Result<()> {
    if cursor.len() < 4 || cursor[..4] != WIRE_MAGIC {
        bail!("invalid pipeline wire magic or version");
    }
    *cursor = &cursor[4..];
    Ok(())
}

impl WireFrame for Request {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&WIRE_MAGIC);
        match self {
            Request::Open { key, context } => {
                write_u8(out, 1);
                write_u64(out, *context as u64);
                write_str(out, key);
            }
            Request::Token { position, token } => {
                write_u8(out, 2);
                write_u64(out, *position as u64);
                write_u32(out, *token);
            }
            Request::Hidden { position, values } => {
                write_u8(out, 3);
                write_u64(out, *position as u64);
                write_f32_slice(out, values);
            }
            Request::TokenBatch { position, tokens } => {
                write_u8(out, 4);
                write_u64(out, *position as u64);
                write_u32_slice(out, tokens);
            }
            Request::HiddenBatch {
                position,
                m,
                values,
            } => {
                write_u8(out, 5);
                write_u64(out, *position as u64);
                write_u32(out, *m as u32);
                write_f32_slice(out, values);
            }
        }
    }

    fn decode(mut bytes: &[u8]) -> Result<Self> {
        check_magic(&mut bytes)?;
        let tag = read_u8(&mut bytes)?;
        let req = match tag {
            1 => {
                let context = read_u64(&mut bytes)? as usize;
                let key = read_str(&mut bytes)?;
                Request::Open { key, context }
            }
            2 => {
                let position = read_u64(&mut bytes)? as usize;
                let token = read_u32(&mut bytes)?;
                Request::Token { position, token }
            }
            3 => {
                let position = read_u64(&mut bytes)? as usize;
                let values = read_f32_vec(&mut bytes)?;
                Request::Hidden { position, values }
            }
            4 => {
                let position = read_u64(&mut bytes)? as usize;
                let tokens = read_u32_vec(&mut bytes)?;
                Request::TokenBatch { position, tokens }
            }
            5 => {
                let position = read_u64(&mut bytes)? as usize;
                let m = read_u32(&mut bytes)? as usize;
                let values = read_f32_vec(&mut bytes)?;
                Request::HiddenBatch {
                    position,
                    m,
                    values,
                }
            }
            other => bail!("unknown pipeline request tag {other}"),
        };
        if !bytes.is_empty() {
            bail!("trailing bytes in pipeline request frame");
        }
        Ok(req)
    }
}

impl WireFrame for Response {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&WIRE_MAGIC);
        match self {
            Response::Info(info) => {
                write_u8(out, 1);
                write_str(out, &info.model_id);
                write_str(out, &info.fingerprint);
                write_u64(out, info.first as u64);
                write_u64(out, info.end as u64);
                write_u64(out, info.layers as u64);
                write_u64(out, info.hidden as u64);
                write_u32(out, info.vocab);
            }
            Response::Values(values) => {
                write_u8(out, 2);
                write_f32_slice(out, values);
            }
            Response::Error(err) => {
                write_u8(out, 3);
                write_str(out, err);
            }
        }
    }

    fn decode(mut bytes: &[u8]) -> Result<Self> {
        check_magic(&mut bytes)?;
        let tag = read_u8(&mut bytes)?;
        let resp = match tag {
            1 => {
                let model_id = read_str(&mut bytes)?;
                let fingerprint = read_str(&mut bytes)?;
                let first = read_u64(&mut bytes)? as usize;
                let end = read_u64(&mut bytes)? as usize;
                let layers = read_u64(&mut bytes)? as usize;
                let hidden = read_u64(&mut bytes)? as usize;
                let vocab = read_u32(&mut bytes)?;
                Response::Info(StageInfo {
                    model_id,
                    fingerprint,
                    first,
                    end,
                    layers,
                    hidden,
                    vocab,
                })
            }
            2 => Response::Values(read_f32_vec(&mut bytes)?),
            3 => Response::Error(read_str(&mut bytes)?),
            other => bail!("unknown pipeline response tag {other}"),
        };
        if !bytes.is_empty() {
            bail!("trailing bytes in pipeline response frame");
        }
        Ok(resp)
    }
}

fn send<T: WireFrame>(stream: &mut TcpStream, value: &T) -> Result<()> {
    let mut frame = Vec::with_capacity(256);
    frame.extend_from_slice(&[0u8; 4]);
    value.encode(&mut frame);
    let payload_len = frame.len() - 4;
    if payload_len > MAX_FRAME {
        bail!("pipeline frame too large");
    }
    frame[..4].copy_from_slice(&(payload_len as u32).to_be_bytes());
    stream.write_all(&frame)?;
    Ok(())
}

fn receive<T: WireFrame>(stream: &mut TcpStream) -> Result<T> {
    let mut header = [0; 4];
    stream.read_exact(&mut header)?;
    let size = u32::from_be_bytes(header) as usize;
    if size == 0 || size > MAX_FRAME {
        bail!("invalid pipeline frame length {size}");
    }
    let mut bytes = vec![0; size];
    stream.read_exact(&mut bytes)?;
    T::decode(&bytes)
}

fn configure(stream: &TcpStream) -> Result<()> {
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(Duration::from_secs(120)))?;
    stream.set_write_timeout(Some(Duration::from_secs(120)))?;
    Ok(())
}

/// Serve a single connection; the caller limits concurrent connections.
pub fn serve_connection(
    mut stream: TcpStream,
    model: &Model<'_>,
    info: StageInfo,
    key: &str,
    max_context: usize,
) -> Result<()> {
    configure(&stream)?;
    if info.first >= info.end
        || info.end > model.config.n_layers as usize
        || info.hidden != model.config.hidden as usize
        || info.layers != model.config.n_layers as usize
        || info.vocab != model.config.vocab
    {
        bail!("stage metadata does not match model");
    }
    let context = match receive::<Request>(&mut stream)? {
        Request::Open {
            key: supplied,
            context,
        } if supplied == key && context > 0 && context <= max_context => context,
        _ => {
            send(
                &mut stream,
                &Response::Error("invalid credentials or context".into()),
            )?;
            bail!("pipeline open rejected");
        }
    };
    let mut session: Session = model.new_session(context);
    send(&mut stream, &Response::Info(info.clone()))?;
    loop {
        let request = receive::<Request>(&mut stream)?;
        let result: Result<Vec<f32>> = match request {
            Request::Token { position, token } if info.first == 0 => (|| {
                if position != session.pos() || position >= context {
                    bail!("invalid pipeline position {position}");
                }
                let hidden = model.embed_token(token)?;
                let hidden = model.forward_layers(&hidden, &mut session, info.first, info.end)?;
                if info.end == info.layers {
                    model.finish_hidden(&hidden)
                } else {
                    Ok(hidden)
                }
            })(),
            Request::Hidden { position, values } if info.first > 0 => (|| {
                if position != session.pos() || position >= context {
                    bail!("invalid pipeline position {position}");
                }
                let hidden = model.forward_layers(&values, &mut session, info.first, info.end)?;
                if info.end == info.layers {
                    model.finish_hidden(&hidden)
                } else {
                    Ok(hidden)
                }
            })(),
            Request::TokenBatch { position, tokens } if info.first == 0 => (|| {
                let m = tokens.len();
                if m == 0
                    || position != session.pos()
                    || position.checked_add(m).is_none_or(|end| end > context)
                {
                    bail!("invalid pipeline batch position {position} or count {m}");
                }
                let is_last_stage = info.end == info.layers;
                let mut out = if is_last_stage {
                    Vec::new()
                } else {
                    Vec::with_capacity(m * info.hidden)
                };
                let mut last_hidden = Vec::new();
                for (i, &token) in tokens.iter().enumerate() {
                    let h = model.embed_token(token)?;
                    let h = model.forward_layers(&h, &mut session, info.first, info.end)?;
                    if is_last_stage {
                        if i + 1 == m {
                            last_hidden = h;
                        }
                    } else {
                        out.extend_from_slice(&h);
                    }
                }
                if is_last_stage {
                    model.finish_hidden(&last_hidden)
                } else {
                    Ok(out)
                }
            })(),
            Request::HiddenBatch {
                position,
                m,
                values,
            } if info.first > 0 => (|| {
                if m == 0
                    || values.len() != m.saturating_mul(info.hidden)
                    || position != session.pos()
                    || position.checked_add(m).is_none_or(|end| end > context)
                {
                    bail!("invalid pipeline hidden batch at position {position}");
                }
                let is_last_stage = info.end == info.layers;
                let mut out = if is_last_stage {
                    Vec::new()
                } else {
                    Vec::with_capacity(m * info.hidden)
                };
                let mut last_hidden = Vec::new();
                for (i, row) in values.chunks_exact(info.hidden).enumerate() {
                    let h = model.forward_layers(row, &mut session, info.first, info.end)?;
                    if is_last_stage {
                        if i + 1 == m {
                            last_hidden = h;
                        }
                    } else {
                        out.extend_from_slice(&h);
                    }
                }
                if is_last_stage {
                    model.finish_hidden(&last_hidden)
                } else {
                    Ok(out)
                }
            })(),
            _ => Err(anyhow::anyhow!("unexpected pipeline input")),
        };
        match result {
            Ok(values) => send(&mut stream, &Response::Values(values))?,
            Err(error) => {
                send(&mut stream, &Response::Error(error.to_string()))?;
                return Err(error);
            }
        }
    }
}

pub struct Pipeline {
    streams: Vec<TcpStream>,
    infos: Vec<StageInfo>,
    position: usize,
    context: usize,
    failed: bool,
}

impl Pipeline {
    /// Addresses are ordered from embedding stage to output stage.
    pub fn connect(addresses: &[String], key: &str, context: usize) -> Result<Self> {
        if addresses.is_empty() || context == 0 {
            bail!("pipeline needs stages and a positive context");
        }
        let mut streams = Vec::new();
        let mut infos: Vec<StageInfo> = Vec::new();
        for address in addresses {
            let addr = address
                .to_socket_addrs()?
                .next()
                .context("empty stage address")?;
            let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(10))?;
            configure(&stream)?;
            send(
                &mut stream,
                &Request::Open {
                    key: key.into(),
                    context,
                },
            )?;
            let info = match receive::<Response>(&mut stream)? {
                Response::Info(info) => info,
                Response::Error(error) => bail!("stage {address}: {error}"),
                _ => bail!("stage did not return metadata"),
            };
            let first = infos.last().map_or(0, |previous| previous.end);
            if info.first != first || info.end <= info.first || info.end > info.layers {
                bail!("pipeline has a gap, overlap or invalid range");
            }
            if let Some(head) = infos.first() {
                if head.model_id != info.model_id
                    || head.fingerprint != info.fingerprint
                    || head.layers != info.layers
                    || head.hidden != info.hidden
                    || head.vocab != info.vocab
                {
                    bail!("pipeline stages use different models");
                }
            }
            infos.push(info);
            streams.push(stream);
        }
        if infos.last().unwrap().end != infos[0].layers {
            bail!("pipeline does not cover all layers");
        }
        Ok(Self {
            streams,
            infos,
            position: 0,
            context,
            failed: false,
        })
    }

    pub fn info(&self) -> &StageInfo {
        &self.infos[0]
    }

    pub fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        if self.failed {
            bail!("pipeline invalidated; reconnect and replay prompt");
        }
        if self.position >= self.context || token >= self.infos[0].vocab {
            bail!("pipeline context exhausted or token out of range");
        }
        let result = self.step(token);
        match result {
            Ok(values) => {
                self.position += 1;
                Ok(values)
            }
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }

    /// Consume a sequence of tokens in batched network chunks and return the
    /// final token's vocabulary logits. Intermediate tokens skip `lm_head` at
    /// the tail stage and share one TCP round trip per chunk per stage.
    pub fn forward_batch(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if self.failed {
            bail!("pipeline invalidated; reconnect and replay prompt");
        }
        if tokens.is_empty()
            || self
                .position
                .checked_add(tokens.len())
                .is_none_or(|end| end > self.context)
            || tokens.iter().any(|&t| t >= self.infos[0].vocab)
        {
            bail!("pipeline context exhausted or token out of range");
        }
        if tokens.len() == 1 {
            return self.forward(tokens[0]);
        }
        let chunk_size = Model::prefill_chunk().max(1);
        let mut logits = Vec::new();
        for chunk in tokens.chunks(chunk_size) {
            if chunk.len() == 1 {
                logits = self.forward(chunk[0])?;
            } else {
                match self.step_batch(chunk) {
                    Ok(values) => {
                        self.position += chunk.len();
                        logits = values;
                    }
                    Err(error) => {
                        self.failed = true;
                        return Err(error);
                    }
                }
            }
        }
        Ok(logits)
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        let mut request = Request::Token {
            position: self.position,
            token,
        };
        let mut values = Vec::new();
        let num_stages = self.streams.len();
        for (index, stream) in self.streams.iter_mut().enumerate() {
            send(stream, &request)?;
            values = match receive::<Response>(stream)? {
                Response::Values(values) => values,
                Response::Error(error) => bail!("pipeline stage {index}: {error}"),
                _ => bail!("unexpected pipeline response"),
            };
            let is_last = index + 1 == num_stages;
            let expected = if is_last {
                self.infos[0].vocab as usize
            } else {
                self.infos[0].hidden
            };
            if values.len() != expected || values.iter().any(|v| !v.is_finite()) {
                bail!("invalid stage output");
            }
            if !is_last {
                request = Request::Hidden {
                    position: self.position,
                    values: std::mem::take(&mut values),
                };
            }
        }
        Ok(values)
    }

    fn step_batch(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let m = tokens.len();
        let mut request = Request::TokenBatch {
            position: self.position,
            tokens: tokens.to_vec(),
        };
        let mut values = Vec::new();
        let num_stages = self.streams.len();
        for (index, stream) in self.streams.iter_mut().enumerate() {
            send(stream, &request)?;
            values = match receive::<Response>(stream)? {
                Response::Values(values) => values,
                Response::Error(error) => bail!("pipeline stage {index}: {error}"),
                _ => bail!("unexpected pipeline response"),
            };
            let is_last = index + 1 == num_stages;
            let expected = if is_last {
                self.infos[0].vocab as usize
            } else {
                m * self.infos[0].hidden
            };
            if values.len() != expected || values.iter().any(|v| !v.is_finite()) {
                bail!("invalid stage output");
            }
            if !is_last {
                request = Request::HiddenBatch {
                    position: self.position,
                    m,
                    values: std::mem::take(&mut values),
                };
            }
        }
        Ok(values)
    }
}

#[cfg(test)]
mod protocol_tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread::JoinHandle;

    fn info(first: usize, end: usize) -> StageInfo {
        StageInfo {
            model_id: "fixture".into(),
            fingerprint: "same-weights".into(),
            first,
            end,
            layers: 2,
            hidden: 4,
            vocab: 8,
        }
    }

    fn worker(info: StageInfo, replies: Vec<Response>) -> (String, JoinHandle<usize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            configure(&stream).unwrap();
            assert!(matches!(
                receive::<Request>(&mut stream).unwrap(),
                Request::Open { .. }
            ));
            send(&mut stream, &Response::Info(info)).unwrap();
            let mut consumed = 0;
            for reply in replies {
                if receive::<Request>(&mut stream).is_err() {
                    break;
                }
                consumed += 1;
                if send(&mut stream, &reply).is_err() {
                    break;
                }
            }
            consumed
        });
        (address, handle)
    }

    #[test]
    fn binary_wire_protocol_round_trips_requests_and_responses() {
        let requests = [
            Request::Open {
                key: "secret-key".into(),
                context: 4096,
            },
            Request::Token {
                position: 7,
                token: 42,
            },
            Request::Hidden {
                position: 7,
                values: vec![1.25, -0.5, 0.0, 3.125],
            },
            Request::TokenBatch {
                position: 0,
                tokens: vec![1, 2, 3, 4],
            },
            Request::HiddenBatch {
                position: 0,
                m: 2,
                values: vec![0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
            },
        ];
        for req in requests {
            let mut buf = Vec::new();
            req.encode(&mut buf);
            assert_eq!(Request::decode(&buf).unwrap(), req);
        }

        let responses = [
            Response::Info(info(0, 2)),
            Response::Values(vec![0.5, -1.5, 2.25]),
            Response::Error("boom".into()),
        ];
        for resp in responses {
            let mut buf = Vec::new();
            resp.encode(&mut buf);
            assert_eq!(Response::decode(&buf).unwrap(), resp);
        }
    }

    #[test]
    fn forward_batch_sends_multi_token_chunk_in_single_pass() {
        let (head, h) = worker(info(0, 1), vec![Response::Values(vec![0.25; 3 * 4])]);
        let (tail, t) = worker(info(1, 2), vec![Response::Values(vec![0.5; 8])]);
        let mut pipeline = Pipeline::connect(&[head, tail], "key", 8).unwrap();
        let logits = pipeline.forward_batch(&[1, 2, 3]).unwrap();
        assert_eq!(logits.len(), 8);
        assert_eq!(pipeline.position, 3);
        drop(pipeline);
        assert_eq!(h.join().unwrap(), 1);
        assert_eq!(t.join().unwrap(), 1);
    }

    #[test]
    fn a_failed_tail_invalidates_the_already_advanced_head() {
        let (head, h) = worker(info(0, 1), vec![Response::Values(vec![0.1; 4])]);
        let (tail, t) = worker(
            info(1, 2),
            vec![Response::Error("worker lost its cache".into())],
        );
        let mut pipeline = Pipeline::connect(&[head, tail], "key", 4).unwrap();
        assert!(pipeline
            .forward(1)
            .unwrap_err()
            .to_string()
            .contains("lost its cache"));
        assert!(pipeline
            .forward(1)
            .unwrap_err()
            .to_string()
            .contains("invalidated"));
        assert_eq!(pipeline.position, 0);
        drop(pipeline);
        assert_eq!(h.join().unwrap(), 1);
        assert_eq!(t.join().unwrap(), 1);
    }
    #[test]
    fn a_dropped_worker_invalidates_the_session() {
        let (address, worker) = worker(info(0, 2), Vec::new());
        let mut pipeline = Pipeline::connect(&[address], "key", 4).unwrap();
        assert!(pipeline.forward(1).is_err());
        assert!(pipeline
            .forward(2)
            .unwrap_err()
            .to_string()
            .contains("invalidated"));
        drop(pipeline);
        assert_eq!(worker.join().unwrap(), 0);
    }
    #[test]
    fn context_and_token_rejections_leave_the_session_usable() {
        let (address, worker) = worker(info(0, 2), vec![Response::Values(vec![0.1; 8])]);
        let mut pipeline = Pipeline::connect(&[address], "key", 1).unwrap();
        assert!(pipeline.forward(8).is_err());
        assert!(!pipeline.failed);
        assert!(pipeline.forward(1).is_ok());
        assert!(pipeline.forward(2).is_err());
        assert!(!pipeline.failed);
        assert_eq!(pipeline.position, 1);
        drop(pipeline);
        assert_eq!(worker.join().unwrap(), 1);
    }
    #[test]
    fn gaps_overlaps_and_changed_weights_are_rejected_before_tokens() {
        for (first, end, fingerprint) in [
            (0, 2, "same-weights"),
            (2, 3, "same-weights"),
            (1, 2, "changed-weights"),
        ] {
            let (head, h) = worker(info(0, 1), Vec::new());
            let mut metadata = info(first, end);
            metadata.fingerprint = fingerprint.into();
            let (tail, t) = worker(metadata, Vec::new());
            assert!(Pipeline::connect(&[head, tail], "key", 4).is_err());
            assert_eq!(h.join().unwrap(), 0);
            assert_eq!(t.join().unwrap(), 0);
        }
    }
    #[test]
    fn malformed_stage_output_invalidates_the_session() {
        let (address, worker) = worker(info(0, 2), vec![Response::Values(vec![0.1; 7])]);
        let mut pipeline = Pipeline::connect(&[address], "key", 4).unwrap();
        assert!(pipeline
            .forward(1)
            .unwrap_err()
            .to_string()
            .contains("invalid stage output"));
        assert!(pipeline.failed);
        drop(pipeline);
        assert_eq!(worker.join().unwrap(), 1);
    }
    #[test]
    fn oversized_wire_frames_are_rejected_before_allocating_payload() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let writer = std::thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .write_all(&((MAX_FRAME + 1) as u32).to_be_bytes())
                .unwrap();
        });
        let (mut stream, _) = listener.accept().unwrap();
        assert!(receive::<Response>(&mut stream)
            .unwrap_err()
            .to_string()
            .contains("frame length"));
        writer.join().unwrap();
    }
}
