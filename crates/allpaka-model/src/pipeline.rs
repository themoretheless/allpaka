//! Stateful, bounded TCP execution of contiguous transformer stages.
//! A connection owns its KV state. A failed exchange invalidates the pipeline;
//! callers must start a fresh connection and replay their prompt.
use crate::{Model, Session};
use anyhow::{bail, Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    io::{Read, Write},
    net::{TcpStream, ToSocketAddrs},
    time::Duration,
};

const MAX_FRAME: usize = 64 * 1024 * 1024;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageInfo {
    pub model_id: String,
    pub fingerprint: String,
    pub first: usize,
    pub end: usize,
    pub layers: usize,
    pub hidden: usize,
    pub vocab: u32,
}
#[derive(Serialize, Deserialize)]
enum Request {
    Open { key: String, context: usize },
    Token { position: usize, token: u32 },
    Hidden { position: usize, values: Vec<f32> },
}
#[derive(Debug, Serialize, Deserialize)]
enum Response {
    Info(StageInfo),
    Values(Vec<f32>),
    Error(String),
}
fn send<T: Serialize>(stream: &mut TcpStream, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > MAX_FRAME {
        bail!("pipeline frame too large");
    }
    stream.write_all(&(bytes.len() as u32).to_be_bytes())?;
    stream.write_all(&bytes)?;
    Ok(())
}
fn receive<T: DeserializeOwned>(stream: &mut TcpStream) -> Result<T> {
    let mut header = [0; 4];
    stream.read_exact(&mut header)?;
    let size = u32::from_be_bytes(header) as usize;
    if size == 0 || size > MAX_FRAME {
        bail!("invalid pipeline frame length {size}");
    }
    let mut bytes = vec![0; size];
    stream.read_exact(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
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
        let result: Result<Vec<f32>> = (|| {
            let (position, hidden) = match request {
                Request::Token { position, token } if info.first == 0 => {
                    (position, model.embed_token(token)?)
                }
                Request::Hidden { position, values } if info.first > 0 => (position, values),
                _ => bail!("unexpected pipeline input"),
            };
            if position != session.pos() || position >= context {
                bail!("invalid pipeline position {position}");
            }
            let hidden = model.forward_layers(&hidden, &mut session, info.first, info.end)?;
            if info.end == info.layers {
                model.finish_hidden(&hidden)
            } else {
                Ok(hidden)
            }
        })();
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
    fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        let mut request = Request::Token {
            position: self.position,
            token,
        };
        let mut values = Vec::new();
        for (index, stream) in self.streams.iter_mut().enumerate() {
            send(stream, &request)?;
            values = match receive::<Response>(stream)? {
                Response::Values(values) => values,
                Response::Error(error) => bail!("pipeline stage {index}: {error}"),
                _ => bail!("unexpected pipeline response"),
            };
            let expected = if index + 1 == self.infos.len() {
                self.infos[0].vocab as usize
            } else {
                self.infos[0].hidden
            };
            if values.len() != expected || values.iter().any(|v| !v.is_finite()) {
                bail!("invalid stage output");
            }
            request = Request::Hidden {
                position: self.position,
                values: values.clone(),
            };
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
