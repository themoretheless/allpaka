//! Routing pinned at the socket: which stage answers a given request line,
//! with which bytes. Requests are written to a real loopback connection and
//! the response is read back whole, so status line, headers and body are
//! compared exactly.
//!
//! The model-stage routes (`/stats`, the chat page, `OPTIONS`, `/v1/prompt/*`,
//! `/v1/chat/completions`) need a loaded model to be answered; with an empty
//! runtime they must all fall through to the unknown-model 404, which is what
//! these tests check for them.

use super::*;
use std::io::Read;
use std::time::Duration;

const DEFAULT_MODEL: &str = "default";

fn connected_pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    (client, server)
}

fn read_all(mut client: TcpStream) -> String {
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    response
}

fn json_response(status: u16, body: &str) -> String {
    let reason = if status == 200 { "OK" } else { "Error" };
    format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n\
         Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// What the reader stage did with a request.
enum Prepared {
    /// Answered before the queue; the full response bytes.
    Answered(String),
    /// Handed on to the scheduler, with the client end still open.
    Queued(TcpStream, String, QueuedConnection),
}

fn prepare(line: &str, body: &str, registry: &Arc<RequestRegistry>) -> Prepared {
    let (mut client, server) = connected_pair();
    write!(
        client,
        "{line}\r\nHost: test\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    match prepare_connection(server, DEFAULT_MODEL, 1 << 15, registry).unwrap() {
        None => Prepared::Answered(read_all(client)),
        Some((model, _, connection)) => Prepared::Queued(client, model, connection),
    }
}

fn answered(line: &str, registry: &Arc<RequestRegistry>) -> String {
    match prepare(line, "", registry) {
        Prepared::Answered(response) => response,
        Prepared::Queued(..) => panic!("{line:?} was queued, expected a reader-stage answer"),
    }
}

fn assert_queued(line: &str, registry: &Arc<RequestRegistry>) {
    if let Prepared::Answered(response) = prepare(line, "", registry) {
        panic!("{line:?} was answered by the reader stage: {response:?}");
    }
}

/// Runs a request through the reader stage and then the dispatcher against a
/// runtime with no models installed.
fn dispatched(line: &str, body: &str) -> String {
    let registry = Arc::new(RequestRegistry::default());
    let Prepared::Queued(client, model, connection) = prepare(line, body, &registry) else {
        panic!("{line:?} was answered by the reader stage");
    };
    let mut runtime = ServerRuntime::new(ServingLimits::default());
    let rag = RagTools::grep_only();
    dispatch_connection(model, connection, &mut runtime, &rag).unwrap();
    read_all(client)
}

#[test]
fn health_answers_only_the_exact_request_line() {
    let registry = Arc::new(RequestRegistry::default());
    assert_eq!(
        answered("GET /health HTTP/1.1", &registry),
        json_response(200, r#"{"status":"ok"}"#)
    );
    for line in [
        "GET /healthz HTTP/1.1",
        "GET /health?probe=1 HTTP/1.1",
        "GET /health HTTP/1.0",
        "POST /health HTTP/1.1",
    ] {
        assert_queued(line, &registry);
    }
}

#[test]
fn resources_needs_the_path_followed_by_a_space() {
    let registry = Arc::new(RequestRegistry::default());
    for line in ["GET /resources HTTP/1.1", "GET /resources HTTP/1.0"] {
        let response = answered(line, &registry);
        assert!(
            response.starts_with("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"),
            "{line:?}: {response:?}"
        );
    }
    for line in [
        "GET /resources?since=1 HTTP/1.1",
        "GET /resourcesx HTTP/1.1",
        "POST /resources HTTP/1.1",
    ] {
        assert_queued(line, &registry);
    }
}

#[test]
fn cancel_matches_prefix_and_exact_suffix_and_reports_the_outcome() {
    let registry = Arc::new(RequestRegistry::default());
    assert_eq!(
        answered("POST /v1/requests/missing/cancel HTTP/1.1", &registry),
        json_response(404, r#"{"cancelled":false}"#)
    );
    let _control = registry
        .register(Some("live"), Duration::from_secs(60))
        .unwrap();
    assert_eq!(
        answered("POST /v1/requests/live/cancel HTTP/1.1", &registry),
        json_response(200, r#"{"cancelled":true}"#)
    );
    // The id is everything between the prefix and the suffix, slashes included.
    assert_eq!(
        answered("POST /v1/requests/a/b/cancel HTTP/1.1", &registry),
        json_response(404, r#"{"cancelled":false}"#)
    );
    for line in [
        "POST /v1/requests/live/cancel HTTP/1.0",
        "POST /v1/requests/live HTTP/1.1",
        "GET /v1/requests/live/cancel HTTP/1.1",
    ] {
        assert_queued(line, &registry);
    }
}

#[test]
fn models_list_is_a_prefix_route_answered_before_model_resolution() {
    let empty = json_response(200, r#"{"data":[],"object":"list"}"#);
    for line in [
        "GET /v1/models HTTP/1.1",
        "GET /v1/models/qwen3 HTTP/1.1",
        "GET /v1/modelsx HTTP/1.1",
    ] {
        assert_eq!(dispatched(line, ""), empty, "{line:?}");
    }
}

#[test]
fn catalog_is_a_prefix_route_answered_before_model_resolution() {
    for line in ["GET /v1/catalog HTTP/1.1", "GET /v1/catalog?all=1 HTTP/1.1"] {
        let response = dispatched(line, "");
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        assert!(
            head.starts_with("HTTP/1.1 200 OK\r\n"),
            "{line:?}: {head:?}"
        );
        assert!(body.starts_with(r#"{"data":["#), "{body:?}");
        assert!(body.ends_with(r#"],"object":"list"}"#), "{body:?}");
    }
}

#[test]
fn launch_rejects_a_missing_path_before_model_resolution() {
    assert_eq!(
        dispatched("POST /v1/launch HTTP/1.1", "{}"),
        json_response(
            400,
            r#"{"error":"model path is empty or not safe to relaunch"}"#
        )
    );
}

#[test]
fn every_other_line_needs_a_resolved_model_first() {
    let unknown = json_response(404, r#"{"error":"unknown model: default"}"#);
    for line in [
        // Model-stage routes.
        "GET /stats HTTP/1.1",
        "GET /stats?ts=1 HTTP/1.1",
        "GET / HTTP/1.1",
        "GET /index.html HTTP/1.1",
        "OPTIONS /v1/chat/completions HTTP/1.1",
        "POST /v1/prompt/predict HTTP/1.1",
        "POST /v1/prompt/prewarm HTTP/1.1",
        "POST /v1/chat/completions HTTP/1.1",
        // Unknown paths and wrong methods for known paths.
        "GET /nope HTTP/1.1",
        "GET /healthz HTTP/1.1",
        "GET /v1/chat/completions HTTP/1.1",
        "POST /v1/models HTTP/1.1",
        "GET /v1/launch HTTP/1.1",
        "DELETE /stats HTTP/1.1",
        "POST /v1/requests/x HTTP/1.1",
    ] {
        assert_eq!(dispatched(line, ""), unknown, "{line:?}");
    }
}
