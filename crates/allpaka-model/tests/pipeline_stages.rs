//! Layer-range stages, locally and over TCP, against the whole model.

mod common;

use allpaka_gguf::GgufFile;
use allpaka_model::{pipeline, Model};
use common::{tiny_model, write_temp, HIDDEN, VOCAB};

#[test]
fn stage_loader_does_not_require_other_layers_or_endpoint_tensors() {
    let original = tiny_model("llama", false);
    // Rename tensors without changing GGUF offsets. The head must load
    // even when the other layer and output head are absent from metadata.
    let mut bytes = original.clone();
    for index in 0..bytes.len() - 5 {
        if &bytes[index..index + 5] == b"blk.1" && bytes.get(index + 6..index + 9) == Some(b"ffn") {
            bytes[index + 4] = b'9';
        }
    }
    for index in 0..bytes.len() - 6 {
        if &bytes[index..index + 6] == b"output" && (index == 0 || bytes[index - 1] != b'_') {
            bytes[index] = b'X';
        }
    }
    let path = write_temp("head-only", &bytes);
    let file = GgufFile::open(&path).unwrap();
    assert!(Model::load(&file).is_err());
    let head = Model::load_stage(&file, 0, 1).unwrap();
    let mut session = head.new_session(16);
    let h = head.embed_token(3).unwrap();
    assert!(head.forward_layers(&h, &mut session, 0, 1).is_ok());
    assert!(head.finish_hidden(&h).is_err());
    assert!(head.forward_layers(&h, &mut session, 1, 2).is_err());
    std::fs::remove_file(path).ok();

    let mut bytes = original;
    for index in 0..bytes.len() - 5 {
        if &bytes[index..index + 5] == b"blk.0" && bytes.get(index + 6..index + 9) == Some(b"ffn") {
            bytes[index + 4] = b'9';
        }
    }
    let path = write_temp("tail-only", &bytes);
    let file = GgufFile::open(&path).unwrap();
    let tail = Model::load_stage(&file, 1, 2).unwrap();
    assert!(tail.embed_token(3).is_err());
    let mut session = tail.new_session(16);
    let h = tail
        .forward_layers(&vec![0.01; HIDDEN as usize], &mut session, 1, 2)
        .unwrap();
    assert!(tail.finish_hidden(&h).is_ok());
    std::fs::remove_file(path).ok();
}

#[test]
fn tcp_pipeline_matches_local_model_and_rejects_wrong_key() {
    use std::net::TcpListener;
    let path = write_temp("tcp-pipeline", &tiny_model("llama", false));
    let mut handles = Vec::new();
    let mut addresses = Vec::new();
    for first in 0..2 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        addresses.push(listener.local_addr().unwrap().to_string());
        let path = path.clone();
        handles.push(std::thread::spawn(move || {
            let file = GgufFile::open(&path).unwrap();
            let model = Model::load_stage(&file, first, first + 1).unwrap();
            let info = pipeline::StageInfo {
                model_id: "tiny-llama-v1".into(),
                fingerprint: allpaka_gguf::fingerprint(&path).unwrap(),
                first,
                end: first + 1,
                layers: 2,
                hidden: HIDDEN as usize,
                vocab: VOCAB as u32,
            };
            // The head first receives a deliberately unauthorised connection.
            if first == 0 {
                let (stream, _) = listener.accept().unwrap();
                assert!(
                    pipeline::serve_connection(stream, &model, info.clone(), "test-key", 16)
                        .is_err()
                );
            }
            let (stream, _) = listener.accept().unwrap();
            let _ = pipeline::serve_connection(stream, &model, info, "test-key", 16);
        }));
    }
    assert!(pipeline::Pipeline::connect(&addresses, "wrong-key", 16).is_err());
    let mut remote = pipeline::Pipeline::connect(&addresses, "test-key", 16).unwrap();
    let file = GgufFile::open(&path).unwrap();
    let model = Model::load(&file).unwrap();
    let mut session = model.new_session(16);
    for token in [3, 1, 4, 2] {
        let expected = model.forward(token, &mut session).unwrap();
        let actual = remote.forward(token).unwrap();
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }
    drop(remote);
    for handle in handles {
        handle.join().unwrap();
    }
    std::fs::remove_file(path).ok();
}

#[test]
fn pipeline_stages_match_whole_model_across_tokens() {
    let path = write_temp("pipeline-stages", &tiny_model("llama", false));
    let file = GgufFile::open(&path).unwrap();
    let model = Model::load(&file).unwrap();
    let mut whole = model.new_session(16);
    let mut head = model.new_session(16);
    let mut tail = model.new_session(16);
    for token in [3, 1, 4, 2] {
        let expected = model.forward(token, &mut whole).unwrap();
        let h = model.embed_token(token).unwrap();
        let h = model.forward_layers(&h, &mut head, 0, 1).unwrap();
        let h = model
            .forward_layers(&h, &mut tail, 1, model.config.n_layers as usize)
            .unwrap();
        let actual = model.finish_hidden(&h).unwrap();
        for (a, b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
        assert_eq!(head.pos(), whole.pos());
        assert_eq!(tail.pos(), whole.pos());
    }
    let pos = head.pos();
    assert!(model.forward_layers(&[], &mut head, 0, 1).is_err());
    assert!(model
        .forward_layers(&vec![0.0; HIDDEN as usize], &mut head, 1, 1)
        .is_err());
    assert_eq!(head.pos(), pos);
    std::fs::remove_file(path).ok();
}
