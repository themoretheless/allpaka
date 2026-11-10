//! GGUF selective mappings on a file written by the shared test builder.

mod common;

use allpaka_gguf::GgufFile;
use common::{write_temp, FileBuilder};

#[test]
fn selective_gpu_mappings_cover_chosen_tensor_without_whole_file() {
    let bytes = FileBuilder::new()
        .str_kv("general.architecture", "llama")
        .tensor("unused-before", &[40000], vec![0.0; 40000])
        .tensor("chosen", &[32], vec![0.5; 32])
        .tensor("unused-after", &[40000], vec![0.0; 40000])
        .build();
    let path = write_temp("selected-mapping", &bytes);
    let file = GgufFile::open(&path).unwrap();
    let mappings = file
        .selected_mappings(|tensor| tensor.name == "chosen")
        .unwrap();
    assert_eq!(mappings.len(), 1);
    assert!(mappings[0].len() <= 32768);
    assert!(mappings[0].len() < file.mapping().len() / 2);
    let data = file.data(file.tensor("chosen").unwrap()).unwrap();
    let start = mappings[0].as_ptr() as usize;
    assert!(data.as_ptr() as usize >= start);
    assert!(data.as_ptr() as usize + data.len() <= start + mappings[0].len());
    assert_eq!((start - file.mapping().as_ptr() as usize) % 16384, 0);
    assert!(file.selected_mappings(|_| false).unwrap().is_empty());
    std::fs::remove_file(path).ok();
}
