use allpaka_gguf::GgufFile;
fn main() {
    let f = GgufFile::open(std::path::Path::new(&std::env::args().nth(1).unwrap())).unwrap();
    let arg = std::env::args().nth(2).unwrap();
    for t in f.tensors() {
        if t.name.starts_with(&arg) { println!("{} {:?} {:?}", t.name, t.ggml_type, t.dims); }
    }
}
