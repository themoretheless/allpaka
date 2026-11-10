// Metal-only example; the body lives in metal/tensor_test.rs so other platforms
// still build the workspace's examples.
#[cfg(target_os = "macos")]
include!("metal/tensor_test.rs");

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("tensor_test needs Metal and runs only on macOS");
}
