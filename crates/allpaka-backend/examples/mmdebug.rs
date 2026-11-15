// Metal-only example; the body lives in metal/mmdebug.rs so other platforms
// still build the workspace's examples.
#[cfg(target_os = "macos")]
include!("metal/mmdebug.rs");

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("mmdebug needs Metal and runs only on macOS");
}
