// Metal-only example; the body lives in metal/mmbench.rs so other platforms
// still build the workspace's examples.
#[cfg(target_os = "macos")]
include!("metal/mmbench.rs");

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("mmbench needs Metal and runs only on macOS");
}
