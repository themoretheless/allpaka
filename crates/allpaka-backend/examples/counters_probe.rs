// Metal-only example; the body lives in metal/counters_probe.rs so other platforms
// still build the workspace's examples.
#[cfg(target_os = "macos")]
include!("metal/counters_probe.rs");

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("counters_probe needs Metal and runs only on macOS");
}
