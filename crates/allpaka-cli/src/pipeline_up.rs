//! Start the stages and gateway a `pipeline-plan` manifest describes.
//!
//! `allpaka pipeline-up manifest.json --node <name>` runs on each machine and
//! starts the stages the manifest places on that node, with the manifest's own
//! arguments; nothing is copied by hand. `--gateway` also starts the chat
//! gateway once every stage endpoint accepts connections. The manifest is data:
//! only `allpaka stage ...` and `allpaka pipeline-serve ...` entries are run,
//! always with this executable.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

/// Set for the children of `pipeline-up`: their stdin is a pipe it holds open.
const PARENT_PIPE: &str = "ALLPAKA_EXIT_WITH_STDIN";

/// In a child of `pipeline-up`, exit when the parent is gone. The parent holds
/// the write end of our stdin and never writes; the pipe reaching end-of-file
/// means it exited or was killed, and a stage nobody supervises must not stay
/// behind holding a port and the model's memory.
pub fn exit_with_parent() {
    if std::env::var_os(PARENT_PIPE).is_none() {
        return;
    }
    std::thread::spawn(|| {
        use std::io::Read;
        let mut sink = [0u8; 64];
        let mut stdin = std::io::stdin();
        loop {
            match stdin.read(&mut sink) {
                Ok(0) | Err(_) => {
                    eprintln!("pipeline-up is gone; exiting");
                    std::process::exit(1);
                }
                Ok(_) => {}
            }
        }
    });
}

/// What to start from a manifest on this machine.
pub struct Options<'a> {
    /// Node names whose stages run here.
    pub nodes: &'a [String],
    /// Every stage, whatever its node: a single-machine run.
    pub all: bool,
    pub gateway: bool,
    /// This machine's path to the GGUF, when it differs from the manifest's.
    pub model: Option<&'a Path>,
    /// Replace the host stages bind to (the manifest says 0.0.0.0).
    pub bind_host: Option<&'a str>,
    pub gateway_bind: Option<&'a str>,
}

/// One process to start: a label for messages and the arguments after the
/// executable.
#[derive(Debug, PartialEq)]
pub struct Launch {
    pub label: String,
    pub args: Vec<String>,
}

fn strings(value: &Value, what: &str) -> Result<Vec<String>> {
    value
        .as_array()
        .with_context(|| format!("{what} is not an array"))?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .with_context(|| format!("{what} holds a non-string"))
        })
        .collect()
}

/// The arguments after `allpaka <subcommand>`, refusing anything else: a
/// manifest must not be a way to run an arbitrary program.
fn checked(argv: Vec<String>, subcommand: &str, what: &str) -> Result<Vec<String>> {
    if argv.len() < 2 || argv[0] != "allpaka" || argv[1] != subcommand {
        bail!("{what} must be `allpaka {subcommand} ...`");
    }
    Ok(argv[1..].to_vec())
}

fn set_after(args: &mut [String], flag: &str, value: String) -> bool {
    match args.iter().position(|arg| arg == flag) {
        Some(at) if at + 1 < args.len() => {
            args[at + 1] = value;
            true
        }
        _ => false,
    }
}

/// The processes `options` selects from `manifest`, stages first, in manifest
/// order, then the gateway.
pub fn select(manifest: &Value, options: &Options<'_>) -> Result<Vec<Launch>> {
    let stages = manifest["stages"]
        .as_array()
        .context("manifest has no stages array")?;
    for wanted in options.nodes {
        if !stages.iter().any(|stage| stage["node"] == wanted.as_str()) {
            bail!("the manifest places no stage on node {wanted}");
        }
    }
    let mut launches = Vec::new();
    for stage in stages {
        let node = stage["node"].as_str().context("stage without a node")?;
        if !options.all && !options.nodes.iter().any(|wanted| wanted == node) {
            continue;
        }
        let mut args = checked(
            strings(&stage["argv"], "stage argv")?,
            "stage",
            "stage argv",
        )?;
        if let Some(model) = options.model {
            // `stage <model> --first ...`: the model is the first operand.
            if args.len() < 2 {
                bail!("stage argv has no model path");
            }
            args[1] = model.to_string_lossy().into_owned();
        }
        if let Some(host) = options.bind_host {
            let at = args
                .iter()
                .position(|arg| arg == "--bind")
                .filter(|at| at + 1 < args.len())
                .context("stage argv has no --bind")?;
            let port = args[at + 1]
                .rsplit_once(':')
                .context("stage --bind has no port")?
                .1
                .to_owned();
            args[at + 1] = format!("{host}:{port}");
        }
        launches.push(Launch {
            label: format!("stage {node} layers {}..{}", stage["first"], stage["end"]),
            args,
        });
    }
    if options.gateway {
        let mut args = checked(
            strings(&manifest["gateway_argv"], "gateway_argv")?,
            "pipeline-serve",
            "gateway_argv",
        )?;
        if let Some(model) = options.model {
            if !set_after(
                &mut args,
                "--tokenizer",
                model.to_string_lossy().into_owned(),
            ) {
                bail!("gateway_argv has no --tokenizer");
            }
        }
        if let Some(bind) = options.gateway_bind {
            args.extend(["--bind".to_owned(), bind.to_owned()]);
        }
        launches.push(Launch {
            label: "gateway".to_owned(),
            args,
        });
    }
    if launches.is_empty() {
        bail!("nothing to start: name a --node, or pass --all or --gateway");
    }
    Ok(launches)
}

/// The stage endpoints the gateway will connect to.
fn gateway_endpoints(manifest: &Value) -> Result<Vec<String>> {
    let argv = strings(&manifest["gateway_argv"], "gateway_argv")?;
    let at = argv
        .iter()
        .position(|arg| arg == "--stages")
        .filter(|at| at + 1 < argv.len())
        .context("gateway_argv has no --stages")?;
    Ok(argv[at + 1].split(',').map(str::to_owned).collect())
}

/// Children that are killed when this goes out of scope, however `run` ends.
struct Running(Vec<(String, Child)>);

impl Running {
    /// The label and exit status of a child that has stopped, if any.
    fn stopped(&mut self) -> Result<Option<String>> {
        for (label, child) in &mut self.0 {
            if let Some(status) = child.try_wait()? {
                return Ok(Some(format!("{label} exited: {status}")));
            }
        }
        Ok(None)
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        for (_, child) in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn accepts(endpoint: &str) -> bool {
    use std::net::ToSocketAddrs;
    endpoint
        .to_socket_addrs()
        .ok()
        .and_then(|mut addresses| addresses.next())
        .is_some_and(|address| {
            std::net::TcpStream::connect_timeout(&address, Duration::from_millis(300)).is_ok()
        })
}

pub fn run(manifest_path: &Path, options: &Options<'_>, wait_secs: u64) -> Result<()> {
    let text = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let manifest: Value = serde_json::from_str(&text).context("the manifest is not JSON")?;
    let launches = select(&manifest, options)?;
    let exe: PathBuf = std::env::current_exe()?;

    let mut running = Running(Vec::new());
    for launch in launches {
        if launch.label == "gateway" {
            // The gateway checks every stage when it starts; give them, local
            // and remote, time to load their layers first.
            let deadline = Instant::now() + Duration::from_secs(wait_secs);
            for endpoint in gateway_endpoints(&manifest)? {
                while !accepts(&endpoint) {
                    if let Some(stopped) = running.stopped()? {
                        bail!("{stopped}");
                    }
                    if Instant::now() > deadline {
                        bail!("stage {endpoint} did not accept connections in {wait_secs} s");
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
                eprintln!("pipeline-up: stage {endpoint} is listening");
            }
        }
        eprintln!("pipeline-up: starting {}", launch.label);
        let child = Command::new(&exe)
            .args(&launch.args)
            .env(PARENT_PIPE, "1")
            // Kept open in `Child` for as long as this process lives.
            .stdin(std::process::Stdio::piped())
            .spawn()
            .with_context(|| format!("starting {}", launch.label))?;
        running.0.push((launch.label, child));
    }

    // Stay up as long as every child does; one stopping takes the rest down.
    loop {
        if let Some(stopped) = running.stopped()? {
            bail!("{stopped}");
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn manifest() -> Value {
        json!({
            "model_id": "m",
            "context": 256,
            "stages": [
                {"node": "mac", "endpoint": "mac.lan:9798", "first": 0, "end": 16,
                 "argv": ["allpaka", "stage", "/models/m.gguf", "--first", "0", "--end", "16",
                          "--model-id", "m", "--context", "256", "--bind", "0.0.0.0:9798"]},
                {"node": "pc", "endpoint": "pc.lan:9799", "first": 16, "end": 32,
                 "argv": ["allpaka", "stage", "/models/m.gguf", "--first", "16", "--end", "32",
                          "--model-id", "m", "--context", "256", "--bind", "0.0.0.0:9799"]}
            ],
            "gateway_argv": ["allpaka", "pipeline-serve", "--stages", "mac.lan:9798,pc.lan:9799",
                             "--tokenizer", "/models/m.gguf", "--model-id", "m", "--context", "256"]
        })
    }

    fn options<'a>(nodes: &'a [String]) -> Options<'a> {
        Options {
            nodes,
            all: false,
            gateway: false,
            model: None,
            bind_host: None,
            gateway_bind: None,
        }
    }

    #[test]
    fn a_node_gets_only_its_own_stage_with_the_manifest_arguments() {
        let nodes = ["pc".to_owned()];
        let launches = select(&manifest(), &options(&nodes)).unwrap();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].label, "stage pc layers 16..32");
        assert_eq!(
            launches[0].args,
            [
                "stage",
                "/models/m.gguf",
                "--first",
                "16",
                "--end",
                "32",
                "--model-id",
                "m",
                "--context",
                "256",
                "--bind",
                "0.0.0.0:9799"
            ]
        );
    }

    #[test]
    fn overrides_touch_only_the_path_and_the_bind_address() {
        let nodes = [];
        let model = PathBuf::from("D:/gguf/m.gguf");
        let launches = select(
            &manifest(),
            &Options {
                all: true,
                gateway: true,
                model: Some(&model),
                bind_host: Some("127.0.0.1"),
                gateway_bind: Some("127.0.0.1:8099"),
                ..options(&nodes)
            },
        )
        .unwrap();
        assert_eq!(launches.len(), 3);
        assert_eq!(launches[0].args[1], "D:/gguf/m.gguf");
        assert_eq!(launches[0].args.last().unwrap(), "127.0.0.1:9798");
        assert_eq!(launches[1].args.last().unwrap(), "127.0.0.1:9799");
        let gateway = &launches[2];
        assert_eq!(gateway.label, "gateway");
        assert_eq!(gateway.args[0], "pipeline-serve");
        assert!(gateway
            .args
            .windows(2)
            .any(|w| w == ["--tokenizer", "D:/gguf/m.gguf"]));
        assert!(gateway
            .args
            .ends_with(&["--bind".to_owned(), "127.0.0.1:8099".to_owned()]));
        assert_eq!(
            gateway_endpoints(&manifest()).unwrap(),
            ["mac.lan:9798", "pc.lan:9799"]
        );
    }

    #[test]
    fn a_manifest_cannot_name_another_program_or_an_unknown_node() {
        let nodes = ["mac".to_owned()];
        let mut other = manifest();
        other["stages"][0]["argv"][0] = json!("powershell");
        assert!(select(&other, &options(&nodes)).is_err());
        let mut wrong = manifest();
        wrong["stages"][0]["argv"][1] = json!("serve");
        assert!(select(&wrong, &options(&nodes)).is_err());
        let mut gateway = manifest();
        gateway["gateway_argv"][1] = json!("studio");
        let none = [];
        assert!(select(
            &gateway,
            &Options {
                gateway: true,
                ..options(&none)
            }
        )
        .is_err());

        let unknown = ["laptop".to_owned()];
        let error = select(&manifest(), &options(&unknown)).err().unwrap();
        assert!(error.to_string().contains("no stage on node laptop"));
        assert!(select(&manifest(), &options(&none)).is_err());
    }
}
