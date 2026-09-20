//! Traces, metrics, and logs exported through airbug from process start.
//!
//! On for every launch. `ALLPAKA_AIRBUG=0` turns it off.

use std::sync::atomic::{AtomicBool, Ordering};

static ON: AtomicBool = AtomicBool::new(false);

pub fn install() {
    imp::install(wanted());
}

pub fn span<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    imp::span(name, f)
}

pub fn log_rate(phase: &'static str, tokens: usize, secs: f64, tok_s: f64) {
    imp::log_rate(phase, tokens, secs, tok_s);
}

pub fn log(body: &str) {
    imp::log(body);
}

pub fn event(scope: &'static str, body: &str) {
    imp::event(scope, body);
}

pub fn histogram(name: &'static str, value: f64) {
    imp::histogram(name, value);
}

pub fn gauge(name: &'static str, value: f64) {
    imp::gauge(name, value);
}

pub fn save_run(prefill_tok_s: f64, prefill_tokens: usize, decode_tok_s: f64, decode_tokens: usize) {
    imp::save_run(prefill_tok_s, prefill_tokens, decode_tok_s, decode_tokens);
}

fn wanted() -> bool {
    !matches!(
        std::env::var("ALLPAKA_AIRBUG").as_deref(),
        Ok("0" | "false" | "off")
    )
}

fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

#[cfg(feature = "airbug")]
mod imp {
    use super::ON;
    use airbug_bench::{Availability, Case, Direction, Metric, Observation, Run, Status};
    use std::collections::BTreeMap;
    use std::sync::atomic::Ordering;
    use std::sync::OnceLock;

    static GUARD: OnceLock<airbug_otel::TelemetryGuard> = OnceLock::new();

    pub fn install(enable: bool) {
        if !enable {
            return;
        }
        match airbug_otel::init(airbug_otel::TelemetryConfig::new().service_name("allpaka")) {
            Ok(guard) => {
                let _ = GUARD.set(guard);
                ON.store(true, Ordering::Relaxed);
                allpaka_model::set_trace_hook(trace_log);
                let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
                    .unwrap_or_else(|_| "http://localhost:4318".into());
                airbug_otel::add_counter("allpaka", "process_starts", 1, &[]);
                airbug_otel::log_info("startup", format!("allpaka started, otlp {endpoint}"));
                eprintln!("airbug: traces and logs -> {endpoint}");
            }
            Err(e) => eprintln!("airbug: {e}"),
        }
    }

    fn trace_log(line: &str) {
        airbug_otel::log_info("trace", line);
    }

    pub fn span<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
        if !super::on() {
            return f();
        }
        airbug_otel::in_span("allpaka", name, f)
    }

    pub fn log_rate(phase: &'static str, tokens: usize, secs: f64, tok_s: f64) {
        if !super::on() {
            return;
        }
        airbug_otel::record_histogram(
            "allpaka",
            "bench.tok_s",
            tok_s,
            &[airbug_otel::KeyValue::new("phase", phase)],
        );
        airbug_otel::log_info(
            "bench",
            format!("{phase} {tokens} tok in {secs:.2}s {tok_s:.1} tok/s"),
        );
    }

    pub fn log(body: &str) {
        event("bench", body);
    }

    pub fn event(scope: &'static str, body: &str) {
        if super::on() {
            airbug_otel::log_info(scope, body);
        }
    }

    pub fn histogram(name: &'static str, value: f64) {
        if super::on() {
            airbug_otel::record_histogram("allpaka", name, value, &[]);
        }
    }

    pub fn gauge(name: &'static str, value: f64) {
        if super::on() {
            airbug_otel::F64Gauge::new("allpaka", name).record(value, &[]);
        }
    }

    pub fn save_run(
        prefill_tok_s: f64,
        prefill_tokens: usize,
        decode_tok_s: f64,
        decode_tokens: usize,
    ) {
        if !super::on() {
            return;
        }
        let mut run = Run::new();
        run.status = Status::Complete;
        run.provenance
            .insert("tool".into(), "allpaka".into());
        run.cases.push(Case {
            id: "engine".into(),
            contract: BTreeMap::new(),
            metrics: vec![
                rate_metric("prefill_tok_s", "prefill"),
                rate_metric("decode_tok_s", "decode"),
            ],
        });
        let pid = std::process::id();
        run.observations.push(rate_obs(
            "prefill_tok_s",
            prefill_tok_s,
            prefill_tokens,
            pid,
        ));
        run.observations
            .push(rate_obs("decode_tok_s", decode_tok_s, decode_tokens, pid));
        let dir = std::path::PathBuf::from("target/allpaka-bench").join(format!("airbug-{}", run.id));
        match run.save_new(&dir) {
            Ok(()) => eprintln!("airbug bench run: {}", dir.display()),
            Err(e) => eprintln!("airbug bench run: {e}"),
        }
    }

    fn rate_metric(id: &str, phase: &str) -> Metric {
        Metric {
            id: id.into(),
            unit: "tok/s".into(),
            scope: "engine".into(),
            phase: phase.into(),
            statistic: "rate".into(),
            direction: Direction::Higher,
        }
    }

    fn rate_obs(metric: &str, tok_s: f64, tokens: usize, pid: u32) -> Observation {
        Observation {
            case: "engine".into(),
            metric: metric.into(),
            variant: "gpu".into(),
            process: pid,
            pair: None,
            sequence: 0,
            value: Some(format!("{tok_s:.6}")),
            operations: tokens.max(1) as u64,
            availability: Availability::Available,
        }
    }
}

#[cfg(not(feature = "airbug"))]
mod imp {
    pub fn install(_enable: bool) {}

    pub fn span<T>(_name: &'static str, f: impl FnOnce() -> T) -> T {
        f()
    }

    pub fn log_rate(_phase: &'static str, _tokens: usize, _secs: f64, _tok_s: f64) {}

    pub fn log(_body: &str) {}

    pub fn event(_scope: &'static str, _body: &str) {}

    pub fn histogram(_name: &'static str, _value: f64) {}

    pub fn gauge(_name: &'static str, _value: f64) {}

    pub fn save_run(
        _prefill_tok_s: f64,
        _prefill_tokens: usize,
        _decode_tok_s: f64,
        _decode_tokens: usize,
    ) {
    }
}
