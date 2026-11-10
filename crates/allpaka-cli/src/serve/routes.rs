//! The route table: every endpoint the server answers, as a method and a
//! pattern over the rest of the request line.
//!
//! Patterns see the raw request line after the method and its space, HTTP
//! version included. That is what the server has always matched on, and some
//! routes depend on it: `/health` and the cancel route only answer
//! `HTTP/1.1`, `/resources` needs the space after the path, and the prefix
//! routes also accept a query string or a longer path (`/v1/models/qwen3`).
//!
//! Which stage answers a route is decided by the caller, not here: the reader
//! answers `Health`, `Resources` and `CancelRequest` before the queue, the
//! dispatcher answers `Catalog`, `Launch` and `Models` before model
//! resolution, and everything else is answered under the model lock. A line
//! that matches no entry still goes through model resolution and ends in the
//! "unknown route" 404 there; there is no 405.

/// An endpoint the server answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Route {
    Health,
    Resources,
    CancelRequest,
    Catalog,
    Launch,
    Models,
    Stats,
    ChatPage,
    Options,
    PromptPredict,
    PromptPrewarm,
    ChatCompletions,
}

/// How the rest of the request line (path and HTTP version) is matched.
#[derive(Debug, Clone, Copy)]
enum Target {
    /// The whole rest of the line.
    Exact(&'static str),
    /// The rest of the line starts with this.
    Prefix(&'static str),
    /// Starts with the first part and ends with the second; what lies between
    /// is the route parameter.
    Around(&'static str, &'static str),
}

impl Target {
    /// The parameter on a match (empty for routes without one).
    fn matches<'a>(&self, rest: &'a str) -> Option<&'a str> {
        match *self {
            Target::Exact(exact) => (rest == exact).then_some(""),
            Target::Prefix(prefix) => rest.starts_with(prefix).then_some(""),
            Target::Around(prefix, suffix) => rest.strip_prefix(prefix)?.strip_suffix(suffix),
        }
    }

    /// The fixed leading text every matching line starts with.
    #[cfg(test)]
    fn lead(&self) -> &'static str {
        match *self {
            Target::Exact(lead) | Target::Prefix(lead) | Target::Around(lead, _) => lead,
        }
    }
}

struct Entry {
    method: &'static str,
    target: Target,
    route: Route,
}

const fn entry(method: &'static str, target: Target, route: Route) -> Entry {
    Entry {
        method,
        target,
        route,
    }
}

/// No two entries with the same method can match the same line (a test checks
/// it), so the order below is for reading only.
const ROUTES: &[Entry] = &[
    // Exact line: a prefix match here used to answer 200 for `/healthz` and
    // anything else starting with `/health`.
    entry("GET", Target::Exact("/health HTTP/1.1"), Route::Health),
    entry("GET", Target::Prefix("/resources "), Route::Resources),
    entry(
        "POST",
        Target::Around("/v1/requests/", "/cancel HTTP/1.1"),
        Route::CancelRequest,
    ),
    entry("GET", Target::Prefix("/v1/catalog"), Route::Catalog),
    entry("POST", Target::Prefix("/v1/launch"), Route::Launch),
    entry("GET", Target::Prefix("/v1/models"), Route::Models),
    // Prefix match, deliberately: `/stats?ts=...` is a normal request from a
    // polling client. Tightening it to an exact line is a separate call.
    entry("GET", Target::Prefix("/stats"), Route::Stats),
    entry("GET", Target::Prefix("/ "), Route::ChatPage),
    entry("GET", Target::Prefix("/index"), Route::ChatPage),
    // CORS preflight for any path.
    entry("OPTIONS", Target::Prefix(""), Route::Options),
    entry(
        "POST",
        Target::Prefix("/v1/prompt/predict"),
        Route::PromptPredict,
    ),
    entry(
        "POST",
        Target::Prefix("/v1/prompt/prewarm"),
        Route::PromptPrewarm,
    ),
    entry(
        "POST",
        Target::Prefix("/v1/chat/completions"),
        Route::ChatCompletions,
    ),
];

/// A request line resolved against the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Matched<'a> {
    pub(super) route: Route,
    /// The text captured by an `Around` pattern; empty otherwise.
    pub(super) param: &'a str,
}

/// Finds the route for a raw request line such as `GET /stats HTTP/1.1`.
pub(super) fn classify(request_line: &str) -> Option<Matched<'_>> {
    let (method, rest) = request_line.split_once(' ')?;
    ROUTES.iter().find_map(|entry| {
        if entry.method != method {
            return None;
        }
        let param = entry.target.matches(rest)?;
        Some(Matched {
            route: entry.route,
            param,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(line: &str) -> Option<Route> {
        classify(line).map(|matched| matched.route)
    }

    #[test]
    fn every_endpoint_reaches_its_route() {
        let cases = [
            ("GET /health HTTP/1.1", Route::Health),
            ("GET /resources HTTP/1.1", Route::Resources),
            (
                "POST /v1/requests/abc/cancel HTTP/1.1",
                Route::CancelRequest,
            ),
            ("GET /v1/catalog HTTP/1.1", Route::Catalog),
            ("POST /v1/launch HTTP/1.1", Route::Launch),
            ("GET /v1/models HTTP/1.1", Route::Models),
            ("GET /stats HTTP/1.1", Route::Stats),
            ("GET / HTTP/1.1", Route::ChatPage),
            ("GET /index.html HTTP/1.1", Route::ChatPage),
            ("OPTIONS /v1/chat/completions HTTP/1.1", Route::Options),
            ("POST /v1/prompt/predict HTTP/1.1", Route::PromptPredict),
            ("POST /v1/prompt/prewarm HTTP/1.1", Route::PromptPrewarm),
            ("POST /v1/chat/completions HTTP/1.1", Route::ChatCompletions),
        ];
        for (line, expected) in cases {
            assert_eq!(route(line), Some(expected), "{line:?}");
        }
    }

    #[test]
    fn prefix_routes_accept_longer_paths_queries_and_other_versions() {
        let cases = [
            ("GET /v1/models/qwen3 HTTP/1.1", Route::Models),
            ("GET /v1/modelsx HTTP/1.0", Route::Models),
            ("GET /v1/catalog?all=1 HTTP/1.1", Route::Catalog),
            ("POST /v1/launch/now HTTP/1.1", Route::Launch),
            ("GET /stats?ts=1 HTTP/1.1", Route::Stats),
            ("GET /statsx HTTP/1.1", Route::Stats),
            ("GET / HTTP/1.0", Route::ChatPage),
            ("GET /indexes HTTP/1.1", Route::ChatPage),
            ("GET /resources HTTP/1.0", Route::Resources),
            ("OPTIONS * HTTP/1.1", Route::Options),
            ("OPTIONS ", Route::Options),
            (
                "POST /v1/chat/completions?x=1 HTTP/1.1",
                Route::ChatCompletions,
            ),
            ("POST /v1/prompt/predictions HTTP/1.1", Route::PromptPredict),
        ];
        for (line, expected) in cases {
            assert_eq!(route(line), Some(expected), "{line:?}");
        }
    }

    #[test]
    fn exact_and_around_routes_need_the_whole_shape() {
        for line in [
            "GET /healthz HTTP/1.1",
            "GET /health?probe=1 HTTP/1.1",
            "GET /health HTTP/1.0",
            "GET /health",
            "GET /resources?since=1 HTTP/1.1",
            "GET /resourcesx HTTP/1.1",
            "POST /v1/requests/abc/cancel HTTP/1.0",
            "POST /v1/requests/abc HTTP/1.1",
            "POST /v1/requests/cancel HTTP/1.1",
        ] {
            assert_eq!(route(line), None, "{line:?}");
        }
    }

    #[test]
    fn cancel_captures_everything_between_prefix_and_suffix() {
        let param = |line| classify(line).map(|matched| matched.param);
        assert_eq!(param("POST /v1/requests/abc/cancel HTTP/1.1"), Some("abc"));
        assert_eq!(param("POST /v1/requests/a/b/cancel HTTP/1.1"), Some("a/b"));
        assert_eq!(param("POST /v1/requests//cancel HTTP/1.1"), Some(""));
        assert_eq!(param("GET /stats HTTP/1.1"), Some(""));
    }

    #[test]
    fn methods_are_exact_and_case_sensitive() {
        for line in [
            "POST /stats HTTP/1.1",
            "PUT /v1/chat/completions HTTP/1.1",
            "GET /v1/chat/completions HTTP/1.1",
            "POST /v1/models HTTP/1.1",
            "GET /v1/launch HTTP/1.1",
            "GET /v1/prompt/predict HTTP/1.1",
            "get /stats HTTP/1.1",
            "DELETE /stats HTTP/1.1",
            "OPTIONS",
            " GET /stats HTTP/1.1",
            "GET",
            "",
        ] {
            assert_eq!(route(line), None, "{line:?}");
        }
    }

    #[test]
    fn unknown_paths_match_nothing() {
        for line in [
            "GET /nope HTTP/1.1",
            "GET /v1 HTTP/1.1",
            "GET //stats HTTP/1.1",
        ] {
            assert_eq!(route(line), None, "{line:?}");
        }
    }

    /// If neither lead is a prefix of the other, no line can start with both,
    /// so first-match order in the table never decides anything.
    #[test]
    fn entries_with_the_same_method_never_overlap() {
        for (i, a) in ROUTES.iter().enumerate() {
            for b in &ROUTES[i + 1..] {
                if a.method != b.method {
                    continue;
                }
                let (a_lead, b_lead) = (a.target.lead(), b.target.lead());
                assert!(
                    !a_lead.starts_with(b_lead) && !b_lead.starts_with(a_lead),
                    "{} {a_lead:?} overlaps {b_lead:?}",
                    a.method
                );
            }
        }
    }
}
