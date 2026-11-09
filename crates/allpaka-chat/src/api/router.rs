//! The HTTP surface: every route of the Studio API in one table, and the
//! guard middleware in front of it.

use crate::*;

pub(crate) fn routes(app: App) -> Router {
    Router::new()
        .route(
            "/",
            get(|| async { Html(include_str!("../../web/index.html")) }),
        )
        .route(
            "/app.js",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                    include_str!("../../web/app.js"),
                )
            }),
        )
        .route(
            "/files.js",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                    include_str!("../../web/files.js"),
                )
            }),
        )
        .route(
            "/style.css",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
                    include_str!("../../web/style.css"),
                )
            }),
        )
        .route(
            "/content.js",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                    include_str!("../../web/content.js"),
                )
            }),
        )
        .route("/api/worker/run", post(remote_worker::run))
        .route("/api/config", get(config))
        .route("/api/memory/notes", get(memory::list).post(memory::save))
        .route("/api/memory/expiry", get(memory::expiry))
        .route(
            "/api/memory/consolidation-input",
            post(memory::consolidation_input),
        )
        .route(
            "/api/memory/consolidation-proposals",
            get(memory_extract::list_consolidations).post(memory_extract::consolidate),
        )
        .route(
            "/api/memory/notes/:id/versions/:version/source-status",
            get(memory::source_status),
        )
        .route(
            "/api/memory/notes/:id/versions/:version",
            get(memory::snapshot),
        )
        .route("/api/observability/traces", get(observability::list))
        .route(
            "/api/observability/online-jobs",
            get(online_evaluation::jobs_api),
        )
        .route(
            "/api/observability/external-traces",
            post(observability::ingest),
        )
        .route(
            "/api/observability/evaluations/summary",
            get(observability::evaluation_summary),
        )
        .route(
            "/api/observability/trace-exports",
            post(observability::bulk_export),
        )
        .route("/api/observability/summary", get(observability::summary))
        .route(
            "/api/evaluation/datasets/:id/lifecycle",
            post(evaluation::lifecycle),
        )
        .route(
            "/api/evaluation/datasets/:id/versions",
            get(evaluation::versions),
        )
        .route(
            "/api/evaluation/datasets/:id/compare",
            get(evaluation::compare),
        )
        .route(
            "/api/guardrail-policies",
            get(guardrail_policies::list_api).post(guardrail_policies::save_api),
        )
        .route(
            "/api/guardrail-policies/create",
            post(guardrail_policies::create_api),
        )
        .route(
            "/api/guardrail-policies/:hash/check",
            post(guardrail_policies::check_api),
        )
        .route(
            "/api/observability/online-rules",
            get(online_evaluation::list_api).post(online_evaluation::save_api),
        )
        .route(
            "/api/observability/online-rules/:hash",
            get(online_evaluation::get_api),
        )
        .route(
            "/api/observability/online-rule-bindings",
            get(online_evaluation::binding_api).post(online_evaluation::bind_api),
        )
        .route(
            "/api/observability/online-selections",
            get(online_evaluation::selection_archive_api).post(online_evaluation::select_api),
        )
        .route(
            "/api/observability/online-model-evaluators",
            post(model_evaluators::save_api),
        )
        .route(
            "/api/observability/online-model-evaluators/:hash",
            get(model_evaluators::read_api),
        )
        .route(
            "/api/observability/online-quality-jobs",
            get(quality_jobs::list_api).post(quality_jobs::submit_api),
        )
        .route(
            "/api/observability/online-quality-jobs/:id",
            get(quality_jobs::get_api),
        )
        .route(
            "/api/observability/online-quality-sources",
            post(online_sources::save_api),
        )
        .route(
            "/api/observability/online-quality-sources/:hash/judge",
            post(judge::judge_source),
        )
        .route(
            "/api/observability/online-quality-sources/:hash",
            get(online_sources::read_api),
        )
        .route(
            "/api/observability/online-assessments",
            post(online_evaluation::assess_api),
        )
        .route(
            "/api/observability/online-jobs/drain",
            post(online_evaluation::drain_api),
        )
        .route(
            "/api/guardrail-policies/:hash",
            get(guardrail_policies::get_api),
        )
        .route(
            "/api/observability/time-series",
            get(observability::time_series),
        )
        .route("/api/observability/traces/:id", get(observability::get))
        .route(
            "/api/observability/traces/:id/export",
            get(observability::export),
        )
        .route(
            "/api/observability/traces/:id/remove",
            post(observability::remove),
        )
        .route(
            "/api/observability/traces/:id/restore",
            post(observability::restore),
        )
        .route(
            "/api/observability/review-queues",
            get(review_queues::list).post(review_queues::create),
        )
        .route(
            "/api/observability/review-queues/:id",
            get(review_queues::detail),
        )
        .route(
            "/api/observability/review-queues/:id/history",
            get(review_queues::history),
        )
        .route(
            "/api/observability/review-queues/:id/assignments",
            post(review_queues::assign),
        )
        .route(
            "/api/observability/review-queues/:id/completion",
            post(review_queues::complete),
        )
        .route(
            "/api/observability/review-queues/:id/lifecycle",
            post(review_queues::lifecycle),
        )
        .route(
            "/api/observability/traces/:id/feedback/versions",
            get(feedback::history),
        )
        .route(
            "/api/observability/traces/:id/feedback",
            get(feedback::list).post(feedback::save),
        )
        .route(
            "/api/evaluation/prompts",
            get(prompts::list).post(prompts::save),
        )
        .route(
            "/api/evaluation/prompts/:id/versions",
            get(prompts::history),
        )
        .route(
            "/api/evaluation/prompts/:id/versions/:version",
            get(prompts::snapshot),
        )
        .route("/api/evaluation/playground", post(experiments::playground))
        .route(
            "/api/evaluation/prompts/:id/versions/:version/preview",
            post(prompts::preview),
        )
        .route(
            "/api/evaluation/datasets",
            get(evaluation::list).post(evaluation::save),
        )
        .route(
            "/api/evaluation/experiments",
            get(experiments::list).post(experiments::start),
        )
        .route("/api/evaluation/experiments/:id", get(experiments::detail))
        .route(
            "/api/evaluation/experiments/:id/export",
            get(experiments::export),
        )
        .route(
            "/api/evaluation/experiments/:id/cancel",
            post(experiments::cancel),
        )
        .route(
            "/api/evaluation/judge-runs",
            get(judge_runs::list).post(judge_runs::start),
        )
        .route("/api/evaluation/judge-runs/:id", get(judge_runs::get))
        .route(
            "/api/evaluation/judge-runs/:id/cancel",
            post(judge_runs::cancel),
        )
        .route(
            "/api/evaluation/judge-runs/compare",
            post(judge_runs::compare),
        )
        .route("/api/evaluation/judge-presets", get(judge_presets::catalog))
        .route("/api/evaluation/judge-plans", post(judge::plan))
        .route("/api/evaluation/judge-plans/:id", get(judge::read_plan))
        .route("/api/evaluation/judge", post(judge::judge))
        .route("/api/evaluation/judge/:id", get(judge::read))
        .route(
            "/api/evaluation/score",
            post(experiments::offline_score).get(experiments::list_offline_scores),
        )
        .route(
            "/api/evaluation/score/:id",
            get(experiments::read_offline_score),
        )
        .route(
            "/api/evaluation/score/compare",
            post(experiments::compare_offline_scores),
        )
        .route("/api/evaluation/metrics", get(experiments::metric_catalog))
        .route("/api/evaluation/compare", post(experiments::compare))
        .route(
            "/api/evaluation/comparisons/:id",
            get(experiments::read_comparison),
        )
        .route(
            "/api/evaluation/comparisons",
            get(experiments::list_comparisons),
        )
        .route(
            "/api/evaluation/matrices",
            post(experiments::save_matrix).get(experiments::list_matrices),
        )
        .route(
            "/api/evaluation/matrices/:id",
            get(experiments::read_matrix),
        )
        .route(
            "/api/evaluation/matrix-jobs",
            post(experiments::matrix_jobs::start).get(experiments::matrix_jobs::list),
        )
        .route(
            "/api/evaluation/matrix-jobs/:id",
            get(experiments::matrix_jobs::detail),
        )
        .route(
            "/api/evaluation/matrix-jobs/:id/cancel",
            post(experiments::matrix_jobs::cancel),
        )
        .route(
            "/api/evaluation/matrix-jobs/:id/resume",
            post(experiments::matrix_jobs::resume),
        )
        .route(
            "/api/evaluation/matrix-jobs/:id/retry",
            post(experiments::matrix_jobs::retry),
        )
        .route(
            "/api/evaluation/datasets/:id/versions/:version",
            get(evaluation::snapshot),
        )
        .route(
            "/api/plugins",
            get(plugins::list_plugins).post(plugins::save_plugin),
        )
        .route("/api/plugins/:id/delete", post(plugins::delete_plugin))
        .route("/api/plugins/:id/reload", post(plugins::reload_plugin))
        .route("/api/projects", post(save_project))
        .route("/api/files/list", post(files::list))
        .route("/api/files/search", post(files::search))
        .route("/api/files/read", post(files::read))
        .route(
            "/api/harness",
            get(harness::get_api).post(harness::save_api),
        )
        .route("/api/harness/activate", post(harness::activate_api))
        .route("/api/harness/import-skill", post(harness::import_skill_api))
        .route("/api/harness/revisions/:id", get(harness::revision_api))
        .route(
            "/api/harness/jobs",
            get(harness_refine::list_api).post(harness_refine::start_api),
        )
        .route("/api/harness/jobs/:id", get(harness_refine::get_api))
        .route(
            "/api/harness/jobs/:id/cancel",
            post(harness_refine::cancel_api),
        )
        .route(
            "/api/harness/jobs/:id/resume",
            post(harness_refine::resume_api),
        )
        .route(
            "/api/harness/jobs/:id/adopt",
            post(harness_refine::adopt_api),
        )
        .route("/api/harness/library", get(harness_library::list_api))
        .route("/api/providers", post(provider::save_provider))
        .route("/api/providers/:id/delete", post(provider::delete_provider))
        .route("/api/providers/:id/key", post(provider::set_key))
        .route("/api/providers/:id/models", get(provider::model_list))
        .route("/api/providers/:id/doctor", post(provider::doctor))
        .route("/api/sessions", get(history::list).post(create_session))
        .route("/api/sessions/:id/background", post(background_action))
        .route("/api/sessions/import", post(history::import))
        .route("/api/sessions/:id", get(session).delete(delete_session))
        .route("/api/sessions/:id/branch", post(branch_session))
        .route(
            "/api/sessions/:id/memory-proposals",
            get(memory_extract::list).post(memory_extract::extract),
        )
        .route("/api/memory/proposals/:id", get(memory_extract::read))
        .route(
            "/api/memory/proposals/:id/source-status",
            get(memory_extract::source_status),
        )
        .route("/api/sessions/:id/bookmarks", post(bookmarks::save))
        .route(
            "/api/sessions/:id/bookmarks/:index",
            axum::routing::delete(bookmarks::remove),
        )
        .route("/api/sessions/:id/actions", post(action))
        .route("/api/sessions/:id/plan", get(read_plan).post(save_plan))
        .layer(DefaultBodyLimit::max(10_000_000))
        .layer(middleware::from_fn_with_state(app.clone(), guard))
        .with_state(app)
}
pub(crate) async fn guard(
    State(app): State<App>,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    if req.uri().path() == "/api/worker/run" {
        if !remote_worker::authorized(req.headers()) {
            return error(StatusCode::UNAUTHORIZED, "Worker authentication required")
                .into_response();
        }
        return next.run(req).await;
    }
    let expected = app.origin.strip_prefix("http://").unwrap();
    let host_ok = req
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        == Some(expected);
    let origin_ok = req
        .headers()
        .get(header::ORIGIN)
        .map(|h| h.to_str().ok() == Some(app.origin.as_str()))
        .unwrap_or(true);
    let site_ok = req
        .headers()
        .get("sec-fetch-site")
        .map(|s| s != "cross-site")
        .unwrap_or(true);
    let mutation = req.method() != axum::http::Method::GET;
    let intent_ok = !mutation
        || req
            .headers()
            .get("x-allpaka-client")
            .is_some_and(|v| v == "studio");
    if !host_ok || !origin_ok || !site_ok || !intent_ok {
        return error(StatusCode::FORBIDDEN, "Use the local Studio page").into_response();
    }
    let mut response = next.run(req).await;
    let h = response.headers_mut();
    h.insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    h.insert("x-content-type-options", "nosniff".parse().unwrap());
    h.insert("content-security-policy","default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'".parse().unwrap());
    response
}
