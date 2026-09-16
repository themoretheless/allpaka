#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  ifdef ALLPAKA_GGML_BUILD
#    define ALLPAKA_GGML_API __declspec(dllexport)
#  else
#    define ALLPAKA_GGML_API __declspec(dllimport)
#  endif
#else
#  define ALLPAKA_GGML_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Initialize ggml CUDA backend (device 0 by default). Idempotent. Returns 0 on success. */
ALLPAKA_GGML_API int allpaka_ggml_init(int device);

ALLPAKA_GGML_API void allpaka_ggml_shutdown(void);

/**
 * Q4_K / Q6_K GEMM via ggml CUDA MMQ/MMVQ.
 * Device pointers (same CUDA context/device as ggml).
 * W layout: row-major GGUF blocks, n_out rows × (n_in/256) blocks.
 * X: [m, n_in] row-major F32; Y: [m, n_out] row-major F32.
 * Computes Y = X @ W^T (ggml_mul_mat).
 * w_type: 12 = Q4_K, 14 = Q6_K (ggml_type enum).
 * Returns 0 on success.
 */
ALLPAKA_GGML_API int allpaka_ggml_mul_mat(
    const void * w_dev,
    const float * x_dev,
    float * y_dev,
    int32_t n_in,
    int32_t n_out,
    int32_t m,
    int32_t w_type);

/**
 * Prefill flash-attn via ggml CUDA (GQA).
 * q_dev:  [m, n_q_heads, head_dim] F32
 * k_dev/v_dev: cache base for K/V; layout [n_kv, n_kv_heads, head_dim] F16
 *   with element stride kv_dim between positions (kv_dim = n_kv_heads * head_dim).
 * out_dev: [m, n_q_heads, head_dim] F32
 * Causal mask: token t attends to positions [0, base + t].
 * If d_pos_dev is non-null and m==1, mask is filled on device from *d_pos_dev
 * (graph-capturable) and `base` is ignored for masking.
 * Returns 0 on success.
 */
ALLPAKA_GGML_API int allpaka_ggml_flash_attn(
    const float * q_dev,
    const void * k_dev,
    const void * v_dev,
    float * out_dev,
    int32_t head_dim,
    int32_t n_q_heads,
    int32_t n_kv_heads,
    int32_t m,
    int32_t base,
    int32_t kv_dim,
    float scale,
    const uint32_t * d_pos_dev);

/** Force next decode FA to re-record mask_from_pos (for CUDA graph capture). */
ALLPAKA_GGML_API void allpaka_fa_reset_mask_lim(void);

/** Drain ggml CUDA stream. Call before reading Y or mixing with other CUDA work. */
ALLPAKA_GGML_API void allpaka_ggml_sync(void);

/**
 * Direct llama MMVQ for m=1 (no cgraph). Faster than mul_mat graph for decode.
 * Same pointer conventions as allpaka_ggml_mul_mat with m=1.
 */
ALLPAKA_GGML_API int allpaka_ggml_mul_mat_vec(
    const void * w_dev,
    const float * x_dev,
    float * y_dev,
    int32_t n_in,
    int32_t n_out,
    int32_t w_type);

/**
 * Bind the host runtime's CUDA stream so ggml can wait/signal via events
 * (no cudaDeviceSynchronize). peer may be nullptr (legacy default stream).
 */
ALLPAKA_GGML_API int allpaka_ggml_bind_peer_stream(void * peer_stream);

/** ggml stream waits for work already queued on the peer stream. */
ALLPAKA_GGML_API void allpaka_ggml_wait_peer(void);

/** Peer stream waits for work already queued on the ggml stream. */
ALLPAKA_GGML_API void allpaka_ggml_signal_peer(void);

/** 1 if ggml is using the bound cudarc stream (no event sync needed). */
ALLPAKA_GGML_API int allpaka_ggml_peer_is_shared(void);

/** Drop shared-stream install so ggml uses its own queue (events). Re-bind after. */
ALLPAKA_GGML_API void allpaka_ggml_clear_shared(void);

/** Non-null => next decode flash_attn skips permute (Rust does permute+Q8). */
ALLPAKA_GGML_API void allpaka_fa_set_q8(void * q8_dev);

/** 1 if the last flash_attn skipped permute for Q8 fuse. */
ALLPAKA_GGML_API int allpaka_fa_q8_consumed(void);

/** FA device buffer from last skip-permute flash_attn (layout [hd,n_q]). */
ALLPAKA_GGML_API void * allpaka_fa_last_src(void);

/**
 * Hybrid decode replay: for each layer launch pre graph, flash-attn, post graph,
 * then tail. All on `stream` (cudaStream_t). Graph execs are cudaGraphExec_t.
 * k_off/v_off are element offsets into the f16 KV cache.
 */
ALLPAKA_GGML_API int allpaka_hybrid_replay(
    void * const * pre_execs,
    void * const * post_execs,
    void * tail_exec,
    int32_t n_layers,
    void * stream,
    const float * q_dev,
    float * out_dev,
    const void * cache_base,
    const int32_t * k_off,
    const int32_t * v_off,
    int32_t head_dim,
    int32_t n_q_heads,
    int32_t n_kv_heads,
    int32_t base,
    int32_t kv_dim,
    float scale,
    const uint32_t * d_pos_dev);

/** Underlying ggml cudaStream_t (opaque). */
ALLPAKA_GGML_API void * allpaka_ggml_cuda_stream(void);

/** Event recorded after each mul_mat completes on the ggml stream. */
ALLPAKA_GGML_API void * allpaka_ggml_done_event(void);

#ifdef __cplusplus
}
#endif
