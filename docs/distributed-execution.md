# Distributed execution

The native engine can execute a token through ordered TCP transformer stages.
Every connection owns independent KV/SSM state. The coordinator rejects gaps,
overlaps, inconsistent model identities/dimensions and differing SHA-256
fingerprints of the complete GGUF parts. Connection failure
invalidates the session: reconnect and replay the prompt rather than retrying a
partially applied token.

Use a private trusted network or an encrypted tunnel: the TCP protocol is not
TLS and transmits activations and its shared credential in plaintext.
Set the same `ALLPAKA_CLUSTER_KEY` (at least 16 bytes) on the machines.

For a 32-layer model, run on the first machine:

```sh
allpaka stage model.gguf --first 0 --end 16 --model-id MODEL_SHA256 --bind 0.0.0.0:9798
```

On the second machine:

```sh
allpaka stage model.gguf --first 16 --end 32 --model-id MODEL_SHA256 --bind 0.0.0.0:9798
```

The model identity is a display name. Every stage additionally computes a
SHA-256 fingerprint of all GGUF parts; differing weights are rejected even
when the display name is identical. Computing the fingerprint reads the files
with a bounded 1 MiB buffer once at startup.
From the coordinator, supply already tokenized prompt IDs:

```sh
allpaka pipeline --stages machine-a:9798,machine-b:9798 --tokens 1,234,567 --generate 16
```

The command prints generated token IDs using greedy sampling. Each stage loads
only its assigned layer tensors, endpoint tensors when needed, and a cache with
only its own layers. GGUF files are memory mapped; unselected pages are not
read by the stage loader. Stage loading registers only page-aligned ranges covering selected tensors
with the GPU. Multi-machine qualification remains outstanding.

Verification uses a deterministic two-layer GGUF fixture. Two separate TCP
listeners execute independent stages; their logits match whole-model execution
across multiple tokens. An incorrect shared key is rejected.

## Remote swarm agents

Start Studio on each worker with its own workspace and provider configuration.
Set the shared `ALLPAKA_CLUSTER_KEY` on both coordinator and workers before
starting Studio. Studio remains bound to loopback; use an SSH tunnel to expose
the worker origin to the coordinator, for example:

```sh
ssh -N -L 18100:127.0.0.1:8100 worker-host
```

In a Swarm member, set its worker URL to `http://127.0.0.1:18100`, its worker
project ID (usually `default`), and the provider/model IDs configured on that
worker. Empty worker URL retains local execution. The worker owns the model
request loop and read-only project tools; the coordinator merges its report.
Provider credentials stay on the worker. Up to three requests run concurrently
per worker; excess requests fail visibly. Requests time out and oversized
responses are refused. Retries preserve the member's selected worker.

The worker endpoint is `/api/worker/run` and requires the shared bearer key.
The implementation currently returns a finished report rather than streaming
worker progress. End-to-end multi-machine qualification remains outstanding.

## Distributed model in Studio

Start a local gateway with the same model identity used by every stage:

```sh
allpaka pipeline-serve --stages machine-a:9798,machine-b:9798 --tokenizer model.gguf --model-id MODEL_SHA256
```

The gateway maps the GGUF for tokenizer metadata without loading its weight
graph. Add its `http://127.0.0.1:8099/v1` origin as a custom Studio provider.
The `/v1/models` response lists the model identity; chat requests tokenize using
the existing chat template formatter and send token activations through the
remote stages. Tool schemas use the same rendering and parsing as local
serving. The gateway supports greedy sampling and text input. SSE responses deliver text incrementally during generation, with tool-call
blocks held until they can be parsed. A failure after streaming starts closes
the response without a successful finish event.

## Planner to execution

```sh
allpaka pipeline-plan model.gguf --config allpaka.toml --model-id MODEL_SHA256 --endpoint mac=machine-a:9798 --endpoint desktop=machine-b:9798
```

This uses the existing cluster planner and emits a JSON manifest containing
ordered stage ranges, each worker's argument array and gateway arguments. Node
names must match the configuration. Memory planning uses the engine's 16-bit KV
cache representation. Infeasible models and missing worker endpoints fail.
The manifest does not execute SSH commands or assume remote model paths; copy
arguments to the named machines and use their corresponding GGUF paths.

The remote-agent integration test starts a real HTTP worker and a deterministic
model endpoint. The model requests a file from the worker-owned project; the
worker reads it and submits the result for a final model report. Missing bearer
authentication and unknown projects are rejected, and multi-step usage is summed.

## Process-level regression

Build the current CLI, then run:

```sh
python3 scripts/test-distributed.py
ALLPAKA_TEST_GPU=1 python3 scripts/test-distributed.py
```

The script generates a deterministic 128-wide, two-layer native GGUF with its
own byte vocabulary and chat markers. It starts separate stage, gateway and
whole-model processes on ephemeral loopback ports. The whole-model process is
a CPU reference even in GPU mode. It compares generated text and token usage,
JSON and SSE responses, independent-session replay, wrong model IDs, oversized
context, unsupported sampling and a terminated worker. Processes and temporary
fixtures are cleaned up on both success and failure. This proves the local
multi-process path; it does not measure network performance or replace physical
multi-machine qualification.

The process regression also changes one weight byte on a replacement stage
while retaining the model name and dimensions. The coordinator rejects it
before running any token. The tokenizer gateway verifies its GGUF fingerprint
against the worker fingerprints as well.

Protocol regressions additionally prove that a tail failure invalidates the
already-advanced head session, reconnect is required after a dropped worker,
local token/context validation does not advance state, malformed outputs are
rejected, gaps/overlaps fail before execution, and oversized frames are rejected
before allocating their declared payload.

Remote provider IDs are resolved on the worker rather than required in the
coordinator's provider list. The member editor accepts worker-specific IDs.
Reports store the worker origin and project ID alongside provider/model IDs;
retry uses this original execution location even if the roster changed or the
member was removed. Remote report limits are enforced at UTF-8 boundaries.

The worker integration regression also exercises the full Swarm controller:
two members whose provider exists only on the worker run their tool loops,
both reports enter the coordinator transcript, and a coordinator-only provider
merges them. Removing the members from the current roster and retrying one
still uses its stored worker and rebuilds the merged answer.

Remote worker execution belongs to the HTTP response body future. Disconnecting
or cancelling the coordinator request drops this future, stops upstream model
I/O and releases the worker slot. The integration regression fills all three
slots with stalled model streams, verifies a fourth request is refused, cancels
the three callers, observes all upstream streams close and verifies a fresh
request succeeds. Worker errors during execution are explicit JSON errors and
are never accepted as successful reports.
