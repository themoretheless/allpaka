#!/usr/bin/env python3
"""Exercise memory admission against a real local model and HTTP server."""
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    model = "models/qwen3-0.6b-Q8_0.gguf"
    mib = 1 << 20
    budget = (os.stat(model).st_size + mib - 1) // mib + 32
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    def call(path, body=None):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def chat(tokens):
        return call("/v1/chat/completions", {
            "model": "qwen3", "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": tokens, "temperature": 0,
        })

    with open("/tmp/allpaka-serving-memory.log", "w") as log:
        server = subprocess.Popen([
            "target/debug/allpaka", "serve", "--model", model,
            "--bind", f"127.0.0.1:{port}", "--memory-budget-mib", str(budget),
            "--prefix-cache-mib", "8",
        ], env=dict(os.environ, ALLPAKA_RAG_TOOLS="0"), stdout=log, stderr=log)
        try:
            for _ in range(100):
                assert server.poll() is None, "server exited before health"
                try:
                    if call("/health")[0] == 200:
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise AssertionError("server startup timed out")
            before = call("/stats")
            assert before[0] == 200, before
            rejected = chat(10000)
            assert rejected[0] == 429, rejected
            assert "memory admission rejected" in rejected[1]["error"], rejected
            assert chat(1)[0] == 200, "server did not recover after memory rejection"
            after = call("/stats")
            assert after[0] == 200, after
            admission = after[1]["memory_admission"]
            assert admission["reserved_bytes"] == before[1]["memory_admission"]["reserved_bytes"]
            assert admission["peak_reserved_bytes"] <= admission["limit_bytes"]
            print("PASS: memory rejection returns 429, reservations release, next request succeeds")
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
