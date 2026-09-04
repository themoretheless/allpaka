#!/usr/bin/env python3
"""Live HTTP regression checks. Requires the small local GGUF and Metal."""
import concurrent.futures
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def chat(**extra):
        body = {"model": "qwen3", "messages": [{"role": "user", "content": "Count upward starting at one."}],
                "max_tokens": 1, "temperature": 0}
        body.update(extra)
        return call("/v1/chat/completions", body)

    env = dict(os.environ, ALLPAKA_RAG_TOOLS="0")
    with open("/tmp/allpaka-serving-controls.log", "w") as log:
        server = subprocess.Popen(["target/debug/allpaka", "serve", "--model", "models/qwen3-0.6b-Q8_0.gguf",
                                   "--bind", f"127.0.0.1:{port}", "--max-queued", "2"],
                                  env=env, stdout=log, stderr=log)
        try:
            for _ in range(100):
                if server.poll() is not None:
                    raise AssertionError("server exited before health")
                try:
                    if call("/health")[0] == 200:
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise AssertionError("server startup timed out")

            a = chat(session_id="a")
            b = chat(session_id="b")
            again = chat(session_id="a")
            assert all(r[0] == 200 for r in (a, b, again)), (a, b, again)
            assert b[1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, b
            assert again[1]["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, again
            anonymous = chat()
            assert anonymous[1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
            assert chat(session_id=123)[0] == 400
            assert chat(max_tokens=2**63)[0] == 400

            with concurrent.futures.ThreadPoolExecutor() as pool:
                active = pool.submit(chat, request_id="cancel-me", max_tokens=10000)
                for _ in range(100):
                    cancelled = call("/v1/requests/cancel-me/cancel", {})
                    if cancelled[0] == 200:
                        break
                    time.sleep(0.01)
                assert cancelled[0] == 200, cancelled
                assert active.result()[0] == 408
                assert call("/health")[0] == 200

            deadline = chat(request_id="deadline", max_tokens=10000, timeout_ms=1)
            assert deadline[0] == 408, deadline
            assert chat()[0] == 200
            print("PASS: session isolation/reuse, anonymous isolation, validation, cancellation, deadline, recovery")
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
