## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-03 | Added MCP compatibility bridge tests for `sequential-thinking` and `serena` after protocol mismatch diagnosis |

## Test Report: Codex MCP Tools (`sequential-thinking`, `serena`)

**Date**: 2026-03-03  
**Environment**: system Python (`Python 3.9.18`), Codex CLI `0.107.0-alpha.5`

### 1. Test Script Information

- Config file: `/root/.codex/config.toml`
- Bridge script: `/root/.codex/mcp_compat_bridge.py`
- Commands (reproducible):

```bash
# Confirm MCP registration after config update
codex mcp list

# Validate bridge syntax
python -m py_compile /root/.codex/mcp_compat_bridge.py

# End-to-end framed MCP probe (initialize -> tools/list -> tools/call)
python -u - <<'PY'
import subprocess, json, os, select, time

def send(proc, obj):
    body = json.dumps(obj, separators=(",", ":")).encode()
    packet = b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    proc.stdin.write(packet)
    proc.stdin.flush()

def recv(proc, timeout=12):
    buf = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.2)
        for s in r:
            chunk = os.read(s.fileno(), 65536)
            if not chunk:
                continue
            if s is proc.stderr:
                continue
            buf += chunk
            while True:
                sep = buf.find(b"\r\n\r\n")
                if sep == -1:
                    break
                headers = buf[:sep].decode("ascii", "ignore").split("\r\n")
                content_length = None
                for h in headers:
                    if h.lower().startswith("content-length:"):
                        content_length = int(h.split(":", 1)[1].strip())
                        break
                if content_length is None:
                    raise RuntimeError("missing content-length")
                body_start = sep + 4
                body_end = body_start + content_length
                if len(buf) < body_end:
                    break
                body = buf[body_start:body_end]
                buf = buf[body_end:]
                return json.loads(body.decode())
    raise TimeoutError("recv timeout")

def with_server(name, cmd, env=None, tool_call=None):
    print(f"\n== {name} ==")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "0"},
                },
            },
        )
        print("init:", recv(proc)["result"]["serverInfo"])
        send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = recv(proc)["result"]["tools"]
        print("tools", len(tools))
        if tool_call:
            send(proc, tool_call)
            print("call:", json.dumps(recv(proc), ensure_ascii=False)[:260])
    finally:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass

bridge = "/root/.codex/mcp_compat_bridge.py"
with_server(
    "sequential-thinking",
    ["/usr/bin/python3", bridge, "--", "/bin/mcp-server-sequential-thinking"],
    tool_call={
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "sequentialthinking",
            "arguments": {
                "thought": "protocol bridge health check",
                "nextThoughtNeeded": False,
                "thoughtNumber": 1,
                "totalThoughts": 1,
            },
        },
    },
)
with_server(
    "serena",
    [
        "/usr/bin/python3",
        bridge,
        "--",
        "/root/.local/bin/serena",
        "start-mcp-server",
        "--context",
        "codex",
        "--project",
        "/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM",
        "--transport",
        "stdio",
    ],
    env={**os.environ, "PATH": "/root/.local/bin:/usr/bin:/bin", "HOME": "/root"},
    tool_call={
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "get_current_config", "arguments": {}},
    },
)
PY
```

### 2. Validation Criteria

- `sequential-thinking` and `serena` must complete MCP `initialize` successfully.
- `tools/list` must return non-empty tool lists.
- At least one `tools/call` must succeed per server.
- No handshake timeout in framed protocol flow.

### 3. Test Results and Evidence

| Test Item | Result | Evidence |
|-----------|--------|----------|
| MCP config registration | PASS | `codex mcp list` shows both servers enabled and routed through `mcp_compat_bridge.py` |
| Bridge syntax check | PASS | `python -m py_compile /root/.codex/mcp_compat_bridge.py` returns `OK` |
| `sequential-thinking` initialize + list + call | PASS | `serverInfo={name: sequential-thinking-server}`, `tools=1`, `tools/call` returns thought state JSON |
| `serena` initialize + list + call | PASS | `serverInfo={name: FastMCP}`, `tools=21`, `get_current_config` returns active project config |

### Evidence Excerpts

- `== sequential-thinking == ... init: {'name': 'sequential-thinking-server', 'version': '0.2.0'} ... tools 1`
- `== serena == ... init: {'name': 'FastMCP', 'version': '1.23.0'} ... tools 21`
- `call response ... get_current_config ... Active project: Megatron-LM`
