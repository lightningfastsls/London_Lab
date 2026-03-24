"""Quick test: can we call claude -p from a subprocess on Windows?"""
import json
import os
import shutil
import subprocess
import sys
import threading
import queue

claude_path = shutil.which("claude")
print(f"Claude path: {claude_path}")

if not claude_path:
    print("ERROR: claude not found in PATH")
    sys.exit(1)

# Strip CLAUDECODE to allow nesting
env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

cmd = [
    claude_path,
    "-p", "just say hello in one word",
    "--output-format", "stream-json",
    "--verbose",
    "--max-turns", "1",
]

print(f"Running: {' '.join(cmd[:4])}...")

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=env,
)

# Thread-based reader
line_q = queue.Queue()

def reader(pipe, q):
    try:
        for raw_line in pipe:
            q.put(raw_line.decode("utf-8", errors="replace"))
    except Exception as e:
        q.put(f"ERROR: {e}")
    finally:
        q.put(None)

t_out = threading.Thread(target=reader, args=(process.stdout, line_q), daemon=True)
t_out.start()

stderr_lines = []
t_err = threading.Thread(target=lambda: stderr_lines.extend(
    process.stderr.read().decode("utf-8", errors="replace").splitlines()
), daemon=True)
t_err.start()

import time
start = time.time()
lines_received = 0
tool_uses = []

while time.time() - start < 30:
    try:
        line = line_q.get(timeout=1.0)
    except queue.Empty:
        if process.poll() is not None:
            break
        continue
    if line is None:
        break
    line = line.strip()
    if not line:
        continue
    lines_received += 1
    try:
        event = json.loads(line)
        etype = event.get("type", "?")
        if etype == "system":
            print(f"  [system] session started")
        elif etype == "assistant":
            msg = event.get("message", {})
            for c in msg.get("content", []):
                if c.get("type") == "text":
                    print(f"  [assistant] {c['text'][:100]}")
                elif c.get("type") == "tool_use":
                    tool_uses.append(c["name"])
                    print(f"  [tool_use] {c['name']}: {str(c.get('input',{}))[:80]}")
        elif etype == "result":
            print(f"  [result] cost=${event.get('cost_usd',0):.4f}")
        elif etype == "stream_event":
            se = event.get("event", {})
            se_type = se.get("type", "")
            if se_type == "content_block_start":
                cb = se.get("content_block", {})
                if cb.get("type") == "tool_use":
                    print(f"  [stream] tool_use start: {cb.get('name','?')}")
    except json.JSONDecodeError:
        print(f"  [raw] {line[:100]}")

process.wait()
t_err.join(timeout=2)

print(f"\nDone. Lines received: {lines_received}, exit code: {process.returncode}")
if stderr_lines:
    print(f"Stderr: {'; '.join(stderr_lines[:3])}")
if tool_uses:
    print(f"Tools invoked: {tool_uses}")
