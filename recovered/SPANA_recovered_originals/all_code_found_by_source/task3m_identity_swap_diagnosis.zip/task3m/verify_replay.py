"""Byte-level determinism check: instrumented replay debug_log vs scored Pass-1 frame_log."""
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))["frame_log"]
n = len(a)
same = sum(json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(a, b[:n]))
print(f"replayed frames: {n}; identical records: {same}/{n}")
