# SPANA R2 DEV jersey-dataset candidate selection: contact sheets from ALLOWED windows only (read-only on sources).
import json, os, subprocess, sys
REG = {"SYO_MAR": ("videoplayback.mp4", 8525.9, [(1482.5,1492.5),(1535.0,1545.0),(2171.9,2181.9),(2622.0,2632.0),(2909.6,2919.6),(3220.3,3230.3),(3411.2,3421.2),(3627.6,3637.6),(3999.8,4009.8),(6352.8,6362.8),(7430.2,7440.2),(7514.9,7524.9),(8000.2,8010.2)]),
       "MCFC_MIT": ("videoplayback (1).mp4", 8840.0, [(2107.5,2117.5),(2143.7,2153.7),(3584.2,3594.2),(4100.8,4110.8),(4566.9,4576.9),(4635.6,4645.6),(6962.1,6972.1),(7028.1,7038.1),(7438.4,7448.4),(8053.0,8063.0),(8125.5,8135.5),(8252.3,8262.3)])}
BUF, STEP = 30.0, 30.0
home = os.path.expanduser("~"); W = os.path.join(home, "mnt/Downloads/_spana_jersey_work"); out = os.path.join(W, "contact")
plan = {}
for s, (f, dur, win) in REG.items():
    ts, t = [], 60.0
    while t < dur - 60:
        if not any(a - BUF <= t <= b + BUF for a, b in win): ts.append(t)
        t += STEP
    plan[s] = ts
    for t in ts:
        p = os.path.join(out, f"{s}_{int(t*10):06d}.jpg")
        if os.path.exists(p): continue
        subprocess.run(["ffmpeg","-v","error","-y","-ss",str(t),"-i",os.path.join(home,"mnt/Downloads",f),"-frames:v","1","-vf",
                        f"scale=320:180,drawtext=text='{s} {t:.0f}':x=4:y=4:fontsize=16:fontcolor=yellow:box=1:boxcolor=black@0.7",p], check=True)
json.dump({"buffer_s": BUF, "step_s": STEP, "times": plan, "excluded_registry_windows": {s: REG[s][2] for s in REG}}, open(os.path.join(W, "contact_plan.json"), "w"), indent=1)
print("DONE", {s: len(v) for s, v in plan.items()})
