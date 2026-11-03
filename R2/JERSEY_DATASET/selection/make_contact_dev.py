# SPANA R2 DEV jersey-dataset candidate selection v2: DEV-split blocks only (r2_protocol split), excluded windows
# (+60 s) and registry clips (+30 s) removed; times computed in the repo by r2_protocol and passed in plan JSON.
import json, os, subprocess, sys
home = os.path.expanduser("~"); W = os.path.join(home, "mnt/Downloads/_spana_jersey_work")
FILES = {"SYO_MAR": "videoplayback.mp4", "MCFC_MIT": "videoplayback (1).mp4"}
plan = json.load(open(sys.argv[1])); only = sys.argv[2]
for t in plan[only]:
    p = os.path.join(W, "contact_dev", f"{only}_{int(t*10):06d}.jpg")
    if os.path.exists(p): continue
    subprocess.run(["ffmpeg","-v","error","-y","-ss",str(t),"-i",os.path.join(home,"mnt/Downloads",FILES[only]),"-frames:v","1","-vf",
                    f"scale=320:180,drawtext=text='{only} {t:.1f}':x=4:y=4:fontsize=16:fontcolor=yellow:box=1:boxcolor=black@0.7",p], check=True)
print("DONE", only, len(plan[only]))
