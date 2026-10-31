"""Probe: persistent union + third walker, optional camera pan / zoom. Population level only."""
import sys
sys.path.insert(0, sys.argv[1])
from player_tracker_rb import population as P
from tests_rb._synth_people import Frame, box_at, cand, run, pan, still

def sh(b, dx):
    return (b[0] + dx, b[1], b[2] + dx, b[3])

def scene(fps, pan_px_s=0.0, third_start=200.0, third_speed=150.0, third_y=360, union_s=5.0, third_t=(1.5, 3.0)):
    fr = []
    off = 0.0
    d = pan_px_s / fps
    def cam():
        return pan(d) if d else None
    for _ in range(int(fps)):
        off += d
        fr.append(Frame([cand(sh(box_at(400, 360, 32, 80), off)), cand(sh(box_at(412, 360, 32, 80), off))],
                        [("TARGET",), ("OTHER",)], camera=cam()))
    for k in range(int(union_s * fps)):
        t = k / fps
        off += d
        dets = [cand(sh((384.0, 320.0, 428.0, 400.0), off))]
        own = [("TARGET", "OTHER")]
        if third_t[0] <= t <= third_t[1]:
            x = third_start + (t - third_t[0]) * third_speed
            dets.append(cand(sh(box_at(x, third_y, 32, 80), off)))
            own.append(("THIRD",))
        fr.append(Frame(dets, own, camera=cam()))
    fr[0].camera = None
    return fr

def clean_union(rn):
    out = []
    for k, (r, f) in enumerate(zip(rn.results, rn.frames)):
        for v in r.tracklets:
            if v.observed and len(f.owners[v.detection_index]) > 1 and P.MERGE_LINEAGE not in v.flags \
                    and P.MERGED_SUSPECTED not in r.observations[v.detection_index].flags:
                out.append(k)
    return out

def main():
    for fps in (30.0, 15.0):
        for pan_s in (0.0, 100.0, -100.0, 200.0, -200.0):
            for start, speed in ((200.0, 150.0), (650.0, -150.0)):
                rn = run(scene(fps, pan_s, start, speed), fps=fps)
                cu = clean_union(rn)
                print(f"fps={fps} pan={pan_s} third_from={start}: clean_union_frames={len(cu)} first={cu[:3]}")

if __name__ == "__main__":
    main()
