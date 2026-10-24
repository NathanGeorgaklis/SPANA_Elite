"""Offline TeamClassifier replay. Reproduces, call for call, the
observe_tracked_player / classify sequence a Pass-2 run made (from a
diag_other_player_capture JSON), on the same video frames, so candidate
classifier changes can be measured in seconds instead of re-running the
whole engine. Fidelity is checked by replaying the CURRENT classifier and
requiring a label-for-label match with the captured run.
"""
import json
import cv2
from football_analysis import possession_detector as pd
from football_analysis.colour import shirt_color_hist


def load_frames(video, start, n):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def replay(capture, video, start_frame, init_bbox, classifier_cls=pd.TeamClassifier, frames=None, box_filter=None):
    """Returns list of (frame, box, label, conf) in original call order.
    box_filter(frame_idx, box, tracked_bbox, frame_shape) -> bool: optional,
    simulates removing a box from the other-player set before classification."""
    n = max(m["frame"] for m in capture["merge"]) + 1
    frames = frames or load_frames(video, start_frame, n)
    tc = classifier_cls(shirt_color_hist(frames[0], init_bbox))
    by_frame = {}
    for c in capture["classify"]:
        by_frame.setdefault(c["frame"], []).append(c)
    out = []
    for m in capture["merge"]:
        k = m["frame"]
        tb = m["tracked_bbox"]
        tc.observe_tracked_player(frames[k], tb)
        for c in by_frame.get(k, []):
            if box_filter is not None and not box_filter(k, c["box"], tb, frames[k].shape):
                out.append((k, c["box"], "EXCLUDED", 0.0))
                continue
            team, conf = tc.classify(frames[k], c["box"], frame_number=k)
            out.append((k, c["box"], team, round(float(conf), 4)))
    return out, frames
