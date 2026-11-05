# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2/JERSEY_DATASET/READER_EVAL_PREREGISTRATION.md §2 (first candidate reader: EasyOCR 1.7.2, pinned
#                 detector + recogniser weights, digit allowlist, preprocessing recorded and never changed per clip)
"""Reader step of the jersey evaluation: pinned EasyOCR over the crops listed in RUN_DIR/units.json.

    <venv with easyocr> python3 tools_rb/r2_jersey_ocr_easyocr.py RUN_DIR MODEL_DIR

Fails closed if a weight file's sha256 differs from the pin or the easyocr version differs. Writes
RUN_DIR/raw_reads.json: {"__reader__": pins + parameters, "<unit_id>": {"results": [[text, conf, [x1,y1,x2,y2]], ...]}}
with boxes in crop pixels. Policy-abstained units (box below jersey_min_box_h_px) are not read.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys

PINS = {"easyocr_version": "1.7.2",
        "craft_mlt_25k.pth": "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
        "english_g2.pth": "e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8"}
# preprocessing / decoding: library defaults except the digit allowlist (fixed before any output; never per clip)
PARAMS = {"lang_list": ["en"], "gpu": False, "allowlist": "0123456789", "detail": 1, "paragraph": False,
          "input": "file path of the PNG crop of the human person box (EasyOCR's canonical loading: RGB for detection, "
                   "greyscale for recognition); no resize / padding by this script"}


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(run_dir, model_dir):
    import cv2
    import easyocr
    import torch
    if easyocr.__version__ != PINS["easyocr_version"]:
        raise SystemExit(f"easyocr {easyocr.__version__} != pinned {PINS['easyocr_version']} (fail closed)")
    for n in ("craft_mlt_25k.pth", "english_g2.pth"):
        if _sha(os.path.join(model_dir, n)) != PINS[n]:
            raise SystemExit(f"{n}: sha256 differs from the pin (fail closed)")
    torch.manual_seed(0)
    torch.set_num_threads(max(1, (os.cpu_count() or 1)))
    reader = easyocr.Reader(PARAMS["lang_list"], gpu=PARAMS["gpu"], model_storage_directory=model_dir,
                            download_enabled=False, verbose=False)
    units = json.load(open(os.path.join(run_dir, "units.json")))["units"]
    out = {"__reader__": {"name": "EasyOCR", "source": "https://github.com/JaidedAI/EasyOCR (PyPI easyocr)",
                          "licence": "Apache-2.0 (library); english_g2 training data UNKNOWN", "pins": PINS,
                          "params": PARAMS, "torch": torch.__version__, "cv2": cv2.__version__,
                          "python": platform.python_version(), "platform": platform.platform(),
                          "pip_freeze": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                                       text=True).stdout.splitlines()}}
    for i, u in enumerate(units):
        if u["policy_abstain"]:
            continue
        res = reader.readtext(os.path.join(run_dir, u["crop"]), allowlist=PARAMS["allowlist"], detail=PARAMS["detail"], paragraph=PARAMS["paragraph"])
        rows = []
        for pts, text, conf in res:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            rows.append([text, float(conf), [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]])
        out[u["unit_id"]] = {"results": rows}
        if i % 50 == 0:
            print("read", i, len(units), flush=True)
    with open(os.path.join(run_dir, "raw_reads.json"), "x") as f:
        json.dump(out, f, indent=1)
    print("DONE", len(out) - 1, "units read")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
