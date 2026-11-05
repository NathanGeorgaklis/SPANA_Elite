# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV human-assisted target reacquisition instruction (Nathan, 2026-11-05 20:22): continuity of the
#                 downstream analysis across a user re-seed; the output contract (one event per episode increment).
"""Production run_rb.run with a checkpoint resolver: contract, checkpoint file, provenance, continuity."""
import json
import os
import shutil
import tempfile
import unittest

from player_tracker_rb import checkpoint as CK
from player_tracker_rb import run_rb as R
from tests_rb._synth_pipeline import box, clip, run_clip
from tests_rb._synth_scene import make_people

KIT_A = (200, 60, 40)


class CheckpointRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="r2_ckpt_run_")
        n1, n2 = 45, 120
        cls.people = make_people([("TARGET", KIT_A, 1.0), ("TARGET_ALT", KIT_A, 0.7)])
        cls.frames = clip({"TARGET": [box(320, 200)] * n1 + [None] * n2,
                           "TARGET_ALT": [None] * n1 + [box(322, 200)] * n2})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def resolver(self):
        frames = self.frames

        def answer(view, res):
            b = frames[view["frame"]].scene.boxes
            name = "TARGET" if "TARGET" in b else "TARGET_ALT"
            return {"decision": CK.SELECT, "box": list(b[name])}
        return answer

    def test_reseed_keeps_the_contract_and_writes_the_checkpoint_file(self):
        # the same-kit same-spot swap is only suspended by the handoff rule (optional mode; production default OFF)
        ov = {"continuity_step_enabled": True}
        e = run_clip(self.frames, self.people, os.path.join(self.tmp, "on"), name="on", overrides=ov,
                     checkpoint_resolver=self.resolver())
        base = run_clip(self.frames, self.people, os.path.join(self.tmp, "off"), name="off", overrides=ov)
        ev = e.summary["v2"]["reacquisition_events"]
        self.assertEqual([x["evidence"].get("source") for x in ev].count(CK.USER_RESEED), 1)
        with open(e.paths["checkpoints"]) as fh:
            ck = json.load(fh)
        self.assertEqual(ck["schema"], "r2_identity_checkpoints_v1")
        self.assertEqual(ck["checkpoints"][0]["status"], CK.RESEEDED)
        with open(e.paths["provenance"]) as fh:
            side = json.load(fh)
        import hashlib
        self.assertEqual(side["identity_checkpoints_sha256"],
                         hashlib.sha256(open(e.paths["checkpoints"], "rb").read()).hexdigest())
        k_re = ck["checkpoints"][0]["closed_frame"]
        sel = lambda doc, k: [t for t in doc["frame_log"][k - 1]["tracks"] if t.get("selected_player")][0]
        self.assertEqual([sel(base.doc, k) for k in range(1, k_re)], [sel(e.doc, k) for k in range(1, k_re)],
                         "pre-checkpoint frames are identical")
        trusted_after = [k for k in range(k_re, len(e.doc["frame_log"]) + 1)
                         if sel(e.doc, k)["v2"]["identity_trusted"]]
        self.assertEqual(len(trusted_after), len(e.doc["frame_log"]) + 1 - k_re, "the same T001 continues, trusted")
        self.assertEqual({sel(e.doc, k)["track_id"] for k in range(1, len(e.doc["frame_log"]) + 1)},
                         {sel(base.doc, 1)["track_id"]}, "one logical target for the whole run")
        self.assertFalse(os.path.exists(base.paths["checkpoints"]), "no checkpoint file without a resolver")


if __name__ == "__main__":
    unittest.main()
