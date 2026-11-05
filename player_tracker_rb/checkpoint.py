# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV human-assisted target reacquisition instruction (Nathan, 2026-11-05 20:22): keep handoff
#                 safety, states and thresholds; when identity stays unresolved beyond the normal safe window emit an
#                 IDENTITY_CHECKPOINT (frame/time, candidate boxes, tracker context); the user may select the player;
#                 the box becomes a new trusted seed only after normal seed validation; appearance evidence restarts
#                 from that point (no contaminated bank sample crosses the re-seed); pre-checkpoint output is kept and
#                 the same player analysis continues; ambiguous selections are refused; no automatic candidate is ever
#                 promoted by a checkpoint.
"""Human-assisted target reacquisition (NEW_R2). Opt-in: used only when a checkpoint resolver is given to run_rb.run.

IdentityCheckpointer wraps the unchanged TargetTracker. Every frame goes through TargetTracker.update first, exactly as
without checkpoints; the wrapper only watches the result.

CHECKPOINT. When the target's identity has been untrusted (identity_trusted false) continuously for
checkpoint_unresolved_s (= max_ambiguous_s + reacq_window_s: the automatic path's own age-out + verification window),
an IDENTITY_CHECKPOINT opens with: frame, time, the frame where trust was last held, the tracker context (state,
reasons, episode, last confirmed frame, reported box) and every current body detection (box, tracklet, flags). While
it is open the resolver is asked once per frame with the current frame's candidates. It closes as:
  * AUTO_RESOLVED: the tracker regained trust on its own (nothing the checkpoint did);
  * DECLINED: the user says the target is not visible / cannot be identified; a new checkpoint may open only after a
    further checkpoint_unresolved_s of unresolved identity;
  * RESEEDED: a user selection passed validation (below).
Answers that fail validation are REFUSED and recorded; the checkpoint stays open for another answer.

USER SELECTION = A NEW HUMAN SEED, NOTHING ELSE. The selected box is validated exactly like the run's initial seed by
a FRESH TargetTracker's seed() (D18 IoU / NEW_R2 containment rules, merged checks, usable crop, bank anchor), after a
re-seed ambiguity pre-check: refused if more than one body matches the box (IoU >= seed_min_iou), or if another body
outside the matched body's nested group lies >= seed_contain_competitor_max inside it. A seed whose bank anchor would
be deferred (F2 pending) is refused (the user is asked again on a later frame): a re-seed is accepted only when the
selected crop anchors the new bank at once. The pre-check also applies the deferred-seed body rules on every re-seed
(review 2026-11-05 #3): the matched detection must have its own population tracklet, no box nested with it may carry
another tracklet, and its tracklet must have no merge lineage. A malformed answer (not 4 finite numbers with positive
size) is REFUSED. On acceptance the old TargetTracker (its bank, contrast history, known others, handoff suspicion,
NOT_TARGET memory, motion state) is discarded entirely: no appearance sample crosses the re-seed. The population
(generic people tracks and their banks) continues unchanged.

PHYSICAL ENTITY (rule 28; review #1/#2). Population tracklets the target was confirmed on before a re-seed remain
the target's former body chains: they stay in target_tracklets (so run_rb never writes the target's own coasting
tracklet as a second person; a former tracklet OBSERVED elsewhere is written, as for the ordinary tracker), and those
NOT observed at the re-seed frame are given to the fresh tracker as its own tracklets, so their predicted boxes do not
count as another body against the selected crop. Only tracklet ids cross the re-seed: no appearance sample.
The re-seed frame is a human identification and, like the run's first seed, carries no camera-continuity or
continuity-step proof requirement. No checkpoint opens while the run's initial seed anchor is still pending (F2).

OUTPUT CONTINUITY. Frames before the re-seed are untouched. The re-seed frame is the fresh tracker's seed frame
(TRACKING, trusted) as the next target episode; a ReacquisitionEvent with evidence.source = USER_CONFIRMED_RESEED keeps
the document rule "one event per episode increment". Later episodes of the new tracker are offset accordingly.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .appearance import AppearanceExtractor
from .config import TrackerConfig, sec_ge
from .detector import box_iou_matrix
from . import population as P
from . import target as T

OPEN, AUTO_RESOLVED, DECLINED, RESEEDED = "OPEN", "AUTO_RESOLVED", "DECLINED", "RESEEDED"
SELECT, NOT_VISIBLE = "SELECT", "NOT_VISIBLE"
USER_RESEED = "USER_CONFIRMED_RESEED"
# resolver(checkpoint_view, frame_result) -> None (no answer this frame) | {"decision": "NOT_VISIBLE"} |
#                                            {"decision": "SELECT", "box": [x1, y1, x2, y2]}
Resolver = Callable[[Dict[str, Any], P.FrameResult], Optional[Dict[str, Any]]]


def _candidates(res: P.FrameResult) -> List[Dict[str, Any]]:
    return [{"detection_index": o.index, "box": [float(v) for v in o.bbox], "tracklet_id": o.tracklet_id,
             "flags": sorted(o.flags)} for o in res.observations if o.is_body]


class IdentityCheckpointer:
    """TargetTracker + identity checkpoints + user-confirmed re-seeds (see the module docstring)."""

    def __init__(self, cfg: TrackerConfig, population: P.Population, extractor: AppearanceExtractor,
                 resolver: Resolver):
        self._cfg = cfg
        self._pop = population
        self._ext = extractor
        self._resolver = resolver
        self.tracker = T.TargetTracker(cfg, population, extractor)
        self._offset = 0                                  # episode offset of the current TargetTracker
        self._archived: List[T.ReacquisitionEvent] = []   # events of discarded trackers (offset) + re-seed events
        self.checkpoints: List[Dict[str, Any]] = []
        self._open: Optional[Dict[str, Any]] = None
        self._unres: Optional[Tuple[int, float]] = None   # (frame, time) since identity is untrusted
        self._arm_t: Optional[float] = None               # unresolved-time origin for the next checkpoint
        self._former: set = set()                          # target tracklets of discarded trackers (rule 28)

    # -- delegation used by run_rb ----------------------------------------------------------------------------
    @property
    def target_tracklets(self):
        return tuple(sorted(set(self.tracker.target_tracklets) | self._former))

    @property
    def current_tracklet(self):
        return self.tracker.current_tracklet

    @property
    def verifying_tracklet(self):
        return self.tracker.verifying_tracklet

    @property
    def reacquisition_events(self) -> Tuple[T.ReacquisitionEvent, ...]:
        cur = tuple(dataclasses.replace(e, episode=e.episode + self._offset) for e in self.tracker.reacquisition_events)
        return tuple(self._archived) + cur

    def state_sizes(self) -> Dict[str, int]:
        d = dict(self.tracker.state_sizes())
        d["identity_checkpoints"] = len(self.checkpoints)
        return d

    def as_record(self) -> Dict[str, Any]:
        return {"schema": "r2_identity_checkpoints_v1", "checkpoint_unresolved_s": self._cfg.get("checkpoint_unresolved_s"),
                "checkpoints": self.checkpoints}

    # -- frames -----------------------------------------------------------------------------------------------
    def seed(self, res: P.FrameResult, image: np.ndarray, seed_bbox: Sequence[float]) -> T.TargetFrame:
        return self.tracker.seed(res, image, seed_bbox)

    def _shift(self, tf: T.TargetFrame) -> T.TargetFrame:
        return tf if self._offset == 0 else dataclasses.replace(tf, target_episode=tf.target_episode + self._offset)

    def update(self, res: P.FrameResult, image: np.ndarray) -> T.TargetFrame:
        tf = self._shift(self.tracker.update(res, image))     # the automatic tracker always decides first
        t, f = res.time_s, res.rel_frame
        if tf.identity_trusted:
            if self._open is not None:
                self._close(AUTO_RESOLVED, f, t)
            self._unres, self._arm_t = None, None
            return tf
        if self._unres is None:
            self._unres = (f, t)
            self._arm_t = t
        if self._open is None and self.tracker._seed_pending is None and \
                sec_ge(t - self._arm_t, self._cfg.get("checkpoint_unresolved_s")):
            self._open_checkpoint(res, tf)
        if self._open is None:
            return tf
        try:
            ans = self._resolver(self._view(res, tf), res)
        except Exception as e:                            # a broken UI never changes identity: record and continue
            self._open["answers"].append({"frame": f, "time_s": t, "decision": None, "box": None,
                                          "outcome": "REFUSED", "reason": f"resolver raised {type(e).__name__}: {e}"})
            return tf
        if ans is None:
            return tf
        box = ans.get("box") if isinstance(ans, dict) else None
        ok_box = isinstance(box, (list, tuple)) and len(box) == 4 and \
            all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box) and \
            box[2] > box[0] and box[3] > box[1]
        rec = {"frame": f, "time_s": t, "decision": ans.get("decision") if isinstance(ans, dict) else repr(ans),
               "box": [float(v) for v in box] if ok_box else None}
        self._open["answers"].append(rec)
        if not isinstance(ans, dict):
            rec.update(outcome="REFUSED", reason="malformed answer")
            return tf
        if ans.get("decision") == NOT_VISIBLE:
            rec["outcome"] = "ACCEPTED"
            self._close(DECLINED, f, t)
            self._arm_t = t
            return tf
        if ans.get("decision") != SELECT or rec["box"] is None:
            rec["outcome"] = "REFUSED"
            rec["reason"] = "malformed answer (expected SELECT with 4 finite numbers of positive size, or NOT_VISIBLE)"
            return tf
        return self._reseed(res, image, tf, rec)

    # -- internals --------------------------------------------------------------------------------------------
    def _open_checkpoint(self, res: P.FrameResult, tf: T.TargetFrame) -> None:
        self._open = {"checkpoint_id": len(self.checkpoints) + 1, "kind": "IDENTITY_CHECKPOINT", "status": OPEN,
                      "opened_frame": res.rel_frame, "opened_time_s": res.time_s,
                      "unresolved_since_frame": self._unres[0], "unresolved_since_s": self._unres[1],
                      "context": {"rb_state": tf.rb_state, "reasons": list(tf.identity_uncertainty_reasons),
                                  "target_episode": tf.target_episode, "last_confirmed_frame": tf.last_confirmed_frame,
                                  "reported_box": [float(v) for v in tf.bbox], "position_valid": tf.position_valid,
                                  "current_tracklet": self.tracker.current_tracklet},
                      "candidates": _candidates(res), "answers": []}
        self.checkpoints.append(self._open)

    def _view(self, res: P.FrameResult, tf: T.TargetFrame) -> Dict[str, Any]:
        v = {k: self._open[k] for k in ("checkpoint_id", "opened_frame", "opened_time_s", "context")}
        v.update(frame=res.rel_frame, time_s=res.time_s, candidates=_candidates(res),
                 reported_box=[float(x) for x in tf.bbox], rb_state=tf.rb_state)
        return v

    def _close(self, status: str, f: int, t: float) -> None:
        self._open.update(status=status, closed_frame=f, closed_time_s=t)
        self._open = None

    def _selection_problem(self, res: P.FrameResult, box: Sequence[float], probe: T.TargetTracker) -> Optional[str]:
        """The re-seed ambiguity pre-check (module docstring). None = not ambiguous by this check."""
        c = self._cfg
        bodies = [o for o in res.observations if o.is_body]
        if not bodies:
            return "no person detection at this frame"
        sb = np.asarray([[float(v) for v in box]])
        ious = box_iou_matrix(sb, np.asarray([o.bbox for o in bodies]))[0]
        if int(np.sum(ious >= c.get("seed_min_iou"))) > 1:
            return "ambiguous selection: more than one detection matches the box"
        k = int(np.argmax(ious))
        if ious[k] < c.get("seed_min_iou"):
            # containment path: seed() decides which single detection the box holds; check that body's group
            geo = [T._seed_geometry(box, o.bbox) for o in bodies]
            inside = [i for i, g in enumerate(geo) if g["contain"] >= c.get("seed_contain_min")]
            if len(inside) != 1:
                return None                                  # seed() refuses or decides; nothing more to check
            k = inside[0]
        o = bodies[k]
        grp = probe._group(res, o.index)
        for q in bodies:
            if q.index not in grp and \
                    T._seed_geometry(box, q.bbox)["contain"] >= c.get("seed_contain_competitor_max"):
                return (f"ambiguous selection: another body (detection {q.index}) lies at least "
                        f"{c.get('seed_contain_competitor_max')} inside the box")
        # the deferred-seed body rules (target._seed_body_tracklet), applied to every re-seed (review #3)
        if o.tracklet_id is None:
            return "ambiguous selection: the box matches a nested / alternative detection without its own track"
        tids = {res.observations[j].tracklet_id for j in grp if res.observations[j].tracklet_id}
        if tids != {o.tracklet_id}:
            return f"ambiguous selection: the selected body's nested group holds other tracks {sorted(tids - {o.tracklet_id})}"
        v = res.tracklet(o.tracklet_id)
        if v is not None and P.MERGE_LINEAGE in v.flags:
            return "ambiguous selection: the selected body's track has merge lineage (may hold two people)"
        return None

    def _reseed(self, res: P.FrameResult, image: np.ndarray, tf: T.TargetFrame,
                rec: Dict[str, Any]) -> T.TargetFrame:
        fresh = T.TargetTracker(self._cfg, self._pop, self._ext)
        former = self._former | set(self.tracker.target_tracklets)
        observed = {v.tracklet_id for v in res.tracklets if v.observed}
        # rule 28: the target's former coasting body chains are its own, not another body against the selected crop
        fresh._target_tracklets |= {t for t in former if t not in observed}
        why = self._selection_problem(res, rec["box"], fresh)
        if why is None:
            try:
                fr = fresh.seed(res, image, rec["box"])
            except T.SeedError as e:
                why = f"seed validation refused: {e}"
            else:
                if not fr.identity_trusted:
                    why = "the selected crop cannot anchor a new appearance bank on this frame (deferred anchor)"
        if why is not None:
            rec.update(outcome="REFUSED", reason=why)
            return tf
        cp = self._open
        ep = tf.target_episode + 1
        self._archived.extend(dataclasses.replace(e, episode=e.episode + self._offset)
                              for e in self.tracker.reacquisition_events)
        uf, ut = self._unres
        lf = self.tracker._lost_from_frame
        self._archived.append(T.ReacquisitionEvent(
            episode=ep, lost_from_frame=uf if lf is None else lf, uncertain_from_frame=uf, reacq_started_frame=cp["opened_frame"],
            reacq_accepted_frame=res.rel_frame, uncertain_duration_s=res.time_s - ut,
            evidence={"source": USER_RESEED, "checkpoint_id": cp["checkpoint_id"],
                      "contrast_verified": True, "contrast_available": False,
                      "identity_basis": "HUMAN_IDENTIFICATION (validated seed)",
                      "lost_from_frame_is_trust_loss": lf is None,
                      "answer_delay_s": res.time_s - cp["opened_time_s"],
                      "seed": {k: fr.decision.get(k) for k in ("seed_rule", "seed_iou", "seed_containment")},
                      "appearance_bank": "RESTARTED (previous bank discarded)"}))
        self._former = former
        self.tracker = fresh
        self._offset = ep - 1
        rec["outcome"] = "ACCEPTED"
        self._close(RESEEDED, res.rel_frame, res.time_s)
        self._unres, self._arm_t = None, None
        dec = dict(fr.decision, identity_checkpoint={"checkpoint_id": cp["checkpoint_id"], "transition": USER_RESEED})
        return dataclasses.replace(fr, target_episode=ep, reidentified_this_frame=True, decision=dec)
