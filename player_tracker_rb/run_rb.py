# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R0/R0_R1_TRACKER_REBUILD_PLAN.md item 10 (run_rb.run: the historical run_mo signature minus
#                 resume); R0/R0_TRACKER_INTERFACE_MAP.md §1-§5 (Pass-1 JSON contract, event_adapter and
#                 actor_attribution consumption); R1/R1_DESIGN_DECISIONS.md D01, D05-D07, D16, D18-D20;
#                 R1-G instruction (Nathan, 2026-10-25)
"""
player_tracker_rb.run_rb: the integrated R1 Pass-1 tracker.

    run(video_path, start_frame, init_box, jersey_number=None, max_frames=None, out_dir=..., debug=False,
        config_overrides=None) -> summary dict

Pipeline per frame (no event logic; nothing here knows about teams, numbers, clips or benchmarks):

    SequentialVideoReader (video_io)          one positioning, then strictly sequential decode, fail closed
      -> DetectorBackend.detect               generic PersonCandidates
      -> CameraMotionEstimator.update         global image motion, hard-cut detection
      -> Population.update                    generic tracklets, merges, contacts, competitors
      -> TargetTracker.seed / update          contrastive identity of the ONE seeded target
      -> records (output_schema builders)     TARGET record (T001) + POPULATION records + frame v2 block
      -> streamed canonical JSON              `{video_basename}_mo_tracking_results.json`
    plus the provenance sidecar `{video_basename}_rb_video_provenance.json` and, with debug=True, the per-frame
    decision log `{video_basename}_rb_decision_log.jsonl`.

The seed frame (relative frame 0) is the initialization frame, not a frame_log step: frame_log frame f is video
frame start_frame + f, f = 1, 2, ... (contract). The run is REFUSED (RunRefusedError, nothing written) when the
seed cannot start a trustworthy target (D18).

Record rules (the TARGET record follows output_schema.target_invariant_errors; NEW semantics are listed in the
summary v2 notes and in R1/R1G/R1G_OUTPUT_MAPPING.md):
  * Target: track_id = identity_id = online_track_id = target_track_id (T001) for the whole run (D07). Historical
    `bbox` is the LAST CONFIRMED (observed) box; `observed_position_box` is this frame's observed, predicted or
    held box (D06; frame 86 of the frozen historical exemplar HE1). identity_confidence = identity_confidence_rb
    when trusted, else at most untrusted_identity_confidence_ceiling (D19). overall_track_confidence (which
    event_adapter turns into the snapshot tracking_confidence) = identity_confidence.
  * PHYSICAL-ENTITY EXCLUSION (project rule 28): a population tracklet that is, or may be, the target this frame
    (every tracklet the target was confirmed on, and the tracklet the target's decision is about this frame) is
    NOT written as a separate person. In overlap_contests it is written as the target id.
  * Population: every other live tracklet, with the historical non-selected triple (detected -> ACTIVE/NONE/
    CONFIRMED_DETECTION, coasting -> TEMPORARILY_LOST/FULL/PREDICTED). No identity claim: identity_confidence 0.
  * overlap_contests: every overlapping pair of bodies (population overlap context), as written track ids.
  * identity_conflict_counts: 'T001|Txxx' -> number of frames in which that tracked person was the competing
    hypothesis that kept the target from being confirmed (F9: conflicts were invisible in the historical log).
    Surviving actor_attribution halves its actor confidence when any conflict involves the target.
  * final_tracks: the last written record of every track (no v2 block); a track absent from the last frame is
    TERMINATED.

Streaming and memory: frame records are serialised as they are produced into a temporary file next to the output
and the final document is assembled at the end (byte-identical to output_schema.dumps_canonical of the whole
document). Nothing grows with the clip except what is bounded by the number of distinct tracks (final_tracks)
and of reacquisition events. Every frame record is validated against the R1 contract before it is written; the
summary, final_tracks and the document-level rules are validated at the end. Any violation raises
ContractViolationError and leaves no output file (fail closed).

Determinism: identical inputs, code and config give a byte-identical document except the wall-clock fields
listed in WALL_CLOCK_FIELDS, which compare_documents() reports separately.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from . import __version__ as TRACKER_VERSION
from . import output_schema as S
from . import population as P
from . import provenance as PV
from . import target as T
from .appearance import AppearanceExtractor, EmbeddingBackend
from .camera import CUT, CameraMotionEstimator
from .config import TrackerConfig
from .detector import DetectorBackend
from .video_io import SequentialVideoReader

RESULTS_SUFFIX = "_mo_tracking_results.json"
DECISION_LOG_SUFFIX = "_rb_decision_log.jsonl"
PARTIAL_SUFFIX = ".rb_partial"
# Fields that hold wall-clock facts; everything else must be byte-identical between runs on the same inputs.
WALL_CLOCK_FIELDS: Tuple[Tuple[str, ...], ...] = (("summary", "processing_time_s"),
                                                  ("summary", "v2", "provenance", "run_started_utc"))
APPEARANCE_EMBEDDER_FOR = {"composite_lab_dinov2": "dinov2_onnx"}


class RunRefusedError(RuntimeError):
    """The run cannot start (bad arguments, backends, seed or provenance). Nothing is written."""


class ContractViolationError(RuntimeError):
    """An output record broke the R1 contract. The run stops and nothing is written (fail closed)."""


# ---------------------------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------------------------
def build_backends(cfg: TrackerConfig, detector: Optional[DetectorBackend] = None,
                   embedder: Optional[EmbeddingBackend] = None) -> Tuple[DetectorBackend, EmbeddingBackend]:
    """The configured detector and embedding backends. A backend may be passed in (tests, or to reuse a loaded
    model), but it must be the one the config names: a production config never runs a substitute model, and a
    'scripted' config never runs without its scripted backend (fail closed; project rules 20-21)."""
    db = cfg.get("detector_backend")
    if detector is None:
        if db != "yolox_onnx":
            raise RunRefusedError(f"detector_backend {db!r} needs a detector to be passed in")
        from .detector_yolox import YoloxOnnxDetector
        detector = YoloxOnnxDetector(cfg)
    name = detector.provenance().get("backend")
    if db == "yolox_onnx" and name != "yolox_onnx":
        raise RunRefusedError(f"config detector_backend yolox_onnx but the detector passed in is {name!r}")
    if db == "scripted" and name == "yolox_onnx":
        raise RunRefusedError("config detector_backend scripted but a yolox_onnx detector was passed in")
    ab = cfg.get("appearance_backend")
    want = APPEARANCE_EMBEDDER_FOR.get(ab)
    if embedder is None:
        if want is None:
            raise RunRefusedError(f"appearance_backend {ab!r} needs an embedding backend to be passed in")
        from .embedding_dinov2 import Dinov2OnnxEmbedding
        embedder = Dinov2OnnxEmbedding(cfg)
    ename = embedder.provenance().get("backend")
    if want is not None and ename != want:
        raise RunRefusedError(f"config appearance_backend {ab} needs embedding backend {want}, got {ename!r}")
    if want is None and ename in APPEARANCE_EMBEDDER_FOR.values():
        raise RunRefusedError(f"config appearance_backend {ab!r} but a production embedder ({ename}) was passed in")
    return detector, embedder


# ---------------------------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------------------------
def _clip01(x: Optional[float]) -> float:
    return 0.0 if x is None else float(min(1.0, max(0.0, x)))


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


class _RecordBuilder:
    """Turns one frame's population result and target decision into contract records. Keeps only per-track state
    whose size is bounded by the number of live tracks (last observed boxes), plus the last written record of every
    track ever written (final_tracks)."""

    def __init__(self, cfg: TrackerConfig, seed_box: Sequence[float]):
        self._cfg = cfg
        self._tid = cfg.get("target_track_id")
        self._last_confirmed_box: Tuple[float, ...] = tuple(float(v) for v in seed_box)
        self._pop_last_observed: Dict[str, Tuple[float, ...]] = {}
        self.final_records: Dict[str, Dict[str, Any]] = {}
        self.population_ids_seen: Set[str] = set()
        self.population_ids_written: Set[str] = set()
        self.conflicts: Dict[str, int] = {}
        self.conflict_frames = 0
        self.lifetimes: Dict[str, Tuple[int, int]] = {}      # written population id -> (first, last) frame
        self.provisional_ids: Set[str] = set()

    # -- target ------------------------------------------------------------------------------------
    def target_record(self, f: int, tf: T.TargetFrame) -> Dict[str, Any]:
        c = self._cfg
        hist, occ, pos = S.TARGET_TRIPLE_BY_RB_STATE[(tf.rb_state, tf.predicted_inside_image
                                                      if tf.rb_state == T.LOST else True)]
        observed = tf.rb_state == T.TRACKING
        if observed:
            self._last_confirmed_box = tuple(tf.bbox)
        lc = tf.last_confirmed_frame if tf.last_confirmed_frame is not None else 0
        fso = 0 if observed else f - lc
        idc = tf.identity_confidence_rb if tf.identity_trusted else \
            min(tf.identity_confidence_rb, c.get("untrusted_identity_confidence_ceiling"))
        conf = {"team_confidence": 0.0,
                "motion_confidence": 1.0 if tf.position_valid else 0.0,
                "appearance_confidence": _clip01(tf.target_similarity),
                "reid_confidence": _clip01(tf.target_similarity),
                "overall_track_confidence": idc,
                "visibility_confidence": 1.0 if observed else 0.0,
                "position_confidence": 1.0 if observed else 0.0,
                "track_continuity_confidence": 1.0 if observed else 0.0,
                "identity_confidence": idc}
        rec = S.build_track_record(
            c, track_id=self._tid, state=hist, age=f, last_seen_frame=lc, bbox=self._last_confirmed_box,
            observed_position_box=tf.bbox, occlusion_status=occ, position_source=pos, selected_player=True,
            confidences=conf, frames_since_observed_position=fso, frames_since_confirmed_identity=fso,
            v2=tf.v2_fields())
        self.final_records[self._tid] = rec
        return rec

    # -- population --------------------------------------------------------------------------------
    def population_records(self, f: int, res: P.FrameResult, excluded: Set[str]) -> List[Dict[str, Any]]:
        c = self._cfg
        out = []
        live = {v.tracklet_id for v in res.tracklets}
        for v in res.tracklets:
            self.population_ids_seen.add(v.tracklet_id)
            if v.observed_box is not None:
                self._pop_last_observed[v.tracklet_id] = tuple(v.observed_box)
            if v.tracklet_id in excluded:
                continue
            detected = v.observed
            score = None
            if detected and v.detection_index is not None:
                score = float(res.observations[v.detection_index].score)
            hist, occ, pos = S.POPULATION_TRIPLE_DETECTED if detected else S.POPULATION_TRIPLE_COASTING
            last_obs = self._pop_last_observed.get(v.tracklet_id, tuple(v.box))
            fso = f - v.last_observed_frame
            conf = {"team_confidence": 0.0, "motion_confidence": 1.0 if detected else 0.0,
                    "appearance_confidence": 0.0, "reid_confidence": 0.0,
                    "overall_track_confidence": _clip01(score) if detected else 0.0,
                    "visibility_confidence": 1.0 if detected else 0.0,
                    "position_confidence": 1.0 if detected else 0.0,
                    "track_continuity_confidence": 1.0 if detected else 0.0, "identity_confidence": 0.0}
            v2 = {"record_kind": "POPULATION", "detected_this_frame": detected, "raw_detection_conf": score,
                  "is_competitor": bool(v.is_competitor), "first_frame": int(v.first_frame)}
            rec = S.build_track_record(
                c, track_id=v.tracklet_id, state=hist, age=f - v.first_frame, last_seen_frame=v.last_observed_frame,
                bbox=last_obs, observed_position_box=v.box, occlusion_status=occ, position_source=pos,
                selected_player=False, confidences=conf, frames_since_observed_position=fso,
                frames_since_confirmed_identity=fso, v2=v2)
            out.append(rec)
            self.final_records[v.tracklet_id] = rec
            self.population_ids_written.add(v.tracklet_id)
            first, _ = self.lifetimes.get(v.tracklet_id, (f, f))
            self.lifetimes[v.tracklet_id] = (first, f)
            if v.provisional:
                self.provisional_ids.add(v.tracklet_id)
            else:
                self.provisional_ids.discard(v.tracklet_id)
        for gone in [k for k in self._pop_last_observed if k not in live]:
            del self._pop_last_observed[gone]                # bounded by the live tracks
        return out

    def state_sizes(self) -> Dict[str, int]:
        """Per-track state kept by the writer: bounded by live tracks, or by distinct tracks ever written."""
        return {"writer_live_last_observed": len(self._pop_last_observed),
                "writer_final_records": len(self.final_records), "writer_lifetimes": len(self.lifetimes),
                "writer_provisional_ids": len(self.provisional_ids), "writer_conflict_pairs": len(self.conflicts),
                "writer_population_ids_seen": len(self.population_ids_seen)}

    # -- frame context -----------------------------------------------------------------------------
    def written_id(self, ref: str, res: P.FrameResult, excluded: Set[str]) -> Optional[str]:
        """A population ref ('D<i>' detection ref or a tracklet id) as the track id written this frame."""
        tid = ref
        if ref.startswith("D") and ref[1:].isdigit():
            i = int(ref[1:])
            tid = res.observations[i].tracklet_id if 0 <= i < len(res.observations) else None
        if tid is None:
            return None
        return self._tid if tid in excluded else tid

    def overlap_contests(self, res: P.FrameResult, excluded: Set[str], written: Set[str]) -> List[List[str]]:
        pairs = set()
        for op in res.overlap_pairs:
            a, b = self.written_id(op.a, res, excluded), self.written_id(op.b, res, excluded)
            if a is None or b is None or a == b or a not in written or b not in written:
                continue
            pairs.add(tuple(sorted((a, b))))
        return [list(p) for p in sorted(pairs)]

    def record_conflicts(self, tf: T.TargetFrame, res: P.FrameResult, excluded: Set[str],
                         written: Set[str]) -> None:
        """Count the competing hypotheses that kept the target from TRACKING this frame (F9)."""
        if tf.rb_state == T.TRACKING:
            return
        names: Set[str] = set()
        for cand in tf.decision.get("candidates", []):
            if cand.get("passed") and tf.rb_state != T.REACQUIRING:
                continue
            refs = [cand.get("best_competitor")] if cand.get("best_competitor") else []
            refs += list(cand.get("unknown_contacts") or [])
            for r in refs:
                w = self.written_id(str(r), res, excluded)
                if w is not None and w != self._tid and w in written:
                    names.add(w)
        rivals = tf.decision.get("reacquisition", {}).get("rivals") or []
        for i in rivals:
            w = self.written_id(P.det_ref(int(i)), res, excluded)
            if w is not None and w != self._tid and w in written:
                names.add(w)
        if names:
            self.conflict_frames += 1
            for n in sorted(names):
                key = f"{self._tid}|{n}"
                self.conflicts[key] = self.conflicts.get(key, 0) + 1


# ---------------------------------------------------------------------------------------------
# Streaming document writer and incremental document checks
# ---------------------------------------------------------------------------------------------
class _StreamingDocument:
    """Writes frame records as they are produced; assembles the canonical document at the end."""

    def __init__(self, out_path: str, chunk_bytes: int):
        self._chunk = chunk_bytes
        if os.path.exists(out_path):
            raise RunRefusedError(f"refusing to overwrite existing output {out_path}")
        self.out_path = out_path
        self._partial = out_path + PARTIAL_SUFFIX
        if os.path.exists(self._partial):
            raise RunRefusedError(f"a partial output from an earlier run exists: {self._partial}")
        self._fh = open(self._partial, "x", encoding="ascii")
        self.n = 0

    def add_frame(self, rec: Dict[str, Any]) -> None:
        if self.n:
            self._fh.write(",")
        self._fh.write(_dumps(rec))
        self.n += 1

    def finish(self, summary: Dict[str, Any], final_tracks: List[Dict[str, Any]]) -> str:
        """Assemble, write atomically, and return the SHA-256 of the written document."""
        self._fh.close()
        tmp = self.out_path + PARTIAL_SUFFIX + ".doc"
        h = hashlib.sha256()
        with open(tmp, "x", encoding="ascii") as out, open(self._partial, "r", encoding="ascii") as frames:
            def w(text: str) -> None:
                out.write(text)
                h.update(text.encode("ascii"))
            # canonical key order: final_tracks < frame_log < summary
            w('{"final_tracks":' + _dumps(final_tracks) + ',"frame_log":[')
            for block in iter(lambda: frames.read(self._chunk), ""):
                w(block)
            w('],"summary":' + _dumps(summary) + "}\n")
        if os.path.exists(self.out_path):
            os.remove(tmp)
            raise RunRefusedError(f"refusing to overwrite existing output {self.out_path}")
        os.rename(tmp, self.out_path)
        os.remove(self._partial)
        return h.hexdigest()

    def abort(self) -> None:
        try:
            self._fh.close()
        finally:
            for p in (self._partial, self.out_path + PARTIAL_SUFFIX + ".doc"):
                if os.path.exists(p):
                    os.remove(p)


class _DocumentChecks:
    """The document-level rules of output_schema.validate_document(require_v2=True), applied incrementally so the
    whole frame_log never has to be held in memory. Per-frame records are validated with output_schema itself."""

    def __init__(self, cfg: TrackerConfig, start_frame: int):
        self._cfg = cfg
        self._start = start_frame
        self.next_frame = 1
        self.episodes_first: Optional[int] = None
        self.last_episode: Optional[int] = None
        self.increments = 0

    def frame(self, fr: Dict[str, Any]) -> None:
        where = f"frame_log[{fr.get('frame')}]"
        errs = S.validate_frame_record(fr, where, require_v2=True, cfg=self._cfg)
        if fr.get("frame") != self.next_frame:
            errs.append(f"{where}: frames must run 1, 2, ... (expected {self.next_frame})")
        if fr[S.V2_KEY]["video_frame_index"] != self._start + fr["frame"]:
            errs.append(f"{where}: v2.video_frame_index must equal start_frame + frame")
        sel = [t for t in fr["tracks"] if t.get("selected_player")]
        if len(sel) == 1:
            ep = sel[0][S.V2_KEY]["target_episode"]
            if self.last_episode is None:
                if ep != 1:
                    errs.append("target_episode must start at 1")
            elif ep not in (self.last_episode, self.last_episode + 1):
                errs.append("target_episode may only stay the same or increase by 1")
            elif ep == self.last_episode + 1:
                self.increments += 1
            self.last_episode = ep
        if errs:
            raise ContractViolationError("; ".join(errs))
        self.next_frame += 1

    def finish(self, summary: Dict[str, Any], final_tracks: List[Dict[str, Any]]) -> None:
        errs = S.validate_summary(summary, require_v2=True)
        for i, t in enumerate(final_tracks):
            errs.extend(S.validate_track_record(t, f"final_tracks[{i}]", cfg=self._cfg, context="final_tracks"))
        if sum(1 for t in final_tracks if t.get("selected_player") is True) > 1:
            errs.append("final_tracks: more than one selected_player record")
        if summary["processed_frames"] != self.next_frame - 1:
            errs.append("summary.processed_frames != number of frame records")
        v2 = summary[S.V2_KEY]
        if len(v2["reacquisition_events"]) != self.increments:
            errs.append(f"{self.increments} episode increments but {len(v2['reacquisition_events'])} events")
        prov = v2["provenance"]
        if float(prov["fps"]) != float(summary["fps"]):
            errs.append("summary.v2.provenance.fps must equal summary.fps")
        if prov["seed_frame"] != summary["start_frame"]:
            errs.append("summary.v2.provenance.seed_frame must equal summary.start_frame")
        if errs:
            raise ContractViolationError("; ".join(errs))


# ---------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------
def output_paths(out_dir: str, video_path: str) -> Dict[str, str]:
    base = os.path.splitext(os.path.basename(video_path))[0]
    return {"results": os.path.join(out_dir, base + RESULTS_SUFFIX),
            "provenance": PV.sidecar_path(out_dir, video_path),
            "decision_log": os.path.join(out_dir, base + DECISION_LOG_SUFFIX)}


def run(video_path: str, start_frame: int, init_box: Sequence[float], jersey_number: Optional[str] = None,
        max_frames: Optional[int] = None, out_dir: Optional[str] = None, debug: bool = False,
        config_overrides: Optional[Mapping[str, Any]] = None, *, detector: Optional[DetectorBackend] = None,
        embedder: Optional[EmbeddingBackend] = None, expected_sha256: Optional[str] = None,
        git: Callable[[], Tuple[str, bool]] = PV.git_state, clock: Callable[[], str] = PV.utc_now_iso,
        timer: Callable[[], float] = time.perf_counter,
        on_frame: Optional[Callable[[int, Dict[str, int]], None]] = None) -> Dict[str, Any]:
    """Run the R1 Pass-1 tracker on one clip and write its outputs into out_dir. Returns the summary dict.

    The historical run_mo signature minus `resume` (no resume exists, F15). `jersey_number` must be None: the
    generic tracker reads no numbers. `max_frames` caps the frames processed after the seed (None: to the end).
    `detector`/`embedder` may be passed in only if they are the backends the config names (build_backends).
    `expected_sha256` pins the source video (fixture hash); a mismatch refuses the run. `on_frame(frame, sizes)`
    is a read-only diagnostics hook receiving the sizes of all state kept between frames (long-run checks)."""
    if jersey_number is not None:
        raise RunRefusedError("jersey_number is not supported: the R1 tracker is generic (no number reading)")
    if out_dir is None or not os.path.isdir(out_dir):
        raise RunRefusedError(f"out_dir must be an existing directory, got {out_dir!r}")
    overrides = dict(config_overrides or {})
    if max_frames is not None:
        overrides["max_frames"] = max_frames
    cfg = TrackerConfig.with_overrides(overrides) if overrides else TrackerConfig.default()
    paths = output_paths(out_dir, video_path)
    for p in paths.values():
        if os.path.exists(p):
            raise RunRefusedError(f"refusing to overwrite existing output {p}")
    det, emb = build_backends(cfg, detector, embedder)
    models = [m for m in (det.model_record(), emb.model_record()) if m]

    t0 = timer()
    doc = None
    log_fh = None
    ok = False
    try:
        with SequentialVideoReader(video_path, start_frame, cfg, expected_sha256=expected_sha256) as reader:
            info = reader.info
            try:
                contract_prov = PV.contract_provenance(reader, init_box, cfg, models=models, git=git, clock=clock)
            except PV.ProvenanceError as e:
                raise RunRefusedError(f"provenance: {e}") from e
            fps = info.fps
            est = CameraMotionEstimator(cfg)
            pop = P.Population(cfg, info.frame_width, info.frame_height)
            tracker = T.TargetTracker(cfg, pop, AppearanceExtractor(cfg, emb))
            seed = reader.seed
            cands = det.detect(seed.image)
            cam = est.update(seed.image, [d.bbox for d in cands], 1.0 / fps)
            res = pop.update(0, 0.0, cands, cam)
            try:
                seed_frame = tracker.seed(res, seed.image, init_box)
            except T.SeedError as e:
                raise RunRefusedError(f"seed refused: {e}") from e
            builder = _RecordBuilder(cfg, seed_frame.bbox)
            checks = _DocumentChecks(cfg, start_frame)
            doc = _StreamingDocument(paths["results"], cfg.get("hash_chunk_bytes"))
            if debug:
                log_fh = open(paths["decision_log"], "x", encoding="ascii")
                log_fh.write(_dumps({"frame": 0, "target": seed_frame.as_record(), "population": res.as_record()})
                             + "\n")
            thumbs: List[str] = []
            last_frame_ids: Set[str] = set()
            hist_states: List[str] = []
            rb_states: List[str] = []
            for fr in reader:
                f = fr.rel_frame
                cands = det.detect(fr.image)
                cam = est.update(fr.image, [d.bbox for d in cands], 1.0 / fps)
                res = pop.update(f, f / fps, cands, cam)
                tf = tracker.update(res, fr.image)
                excluded = set(tracker.target_tracklets)
                if tf.observation_tracklet_id is not None:
                    excluded.add(tf.observation_tracklet_id)
                trec = builder.target_record(f, tf)
                precs = builder.population_records(f, res, excluded)
                written = {builder._tid} | {r["track_id"] for r in precs}
                builder.record_conflicts(tf, res, excluded, written)
                h = tf.target_height_px
                shift = (cam.shift_px[0] + 0.0, cam.shift_px[1] + 0.0)          # no negative zero in the JSON
                frec = S.build_frame_record(
                    cfg, frame=f, is_cut=cam.status == CUT, camera_shift_px=shift, n_detections=len(cands),
                    overlap_contests=builder.overlap_contests(res, excluded, written), tracks=[trec] + precs,
                    v2={"video_frame_index": fr.video_frame_index, "timestamp_s": fr.video_time_s,
                        "frame_thumb_sha1": fr.thumb_sha1, "cut_detected": cam.status == CUT,
                        "camera_shift_h": [shift[0] / h, shift[1] / h] if h else None, "target_height_px": h})
                checks.frame(frec)
                doc.add_frame(frec)
                last_frame_ids = written
                if log_fh is not None:
                    log_fh.write(_dumps({"frame": f, "target": tf.as_record(), "camera": cam.as_record(),
                                         "population": res.as_record()}) + "\n")
                thumbs.append(fr.thumb_sha1)
                hist_states.append(trec["state"])
                rb_states.append(tf.rb_state)
                if on_frame is not None:
                    sizes = dict(pop.state_sizes(), **tracker.state_sizes(), **builder.state_sizes())
                    on_frame(f, sizes)
        # the reader is closed here and the source re-hashed (raises SourceChangedError on any change)
        summary, finals = _summary(cfg, info, start_frame, contract_prov, builder, tracker, hist_states, rb_states,
                                   timer() - t0, last_frame_ids)
        checks.finish(summary, finals)
        sidecar = PV.video_provenance_sidecar(reader, contract_prov, cfg, frame_thumbs=thumbs)
        sidecar["backends"] = {"detector": det.provenance(), "embedding": emb.provenance()}
        doc_sha = doc.finish(summary, finals)
        doc = None
        sidecar["results_file"] = os.path.basename(paths["results"])
        sidecar["results_sha256"] = doc_sha
        if log_fh is not None:
            log_fh.close()
            log_fh = None
        PV.write_sidecar(sidecar, paths["provenance"])
        ok = True
        return summary
    finally:
        if doc is not None:
            doc.abort()
        if log_fh is not None:
            log_fh.close()
        if not ok:                                           # fail closed: no partial output survives a failure
            for k in ("results", "decision_log", "provenance"):
                if k != "provenance" or not os.path.exists(paths["results"]):
                    if os.path.exists(paths[k]):
                        os.remove(paths[k])


SUMMARY_NOTES: Tuple[str, ...] = (
    "R1 (player_tracker_rb) output; historical keys keep their historical meaning; NEW information is under v2.",
    "Target T001 is one logical target for the whole run (D07); target_episode and reacquisition_events record "
    "every loss and reacquisition. Only rb_state TRACKING (historical ACTIVE/CONFIRMED_DETECTION) is a trusted, "
    "observed identity.",
    "Target bbox = last confirmed (observed) box; observed_position_box = this frame's observed, predicted or held "
    "box (D06). v2.observation_source / bbox_is_observed / bbox_is_held / bbox_is_predicted say which.",
    "identity_confidence: v2.identity_confidence_rb when trusted, else <= untrusted_identity_confidence_ceiling "
    "(D19). overall_track_confidence = identity_confidence. A reacquisition accepted without any banked competitor "
    "to contrast (reacquisition_events[].evidence.contrast_available = false) keeps its whole episode at the "
    "minimal trusted confidence 0.5 until a frame passes a real contrast. identity_confidence_rb is UNCALIBRATED.",
    "NEW semantics (not the historical formulas, which are UNKNOWN): appearance_confidence and reid_confidence = "
    "composite appearance similarity of the reported box to the target bank (0 if none); motion_confidence = 1 "
    "while positional continuity is valid; visibility/position/track_continuity_confidence = 1 only when observed. "
    "Population records carry no identity claim (identity_confidence 0; overall_track_confidence = detector score "
    "when detected). team = UNKNOWN and team_confidence = 0 everywhere: the generic tracker has no team logic.",
    "Physical-entity exclusion: a population tracklet that is or may be the target this frame is not written as a "
    "separate person, and is written as T001 in overlap_contests.",
    "overlap_contests = every overlapping pair of bodies this frame. identity_conflict_counts['T001|Txxx'] = frames in "
    "which Txxx was the competing hypothesis that kept the target from being confirmed; identity_conflicts_logged = "
    "number of such frames.",
    "spawn_guard_stats is empty and health_metrics provisional_to_confirmed_ratio, duplicate_likelihood, "
    "live_association_eligible_dormant_count and pruned_from_live_association_count are null: R1 has no such "
    "mechanisms. total_tracks_spawned counts population tracklets (the target's own included).",
    "Wall-clock fields (excluded from the determinism comparison): summary.processing_time_s, "
    "summary.v2.provenance.run_started_utc.",
)


def _summary(cfg: TrackerConfig, info: Any, start_frame: int, prov: Dict[str, Any], b: _RecordBuilder,
             tracker: T.TargetTracker, hist_states: List[str], rb_states: List[str], elapsed_s: float,
             last_frame_ids: Set[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n = len(hist_states)
    tid = cfg.get("target_track_id")
    finals = []
    active = confirmed = provisional = 0
    for k in [tid] + sorted(k for k in b.final_records if k != tid):
        rec = {kk: vv for kk, vv in b.final_records[k].items() if kk != S.V2_KEY}
        if k in last_frame_ids:
            active += 1
            if k in b.provisional_ids:
                provisional += 1
            else:
                confirmed += 1
        else:
            rec["state"] = "TERMINATED"
        finals.append(rec)
    spawned = len(b.population_ids_seen)
    lifetimes = [last - first + 1 for first, last in b.lifetimes.values()]
    summary = {
        "video": info.path, "start_frame": start_frame, "fps": info.fps, "processed_frames": n,
        "processing_time_s": round(float(elapsed_s), 3),
        "selected_state_pct": S.selected_state_pct(hist_states, cfg.get("state_pct_round_dp")),
        "n_tracks_ever_created": 1 + len(b.population_ids_written),
        "n_tracks_final_active": active, "clip_output": None,
        "final_confirmed_count": confirmed, "final_provisional_count": provisional,
        "spawn_guard_stats": {},
        "identity_conflicts_logged": b.conflict_frames,
        "identity_conflict_counts": dict(sorted(b.conflicts.items())),
        "health_metrics": {
            "frames_processed": n, "total_tracks_spawned": spawned,
            "new_tracks_per_frame": round(spawned / n, 4) if n else 0.0,
            "tracks_per_100_frames": round(100.0 * spawned / n, 2) if n else 0.0,
            "provisional_to_confirmed_ratio": None,
            "median_track_lifetime_frames": statistics.median(lifetimes) if lifetimes else None,
            "duplicate_likelihood": None, "live_association_eligible_dormant_count": None,
            "pruned_from_live_association_count": None},
        S.V2_KEY: {
            "schema": S.SCHEMA_NAME, "schema_version": S.V2_SCHEMA_VERSION, "tracker": S.TRACKER_NAME,
            "tracker_version": TRACKER_VERSION, "provenance": prov, "config_hash": cfg.config_hash(),
            "reacquisition_events": [e.as_record() for e in tracker.reacquisition_events],
            "rb_state_pct": S.selected_state_pct(rb_states, cfg.get("state_pct_round_dp")),
            "notes": list(SUMMARY_NOTES)},
    }
    return summary, finals


# ---------------------------------------------------------------------------------------------
# Determinism comparison
# ---------------------------------------------------------------------------------------------
def _without(doc: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    node = doc
    for k in path[:-1]:
        node = node.get(k, {})
    return node.pop(path[-1], None)


def compare_documents(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two run documents. Returns {"identical_except_wall_clock": bool, "wall_clock": {...},
    "first_difference": path or None}. Every field other than WALL_CLOCK_FIELDS must be identical."""
    import copy
    a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
    wall = {}
    for path in WALL_CLOCK_FIELDS:
        wall[".".join(path)] = (_without(a2, path), _without(b2, path))
    first = _first_difference(a2, b2, "")
    return {"identical_except_wall_clock": first is None, "wall_clock": wall, "first_difference": first}


def _first_difference(a: Any, b: Any, path: str) -> Optional[str]:
    if type(a) is not type(b):
        return path or "/"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}"
            d = _first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}[len]"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else (path or "/")
