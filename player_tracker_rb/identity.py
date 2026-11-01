# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV persistent football identity layer instruction (Nathan, 2026-11-01 21:39): persistent PERSON_IDs
#                 across tracklet breaks from the evidence the tracker already has (colour + DINOv2 appearance history
#                 over trusted / clean observations, trajectory, temporal team evidence, optional temporal jersey
#                 numbers); ReID / team / number are supporting evidence, never proof alone; aggregate over time;
#                 UNKNOWN remains valid; do not weaken identity_trusted; untrusted target frames never update the
#                 trusted appearance history; do not replace the tracker.
"""Persistent football identity layer (NEW_R2, DIAGNOSTIC ONLY).

A read-only layer over the population (tracklets, closures, per-tracklet clean appearance banks) and the target
tracker's output. It assigns persistent PERSON_IDs across tracklet breaks. It NEVER changes the target tracker:
not its state, trust, bank, known others or candidates. It is not called by run_rb; tools and tests run it beside
the tracker.

PERSONS. Every tracklet gets a link status:
  * PENDING   just born, evidence being aggregated (person_id None);
  * LINKED    continues an earlier person (person_id = that person's id);
  * NEW       no earlier person can be it (every candidate is physically impossible, vetoed, or clearly
              different in appearance): a new person id;
  * UNKNOWN   evidence does not decide: a new person id is opened, and the earlier persons it might be are recorded in
              `possible_persons`. UNKNOWN is a valid, final answer: it is never forced into a link later.

LINK DECISION for a newborn tracklet B (decided once, when B has >= identity_min_samples clean observed frames -- the
population's bank-eligible boxes: uncontested, unmerged -- and is >= identity_evidence_s old; UNKNOWN when still
undecided at identity_decide_max_s; NEW at once when no person is a candidate at all). B's evidence is its own
appearance on every one of those frames (extracted with the production extractor), aggregated over time:
  * CANDIDATES: persons with no live tracklet (a person cannot be two bodies at once) last seen at most
    identity_link_window_s before B's birth.
  * PHYSICAL GATE (veto only): when both positions are valid, the candidate's last box centre (camera-compensated,
    invalidated by a cut) must be within identity_gate_base_h + identity_max_speed_h_per_s * gap of B's first box, in
    B's box heights.
  * KIT (temporal team evidence, veto only): the median over B's samples of the best LAB similarity to the
    candidate's history must reach identity_kit_veto_lab. It never makes a link.
  * NUMBER (optional, veto only): a tracklet's number is CONFIRMED after identity_number_min_reads agreeing readings
    and no conflicting one (duplicate / unreadable / conflicting readings confirm nothing). Confirmed different numbers
    refuse a link; equal numbers never make one. No reader is configured in R2 (reader None).
  * APPEARANCE (temporal aggregation): S = median over B's samples of the best composite similarity to the
    candidate's history. A candidate is VIABLE if not vetoed and S >= appearance_floor (reacq_appearance_floor when
    either position is invalid: appearance only, so the stricter floor).
  * LINKED to the best viable candidate P only if it beats every other candidate not refused by kit / number
    (including candidates refused only by the physical gate, which can be wrong) by ambiguity_margin (contrastive,
    not raw similarity) AND no other tracklet born within identity_link_window_s is LINKED to P, has P among its
    possible persons, or is also viable for P within ambiguity_margin (exclusivity both ways). Otherwise UNKNOWN.
    NEW only with positive evidence against every candidate (kit / number veto, or appearance below
    appearance_floor - ambiguity_margin); a physical veto alone is not enough.
  * Candidates are recomputed at decision time; a person whose own tracklet is still live but unseen (hidden / held)
    keeps the newborn undecided (UNKNOWN at identity_decide_max_s).
  * OPEN WORLD: no appearance-only link is ever made (positions must be valid; after a cut: UNKNOWN), and a link INTO
    the target person needs reacq_appearance_floor. A never-seen look-alike whose appearance passes that floor where
    the target was last seen cannot be told from the target (residual; R2 ReID evaluation).

APPEARANCE HISTORY. A person's history is the clean samples of its own tracklets' population banks (population
sampling rules: uncontested, unmerged boxes), except the TARGET person: its history is the target tracker's own bank
only (samples taken on trusted frames). Population samples of tracklets the target was on are never added to the
target person's history, so an untrusted target frame never updates the trusted history.

TARGET PERSON (read-only of the tracker). Once the target has been TRUSTED on the same tracklet for
target_known_other_min_s (one trusted frame can be a tracker error), that tracklet's person is the target person
(is_target), and every person observed as a separate body (a different tracklet, outside the target's nested group)
in that frame is NOT_TARGET: two simultaneous bodies are two people. A person flagged both ways, or two target
persons seen at once, is a CONFLICT: its tracklets report neither flag (no narrowing; fail closed). This is the
person-level form of the tracker's tracklet-level known others; it survives tracklet breaks only through links, so
it is as safe as the links.

OUTPUT per frame: IdentityFrame with one TrackletIdentity per live tracklet and the link decisions made this frame
(candidates, scores, vetoes and the reason, for audit).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from . import population as P
from .appearance import AppearanceBank, AppearanceExtractor, AppearanceFeature, compare, lab_similarity
from .config import TrackerConfig, sec_ge, sec_gt, sec_le

PENDING = "PENDING"
LINKED = "LINKED"
NEW = "NEW"
UNKNOWN = "UNKNOWN"

VETO_PHYSICAL = "PHYSICALLY_UNREACHABLE"
VETO_KIT = "KIT_DIFFERENT"
VETO_NUMBER = "NUMBER_DIFFERENT"

Box = Tuple[float, float, float, float]
# a jersey-number reader: (image, box) -> (number string, confidence) or None (unreadable). Optional; none in R2.
NumberReader = Callable[[Any, Box], Optional[Tuple[str, float]]]


def _centre(b: Sequence[float]) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


@dataclass
class _Person:
    pid: str
    tracklets: List[str]
    last_t: float
    last_box: Box
    position_valid: bool
    history: List[AppearanceFeature] = field(default_factory=list)
    alive: Optional[str] = None                 # its live tracklet, if any
    is_target: bool = False
    not_target: bool = False
    conflict: bool = False                      # flagged both ways (or two target persons at once): fail closed
    numbers: Dict[str, int] = field(default_factory=dict)


@dataclass
class _Pending:
    tid: str
    birth_t: float
    position_valid: bool
    numbers: Dict[str, int] = field(default_factory=dict)
    feats: List[AppearanceFeature] = field(default_factory=list)   # its own clean observed frames


@dataclass(frozen=True)
class TrackletIdentity:
    tracklet_id: str
    person_id: Optional[str]
    status: str
    is_target_person: bool
    not_target_person: bool
    possible_persons: Tuple[str, ...] = ()

    def as_record(self) -> Dict[str, Any]:
        return {"tracklet_id": self.tracklet_id, "person_id": self.person_id, "status": self.status,
                "is_target_person": self.is_target_person, "not_target_person": self.not_target_person,
                "possible_persons": list(self.possible_persons)}


@dataclass(frozen=True)
class IdentityFrame:
    rel_frame: int
    tracklets: Tuple[TrackletIdentity, ...]
    decisions: Tuple[Dict[str, Any], ...]
    target_person: Optional[str]

    def of(self, tracklet_id: str) -> Optional[TrackletIdentity]:
        for x in self.tracklets:
            if x.tracklet_id == tracklet_id:
                return x
        return None

    def as_record(self) -> Dict[str, Any]:
        return {"rel_frame": self.rel_frame, "tracklets": [x.as_record() for x in self.tracklets],
                "decisions": list(self.decisions), "target_person": self.target_person}


class IdentityError(RuntimeError):
    pass


class PersonIdentityLayer:
    def __init__(self, cfg: TrackerConfig, population: P.Population, extractor: AppearanceExtractor,
                 number_reader: Optional[NumberReader] = None):
        self._cfg = cfg
        self._pop = population
        self._ext = extractor
        self._reader = number_reader
        self._persons: Dict[str, _Person] = {}
        self._person_of: Dict[str, str] = {}          # decided tracklet -> person id
        self._status: Dict[str, str] = {}             # tracklet -> link status
        self._possible: Dict[str, Tuple[str, ...]] = {}
        self._pending: Dict[str, _Pending] = {}
        self._births: List[Tuple[str, float]] = []    # (tracklet, birth time) for the exclusivity check
        self._n = 0
        self._last_frame: Optional[int] = None
        self._trust: Optional[Tuple[str, float]] = None   # (tracklet, since) the target has been trusted on

    # -- helpers --------------------------------------------------------------------------------------------
    def _new_pid(self) -> str:
        self._n += 1
        return f"P{self._n:04d}"

    def _samples_of(self, tid: str) -> List[AppearanceFeature]:
        bank = self._pop.bank_of(tid)
        return list(bank.samples) if bank is not None else []

    def _history(self, p: _Person, target_bank: Optional[AppearanceBank]) -> List[AppearanceFeature]:
        if p.is_target:
            return list(target_bank.samples) if target_bank is not None else []   # trusted samples only
        live = self._samples_of(p.alive) if p.alive else []
        return p.history + live

    def _keep(self, xs: List[AppearanceFeature]) -> List[AppearanceFeature]:
        cap = self._cfg.get("identity_history_max_samples")
        if len(xs) <= cap:
            return xs
        return xs[:1] + xs[len(xs) - (cap - 1):]          # the earliest view, then the most recent ones

    def _confirmed_number(self, votes: Dict[str, int]) -> Optional[str]:
        if len(votes) != 1:
            return None                                   # none, or conflicting / duplicate readings
        (num, n), = votes.items()
        return num if n >= self._cfg.get("identity_number_min_reads") else None

    # -- per frame ------------------------------------------------------------------------------------------
    def update(self, res: P.FrameResult, image: Any, *, target_tracklet: Optional[str], target_trusted: bool,
               target_bank: Optional[AppearanceBank]) -> IdentityFrame:
        """After population.update / sample_appearance and the target tracker for the same frame. The target inputs
        are read only (the tracklet the target is on, whether it is trusted, its bank)."""
        if self._last_frame is not None and res.rel_frame <= self._last_frame:
            raise IdentityError("frames must be given in increasing order")
        self._last_frame = res.rel_frame
        c = self._cfg
        t = res.time_s
        cam = res.camera
        decisions: List[Dict[str, Any]] = []
        # camera: remembered positions follow the measured camera; a continuity break invalidates them
        for p in self._persons.values():
            if p.alive is not None:
                continue
            if cam.continuity_broken:
                p.position_valid = False
            elif p.position_valid and cam.compensated:
                p.last_box = cam.warp_box(p.last_box)
        for pd in self._pending.values():
            if cam.continuity_broken:
                pd.position_valid = False
        # closures: a closed tracklet's person is no longer alive; keep its clean samples. Its last time stays its
        # last OBSERVATION (review M6: the time it has really been unseen) and its box the last known one.
        for cl in res.closed:
            pid = self._person_of.get(cl.tracklet_id)
            if pid is not None:
                p = self._persons[pid]
                if p.alive == cl.tracklet_id:
                    p.alive = None
                    if cl.position_valid and not cl.merge_lineage:
                        p.last_box = cl.last_box
                    else:
                        p.position_valid = False
                    if not p.is_target:
                        p.history = self._keep(p.history + list(cl.bank.samples))
            self._pending.pop(cl.tracklet_id, None)        # closed before a decision: nothing to link
        # live tracklets: births and bookkeeping
        live = {v.tracklet_id: v for v in res.tracklets}
        for tid, v in live.items():
            if tid in self._person_of or tid in self._pending:
                continue
            self._pending[tid] = _Pending(tid, t, not cam.continuity_broken)
            self._status[tid] = PENDING
            self._births.append((tid, t))
        self._births = [(b, bt) for (b, bt) in self._births if sec_le(t - bt, c.get("identity_link_window_s"))]
        for tid, v in live.items():
            pid = self._person_of.get(tid)
            if pid is None:
                continue
            p = self._persons[pid]
            if v.observed:
                p.last_t, p.last_box, p.position_valid = t, v.box, True
            elif p.position_valid:
                p.last_box = v.box                         # predicted / held (camera-compensated by the population)
        # newborn evidence: the tracklet's own clean observed frames (population bank-eligible boxes)
        bodies = [o for o in res.observations if o.is_body]
        for tid, pd in self._pending.items():
            v = live.get(tid)
            if v is None or not v.observed or not v.bank_eligible:
                continue
            i = v.detection_index
            feat = self._ext.extract(image, res.observations[i].bbox,
                                     others=[o.bbox for o in bodies if o.index != i])
            if feat.usable:
                pd.feats.append(feat)
        # optional jersey numbers (supporting evidence)
        if self._reader is not None:
            for tid, v in live.items():
                if not v.observed:
                    continue
                r = self._reader(image, v.box)
                if r is None:
                    continue
                votes = self._pending[tid].numbers if tid in self._pending else \
                    self._persons[self._person_of[tid]].numbers if tid in self._person_of else None
                if votes is not None:
                    votes[r[0]] = votes.get(r[0], 0) + 1
        # decisions
        for tid in list(self._pending):
            pd = self._pending[tid]
            age = t - pd.birth_t
            samples = pd.feats
            cands, held = self._candidates(pd, live)
            if not cands and not held:
                decisions.append(self._open(pd, NEW, (), {"reason": "NO_CANDIDATE_PERSON", "scores": []}))
                continue
            ready = sec_ge(age, c.get("identity_evidence_s")) and len(samples) >= c.get("identity_min_samples")
            if ready and held:
                # review HIGH-2: a person whose own tracklet is still live but unseen (hidden / held) could be this
                # newborn: undecidable while that tracklet lives
                if sec_gt(age, c.get("identity_decide_max_s")):
                    decisions.append(self._open(pd, UNKNOWN, tuple(held) + tuple(cands),
                                                {"reason": "HELD_PERSON_POSSIBLE", "scores": []}))
                continue
            if ready:
                decisions.append(self._decide(pd, samples, cands, target_bank, live))
            elif sec_gt(age, c.get("identity_decide_max_s")):
                decisions.append(self._open(pd, UNKNOWN, tuple(cands) + tuple(held),
                                            {"reason": "NOT_ENOUGH_EVIDENCE", "scores": []}))
        # target person and not-target persons (read-only of the tracker)
        # review M5: only once the target has been trusted on the same tracklet for target_known_other_min_s (one
        # trusted frame can be a tracker error); a person flagged both ways, or two target persons seen at once, is a
        # CONFLICT and narrows nothing (fail closed)
        target_pid = None
        if target_trusted and target_tracklet is not None:
            if self._trust is None or self._trust[0] != target_tracklet:
                self._trust = (target_tracklet, t)
        else:
            self._trust = None
        settled = self._trust is not None and sec_ge(t - self._trust[1], c.get("target_known_other_min_s"))
        if settled:
            target_pid = self._person_of.get(target_tracklet)
            if target_pid is not None:
                tp = self._persons[target_pid]
                tp.is_target = True
                tv = live.get(target_tracklet)
                grp = self._nested_group(res, tv.detection_index) if tv is not None and tv.observed else set()
                for tid, v in live.items():
                    if tid == target_tracklet or not v.observed or v.detection_index in grp:
                        continue
                    pid = self._person_of.get(tid)
                    if pid is None or pid == target_pid:
                        continue
                    op = self._persons[pid]
                    if op.is_target:                       # two target persons at once: one of the links is wrong
                        op.conflict = tp.conflict = True
                    op.not_target = True
                for p in (tp,):
                    if p.not_target:
                        p.conflict = True
        # bounded memory (long matches): a person gone longer than the link window can never be a candidate again
        # (a person still listed as a candidate of an undecided newborn is kept until that decision: the link window
        # is measured from the newborn's birth)
        held = {pid for pd in self._pending.values() for pid, p in self._persons.items()
                if sec_le(pd.birth_t - p.last_t, c.get("identity_link_window_s"))}
        gone = [pid for pid, p in self._persons.items() if p.alive is None and not p.is_target and pid not in held
                and sec_gt(t - p.last_t, c.get("identity_link_window_s"))]
        for pid in gone:
            for tid in self._persons.pop(pid).tracklets:
                self._person_of.pop(tid, None)
                self._status.pop(tid, None)
                self._possible.pop(tid, None)
        out = []
        for tid in live:
            pid = self._person_of.get(tid)
            p = self._persons.get(pid) if pid is not None else None
            ok = bool(p and not p.conflict)
            out.append(TrackletIdentity(tid, pid, self._status.get(tid, PENDING), ok and p.is_target,
                                        ok and p.not_target, self._possible.get(tid, ())))
        return IdentityFrame(res.rel_frame, tuple(out), tuple(decisions), target_pid)

    @staticmethod
    def _nested_group(res: P.FrameResult, i: Optional[int]) -> Set[int]:
        if i is None:
            return set()
        obs = res.observations
        root = i
        while obs[root].nested_in is not None:
            root = obs[root].nested_in
        grp, changed = {root}, True
        while changed:
            changed = False
            for o in obs:
                if o.nested_in is not None and o.nested_in in grp and o.index not in grp:
                    grp.add(o.index)
                    changed = True
        return grp

    def _candidates(self, pd: _Pending, live: Dict[str, P.TrackletView]) -> Tuple[List[str], List[str]]:
        """Persons this newborn could be, recomputed at decision time (review HIGH-2): (unseen persons last observed
        within identity_link_window_s of its birth, persons whose own tracklet is live but not observed now)."""
        w = self._cfg.get("identity_link_window_s")
        cands, held = [], []
        for pid, p in self._persons.items():
            if not sec_le(pd.birth_t - p.last_t, w):
                continue
            if p.alive is None:
                cands.append(pid)
            elif p.alive != pd.tid and p.alive in live and not live[p.alive].observed:
                held.append(pid)
        return cands, held

    def _score(self, pd: _Pending, box: Box, t: float, samples: List[AppearanceFeature], pid: str,
               target_bank: Optional[AppearanceBank]) -> Dict[str, Any]:
        c = self._cfg
        p = self._persons[pid]
        hist = self._history(p, target_bank)
        vetoes: List[str] = []
        positions = pd.position_valid and p.position_valid
        if positions:
            # review M6: both boxes in the CURRENT frame's coordinates (the person's last box follows the camera;
            # the newborn's box is its current one) and the time it has really been unseen
            (bx, by), (px, py) = _centre(box), _centre(p.last_box)
            h = max(box[3] - box[1], 1.0)
            d_h = math.hypot(bx - px, by - py) / h
            lim = c.get("identity_gate_base_h") + c.get("identity_max_speed_h_per_s") * max(0.0, t - p.last_t)
            if d_h > lim:
                vetoes.append(VETO_PHYSICAL)
        else:
            d_h = None
        comps, labs = [], []
        for s in samples:
            best_c, best_l = None, None
            for hsmp in hist:
                sim = compare(s, hsmp, c)
                if sim.composite is not None and (best_c is None or sim.composite > best_c):
                    best_c = sim.composite
                ls = lab_similarity(s.lab, hsmp.lab)
                if ls is not None and (best_l is None or ls > best_l):
                    best_l = ls
            if best_c is not None:
                comps.append(best_c)
            if best_l is not None:
                labs.append(best_l)
        s_med = statistics.median(comps) if comps else None
        kit = statistics.median(labs) if labs else None
        if kit is not None and kit < c.get("identity_kit_veto_lab"):
            vetoes.append(VETO_KIT)
        na, nb = self._confirmed_number(pd.numbers), self._confirmed_number(p.numbers)
        if na is not None and nb is not None and na != nb:
            vetoes.append(VETO_NUMBER)
        # review HIGH-3 (open world): an appearance-only link (no valid positions) is never made, and a link INTO the
        # target person needs the stricter reacquisition floor
        floor = c.get("reacq_appearance_floor") if p.is_target else c.get("appearance_floor")
        return {"person": pid, "similarity": s_med, "kit": kit, "distance_h": d_h, "vetoes": vetoes,
                "floor": floor, "history_samples": len(hist), "is_target": p.is_target, "positions": positions,
                "viable": positions and not vetoes and s_med is not None and s_med >= floor}

    def _decide(self, pd: _Pending, samples: List[AppearanceFeature], cands: List[str],
                target_bank: Optional[AppearanceBank], live: Dict[str, P.TrackletView]) -> Dict[str, Any]:
        c = self._cfg
        m = c.get("ambiguity_margin")
        t = self._pop.last.time_s if self._pop.last is not None else pd.birth_t
        box = live[pd.tid].box
        scores = [self._score(pd, box, t, samples, pid, target_bank) for pid in cands]
        log: Dict[str, Any] = {"scores": scores}
        uncomparable = [s["person"] for s in scores if not s["vetoes"] and s["similarity"] is None]
        if uncomparable:                                  # a candidate without history cannot be ruled out
            return self._open(pd, UNKNOWN, tuple(s["person"] for s in scores),
                              dict(log, reason="CANDIDATE_WITHOUT_HISTORY"))
        # NEW needs positive evidence against every candidate (review M6): a kit / number veto, or appearance clearly
        # below the floor; a physical veto alone (a gate can be wrong) is not enough
        def excluded(s):
            return VETO_KIT in s["vetoes"] or VETO_NUMBER in s["vetoes"] or \
                (s["similarity"] is not None and s["similarity"] < s["floor"] - m)
        open_ = [s for s in scores if not s["vetoes"] and s["similarity"] is not None]
        viable = sorted((s for s in open_ if s["viable"]), key=lambda s: -s["similarity"])
        if not viable:
            if all(excluded(s) for s in scores):
                return self._open(pd, NEW, (), dict(log, reason="NO_CANDIDATE_CAN_BE_IT"))
            reason = "NO_VALID_POSITION" if any(not s["positions"] for s in scores) else "BELOW_FLOOR"
            return self._open(pd, UNKNOWN, tuple(s["person"] for s in scores if not excluded(s)),
                              dict(log, reason=reason))
        best = viable[0]
        # a candidate refused ONLY by the physical gate still competes on appearance (the gate can be wrong: a
        # detector or camera error): if it looks as much like the newborn, the link is not unique
        rivals = [s for s in scores if s is not best and s["similarity"] is not None
                  and VETO_KIT not in s["vetoes"] and VETO_NUMBER not in s["vetoes"]
                  and s["similarity"] >= best["similarity"] - m]
        if rivals:
            return self._open(pd, UNKNOWN, tuple(s["person"] for s in [best] + rivals),
                              dict(log, reason="AMBIGUOUS_CANDIDATES"))
        # exclusivity both ways (review HIGH-1): any other recent newborn that is, might be, or is as good a match for
        # the same person blocks the link, including one already decided UNKNOWN with this person as possible
        for other, _bt in self._births:
            if other == pd.tid or other not in live:
                continue
            if self._person_of.get(other) == best["person"]:
                return self._open(pd, UNKNOWN, (best["person"],), dict(log, reason="PERSON_ALREADY_CLAIMED"))
            if best["person"] in self._possible.get(other, ()):
                return self._open(pd, UNKNOWN, (best["person"],),
                                  dict(log, reason="PERSON_POSSIBLY_CLAIMED", other=other))
            if other in self._pending:
                osamp = self._pending[other].feats
                if osamp:
                    os_ = self._score(self._pending[other], live[other].box, t, osamp, best["person"], target_bank)
                    if os_["viable"] and os_["similarity"] >= best["similarity"] - m:
                        return self._open(pd, UNKNOWN, (best["person"],),
                                          dict(log, reason="ANOTHER_NEWBORN_MATCHES", other=other))
        pid = best["person"]
        p = self._persons[pid]
        p.tracklets.append(pd.tid)
        p.alive = pd.tid
        for k, n in pd.numbers.items():
            p.numbers[k] = p.numbers.get(k, 0) + n
        self._person_of[pd.tid] = pid
        self._status[pd.tid] = LINKED
        del self._pending[pd.tid]
        return dict(log, tracklet=pd.tid, status=LINKED, person=pid, reason="UNIQUE_CONTRASTIVE_MATCH")

    def _open(self, pd: _Pending, status: str, possible: Tuple[str, ...], log: Dict[str, Any]) -> Dict[str, Any]:
        pid = self._new_pid()
        last = self._pop.last
        v = last.tracklet(pd.tid) if last is not None else None
        box = v.box if v is not None else (0.0, 0.0, 1.0, 1.0)
        self._persons[pid] = _Person(pid, [pd.tid], last.time_s if last is not None else pd.birth_t, box,
                                     pd.position_valid, alive=pd.tid, numbers=dict(pd.numbers))
        self._person_of[pd.tid] = pid
        self._status[pd.tid] = status
        self._possible[pd.tid] = tuple(possible)
        del self._pending[pd.tid]
        return dict(log, tracklet=pd.tid, status=status, person=pid, possible=list(possible))

    # -- read-only queries ----------------------------------------------------------------------------------
    def person_of(self, tracklet_id: str) -> Optional[str]:
        return self._person_of.get(tracklet_id)

    def is_not_target(self, tracklet_id: str) -> bool:
        pid = self._person_of.get(tracklet_id)
        return bool(pid is not None and self._persons[pid].not_target)

    def is_target_person(self, tracklet_id: str) -> bool:
        pid = self._person_of.get(tracklet_id)
        return bool(pid is not None and self._persons[pid].is_target)
