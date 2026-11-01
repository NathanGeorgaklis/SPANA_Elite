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
  * LINKED to the best viable candidate P only if it beats every other non-vetoed candidate by ambiguity_margin
    (contrastive, not raw similarity) AND no other tracklet born within identity_link_window_s is also viable for P
    within ambiguity_margin of B (exclusivity both ways). Otherwise UNKNOWN; NEW only if no candidate is viable and
    every non-vetoed candidate is below appearance_floor - ambiguity_margin.

APPEARANCE HISTORY. A person's history is the clean samples of its own tracklets' population banks (population
sampling rules: uncontested, unmerged boxes), except the TARGET person: its history is the target tracker's own bank
only (samples taken on trusted frames). Population samples of tracklets the target was on are never added to the
target person's history, so an untrusted target frame never updates the trusted history.

TARGET PERSON (read-only of the tracker). The person of the tracklet the target is TRUSTED on is the target person
(is_target). A person observed as a separate body (a different tracklet, outside the target's nested group) in the
same frame while the target is trusted is NOT_TARGET for good: two simultaneous bodies are two people. This is the
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
    numbers: Dict[str, int] = field(default_factory=dict)


@dataclass
class _Pending:
    tid: str
    birth_t: float
    birth_box: Box
    position_valid: bool
    candidates: Tuple[str, ...]
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
        # closures: a closed tracklet's person is no longer alive; keep its clean samples
        for cl in res.closed:
            pid = self._person_of.get(cl.tracklet_id)
            if pid is not None:
                p = self._persons[pid]
                if p.alive == cl.tracklet_id:
                    p.alive = None
                    p.last_t, p.last_box = cl.closed_time_s, cl.last_box
                    p.position_valid = bool(cl.position_valid) and not cl.merge_lineage
                    if not p.is_target:
                        p.history = self._keep(p.history + list(cl.bank.samples))
            self._pending.pop(cl.tracklet_id, None)        # closed before a decision: nothing to link
        # live tracklets: births and bookkeeping
        live = {v.tracklet_id: v for v in res.tracklets}
        for tid, v in live.items():
            if tid in self._person_of or tid in self._pending:
                continue
            cands = tuple(pid for pid, p in self._persons.items()
                          if p.alive is None and sec_le(t - p.last_t, c.get("identity_link_window_s")))
            self._pending[tid] = _Pending(tid, t, v.box, not cam.continuity_broken, cands)
            self._status[tid] = PENDING
            self._births.append((tid, t))
        self._births = [(b, bt) for (b, bt) in self._births if sec_le(t - bt, c.get("identity_link_window_s"))]
        for tid, v in live.items():
            pid = self._person_of.get(tid)
            if pid is not None and v.observed:
                p = self._persons[pid]
                p.last_t, p.last_box, p.position_valid = t, v.box, True
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
            if not [pid for pid in pd.candidates if self._persons[pid].alive is None]:
                decisions.append(self._open(pd, NEW, (), {"reason": "NO_CANDIDATE_PERSON", "scores": []}))
                continue
            ready = sec_ge(age, c.get("identity_evidence_s")) and len(samples) >= c.get("identity_min_samples")
            if ready:
                decisions.append(self._decide(pd, samples, target_bank, live))
            elif sec_gt(age, c.get("identity_decide_max_s")):
                decisions.append(self._open(pd, UNKNOWN, pd.candidates,
                                            {"reason": "NOT_ENOUGH_EVIDENCE", "scores": []}))
        # target person and not-target persons (read-only of the tracker)
        target_pid = None
        if target_trusted and target_tracklet is not None:
            target_pid = self._person_of.get(target_tracklet)
            if target_pid is not None:
                self._persons[target_pid].is_target = True
                tv = live.get(target_tracklet)
                grp = self._nested_group(res, tv.detection_index) if tv is not None and tv.observed else set()
                for tid, v in live.items():
                    if tid == target_tracklet or not v.observed or v.detection_index in grp:
                        continue
                    pid = self._person_of.get(tid)
                    if pid is not None and pid != target_pid:
                        self._persons[pid].not_target = True
        # bounded memory (long matches): a person gone longer than the link window can never be a candidate again
        gone = [pid for pid, p in self._persons.items() if p.alive is None and not p.is_target
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
            out.append(TrackletIdentity(tid, pid, self._status.get(tid, PENDING), bool(p and p.is_target),
                                        bool(p and p.not_target), self._possible.get(tid, ())))
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

    def _score(self, pd: _Pending, samples: List[AppearanceFeature], pid: str,
               target_bank: Optional[AppearanceBank]) -> Dict[str, Any]:
        c = self._cfg
        p = self._persons[pid]
        hist = self._history(p, target_bank)
        vetoes: List[str] = []
        if pd.position_valid and p.position_valid:
            (bx, by), (px, py) = _centre(pd.birth_box), _centre(p.last_box)
            h = max(pd.birth_box[3] - pd.birth_box[1], 1.0)
            d_h = math.hypot(bx - px, by - py) / h
            lim = c.get("identity_gate_base_h") + c.get("identity_max_speed_h_per_s") * max(0.0, pd.birth_t - p.last_t)
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
        floor = c.get("appearance_floor") if (pd.position_valid and p.position_valid) else \
            c.get("reacq_appearance_floor")
        return {"person": pid, "similarity": s_med, "kit": kit, "distance_h": d_h, "vetoes": vetoes,
                "floor": floor, "history_samples": len(hist), "is_target": p.is_target,
                "viable": not vetoes and s_med is not None and s_med >= floor}

    def _decide(self, pd: _Pending, samples: List[AppearanceFeature], target_bank: Optional[AppearanceBank],
                live: Dict[str, P.TrackletView]) -> Dict[str, Any]:
        c = self._cfg
        m = c.get("ambiguity_margin")
        cands = [pid for pid in pd.candidates if self._persons[pid].alive is None]
        scores = [self._score(pd, samples, pid, target_bank) for pid in cands]
        open_ = [s for s in scores if not s["vetoes"] and s["similarity"] is not None]
        viable = sorted((s for s in open_ if s["viable"]), key=lambda s: -s["similarity"])
        log = {"scores": scores}
        uncomparable = [s["person"] for s in scores if not s["vetoes"] and s["similarity"] is None]
        if uncomparable:                                  # a candidate without history cannot be ruled out
            return self._open(pd, UNKNOWN, tuple(uncomparable) + tuple(s["person"] for s in open_),
                              dict(log, reason="CANDIDATE_WITHOUT_HISTORY"))
        if not viable:
            if all(s["similarity"] < s["floor"] - m for s in open_):
                return self._open(pd, NEW, (), dict(log, reason="NO_CANDIDATE_CAN_BE_IT"))
            return self._open(pd, UNKNOWN, tuple(s["person"] for s in open_), dict(log, reason="BELOW_FLOOR"))
        best = viable[0]
        rivals = [s for s in open_ if s is not best and s["similarity"] >= best["similarity"] - m]
        if rivals:
            return self._open(pd, UNKNOWN, tuple(s["person"] for s in [best] + rivals),
                              dict(log, reason="AMBIGUOUS_CANDIDATES"))
        # exclusivity the other way: another recent newborn that is as good a match for the same person
        for other, _bt in self._births:
            if other == pd.tid or other not in live:
                continue
            if self._person_of.get(other) == best["person"]:
                return self._open(pd, UNKNOWN, (best["person"],), dict(log, reason="PERSON_ALREADY_CLAIMED"))
            if other in self._pending:
                osamp = self._pending[other].feats
                if osamp:
                    os_ = self._score(self._pending[other], osamp, best["person"], target_bank)
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
        self._persons[pid] = _Person(pid, [pd.tid], pd.birth_t, pd.birth_box, pd.position_valid, alive=pd.tid,
                                     numbers=dict(pd.numbers))
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
