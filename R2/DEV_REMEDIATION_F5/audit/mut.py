import subprocess, sys, json, pathlib
ROOT = pathlib.Path("/home/claude/mut_f5")
F = ROOT / "player_tracker_rb/target.py"
ORIG = F.read_text()
PY = "/home/claude/venv_r2dev/bin/python"
M = {
 "M01_drop_unknown_local": ('''            if cp.bank is None or not len(cp.bank):
                return "UNKNOWN_LOCAL_BODY"''', '''            if cp.bank is None or not len(cp.bank):
                continue'''),
 "M02_drop_confusable_local": ('''                return "CONFUSABLE_LOCAL_BODY"''', '''                continue'''),
 "M03_local_ge_to_gt": ('''cs.composite is None or cs.composite >= floor:''', '''cs.composite is None or cs.composite > floor:'''),
 "M04_local_none_not_confusable": ('''if cs is None or cs.composite is None or cs.composite >= floor:''', '''if cs is not None and cs.composite is not None and cs.composite >= floor:'''),
 "M05_drop_global_lookalike": ('''                return "TARGET_LOOKALIKE_IN_FRAME"''', '''                pass'''),
 "M06_global_ge_to_gt": ('''ts.composite is not None and ts.composite >= floor:''', '''ts.composite is not None and ts.composite > floor:'''),
 "M07_global_use_reacq_floor": ('''ts.composite is not None and ts.composite >= floor:''', '''ts.composite is not None and ts.composite >= c.get("reacq_appearance_floor"):'''),
 "M08_skip_all_recently_closed": ('''            if cp.kind == P.RECENTLY_CLOSED and cp.closed_age_s is not None and \\
                    cp.closed_age_s > c.get("contact_memory_s"):''', '''            if cp.kind == P.RECENTLY_CLOSED:'''),
 "M09_skip_hidden_competitors": ('''            if cp.kind == P.RECENTLY_CLOSED and cp.closed_age_s is not None and \\
                    cp.closed_age_s > c.get("contact_memory_s"):''', '''            if cp.kind != P.OBSERVATION:'''),
 "M10_feature_unusable_none": ('''            return "FEATURE_NOT_USABLE"''', '''            return None'''),
 "M11_drop_passed": ('''        if not chosen.passed:
            why = "EVALUATION_FAILED"
        elif bad_flags:''', '''        if bad_flags:'''),
 "M12_drop_bad_flags": ('''        elif bad_flags:
            why = "BOX_NOT_CLEAN:" + "+".join(sorted(bad_flags))
''', ''),
 "M13_drop_contains_other": ('''_BANK_REFUSE_FLAGS + (P.CONTAINS_OTHER,)''', '''_BANK_REFUSE_FLAGS'''),
 "M14_drop_merge_lineage": ('''        elif v is not None and P.MERGE_LINEAGE in v.flags:
            why = "MERGE_LINEAGE"
''', ''),
 "M15_appearance_floor_not_reacq": ('''chosen.composite < c.get("reacq_appearance_floor")''', '''chosen.composite < c.get("appearance_floor")'''),
 "M16_reacq_floor_lt_to_le": ('''chosen.composite < c.get("reacq_appearance_floor")''', '''chosen.composite <= c.get("reacq_appearance_floor")'''),
 "M17_drop_competitor_check": ('''            why = self._plausible_competitor(res, image, cache, chosen, h)''', '''            why = None'''),
 "M18_fail_no_restart": ('''        if why is not None:
            rv["t0"], rv["box0"] = None, None
            return {"status": "RESTART", "why": why}''', '''        if why is not None:
            return {"status": "RESTART", "why": why}'''),
 "M19_not_tracking_no_restart": ('''            rv["t0"], rv["box0"] = None, None
            return {"status": "RESTART", "why": "NOT_TRACKING_ON_ORIGIN"}''', '''            return {"status": "RESTART", "why": "NOT_TRACKING_ON_ORIGIN"}'''),
 "M20_left_tracklet_not_ended": ('''            if self._tracklet != rv["origin"]:''', '''            if False:'''),
 "M21_any_frame_accept": ('''if elapsed >= c.get("target_reverify_window_s") and travel_h''', '''if travel_h'''),
 "M22_skip_travel": ('''and travel_h >= c.get("target_reverify_min_travel_h"):''', ''':'''),
 "M23_elapsed_ge_to_gt": ('''elapsed >= c.get("target_reverify_window_s")''', '''elapsed > c.get("target_reverify_window_s")'''),
 "M24_travel_ge_to_gt": ('''travel_h >= c.get("target_reverify_min_travel_h")''', '''travel_h > c.get("target_reverify_min_travel_h")'''),
 "M25_skip_vacated": ('''            for spot in (rv["spot"], rv["box0"]):''', '''            for spot in ():'''),
 "M26_only_anomaly_spot": ('''            for spot in (rv["spot"], rv["box0"]):''', '''            for spot in (rv["spot"],):'''),
 "M27_only_box0": ('''            for spot in (rv["spot"], rv["box0"]):''', '''            for spot in (rv["box0"],):'''),
 "M28_unusable_spot_clean": ('''                    rec["vacated_spot"] = "UNCHECKABLE"
                    return rec''', '''                    continue'''),
 "M29_vacated_lookalike_ignored": ('''                    rv["t0"], rv["box0"] = None, None
                    return {"status": "RESTART", "why": "TARGET_LOOKALIKE_AT_VACATED_SPOT", "similarity": vs.composite}''', '''                    pass'''),
 "M30_vacated_reacq_floor": ('''vs.composite >= c.get("appearance_floor"):''', '''vs.composite >= c.get("reacq_appearance_floor"):'''),
 "M31_vacated_no_others": ('''others=[ob.bbox for ob in res.observations])''', '''others=[])'''),
 "M32_origin_even_unverified": ('''self._anomaly_origin = self._tracklet if self._contrast_verified else None''', '''self._anomaly_origin = self._tracklet'''),
 "M33_no_clear_on_cut": ('''            self._anomaly_origin, self._anomaly_spot, self._reverify = None, None, None   # no continuity (F5)
''', ''),
 "M34_no_warp_reverify": ('''                if self._reverify is not None:
                    for key in ("box0", "spot"):''', '''                if False:
                    for key in ("box0", "spot"):'''),
 "M35_no_warp_anomaly_spot": ('''                if self._anomaly_spot is not None:
                    self._anomaly_spot = cam.warp_box(self._anomaly_spot)''', '''                pass'''),
 "M36_open_any_tracklet": ('''and ev.tracklet_id == self._anomaly_origin) else None)''', '''and self._anomaly_origin is not None) else None)'''),
 "M37_open_any_reacq": ('''                                  if (not self._contrast_verified and ev.tracklet_id is not None
                                      and ev.tracklet_id == self._anomaly_origin) else None)''', '''                                  if (not self._contrast_verified and ev.tracklet_id is not None) else None)'''),
 "M38_no_clear_origin_after_acc": ('''                self._anomaly_origin, self._anomaly_spot = None, None
            elif self._verif is not None:''', '''            elif self._verif is not None:'''),
 "M39_no_conf_recompute": ('''            if self._contrast_verified:
                conf_trusted = self._confidence_trusted(chosen)''', '''            pass'''),
 "M40_drop_origin_tracklet_check": ('''if state != TRACKING or chosen is None or chosen.tracklet_id != rv["origin"]:''', '''if state != TRACKING or chosen is None:'''),
 "M41_own_only_ev": ('''        own = set(self._target_tracklets) | ({ev.tracklet_id} if ev.tracklet_id else set())''', '''        own = set(self._target_tracklets) | ({ev.tracklet_id} if ev.tracklet_id else set())
        own = own | {o.tracklet_id for o in res.observations if o.tracklet_id}'''),
 "M42_drop_size_filter_inverse": ('''            if cp.position_valid and not self._could_be_target(cp.box, h, t):''', '''            if not self._could_be_target(cp.box, h, t):'''),
 "M43_spot_none_from_anomaly": ('''"spot": self._anomaly_spot}''', '''"spot": None}'''),
 "M44_global_skip_tracklets": ('''            if not o.is_body or o.index in grp or (o.tracklet_id is not None and o.tracklet_id in own):''', '''            if not o.is_body or o.index in grp or o.tracklet_id is not None:'''),
 "M45_already_verified_noop": ('''        if self._contrast_verified:                          # verified meanwhile (e.g. by ordinary contrast)
            self._reverify = None
            return {"status": "ALREADY_VERIFIED"}''', '''        pass'''),
 "M46_global_unusable_withholds_not": ('''            f = self._feature(cache, res, image, o.index)
            if not f.usable:
                continue''', '''            f = self._feature(cache, res, image, o.index)
            if not f.usable:
                return None'''),
 "M47_travel_from_spot": ('''        cx0, cy0 = centre(rv["box0"])''', '''        cx0, cy0 = centre(rv["spot"] or rv["box0"])'''),
}
names = sys.argv[1:] or list(M)
tests = ["tests_rb/test_reacquisition_f5.py", "tests_rb/test_merge_guard.py"]
if "--target" in names:
    names.remove("--target"); tests = ["tests_rb/test_target.py"]
for n in names:
    a, b = M[n]
    cnt = ORIG.count(a)
    if cnt != 1:
        print(n, "BADMATCH", cnt, flush=True); continue
    F.write_text(ORIG.replace(a, b))
    try:
        p = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"] + tests, cwd=ROOT, capture_output=True, text=True, timeout=1200)
        last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-300:]
        fail = [l for l in p.stdout.splitlines() if l.startswith("FAILED")][:1]
        print(n, "KILLED" if p.returncode else "SURVIVED", "|", last, "|", fail, flush=True)
    finally:
        F.write_text(ORIG)
