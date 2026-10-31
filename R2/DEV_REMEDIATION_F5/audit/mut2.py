import subprocess, sys, pathlib
ROOT = pathlib.Path("/home/claude/mut_f5")
PY = "/home/claude/venv_r2dev/bin/python"
TGT = "player_tracker_rb/target.py"; CFGF = "player_tracker_rb/config.py"
M = {
 "N01_drop_unknown_local": (TGT, '''                if cp.bank is None or not len(cp.bank):
                    return "UNKNOWN_LOCAL_BODY"''', '''                if cp.bank is None or not len(cp.bank):
                    continue'''),
 "N02_drop_confusable_local": (TGT, '''                    return "CONFUSABLE_LOCAL_BODY"''', '''                    continue'''),
 "N03_local_ge_to_gt": (TGT, '''cs.composite is None or cs.composite >= floor:''', '''cs.composite is None or cs.composite > floor:'''),
 "N04_local_none_not_confusable": (TGT, '''if cs is None or cs.composite is None or cs.composite >= floor:''', '''if cs is not None and cs.composite is not None and cs.composite >= floor:'''),
 "N05_drop_global_lookalike": (TGT, '''                return "TARGET_LOOKALIKE_IN_FRAME"''', '''                pass'''),
 "N06_global_ge_to_gt": (TGT, '''ts.composite is not None and ts.composite >= floor:''', '''ts.composite is not None and ts.composite > floor:'''),
 "N07_global_use_reacq_floor": (TGT, '''ts.composite is not None and ts.composite >= floor:''', '''ts.composite is not None and ts.composite >= c.get("reacq_appearance_floor"):'''),
 "N08_closed_age_ignored_all_closed": (TGT, '''                if cp.kind == P.RECENTLY_CLOSED and cp.closed_age_s is not None and \\
                        cp.closed_age_s > c.get("contact_memory_s"):''', '''                if cp.kind == P.RECENTLY_CLOSED:'''),
 "N09_only_observed_competitors": (TGT, '''                if cp.kind == P.RECENTLY_CLOSED and cp.closed_age_s is not None and \\
                        cp.closed_age_s > c.get("contact_memory_s"):''', '''                if cp.kind != P.OBSERVATION:'''),
 "N10_feature_unusable_none": (TGT, '''                return "FEATURE_NOT_USABLE"''', '''                return None'''),
 "N10b_no_restart_on_unusable": (TGT, '''        elif why == "FEATURE_NOT_USABLE":
            restart = why
''', ''),
 "N11_drop_passed": (TGT, '''        elif not chosen.passed:
            restart = "EVALUATION_FAILED"
''', ''),
 "N12_drop_bad_flags": (TGT, '''        elif bad_flags:
            restart = "BOX_NOT_CLEAN:" + "+".join(sorted(bad_flags))
''', ''),
 "N13_drop_contains_other": (TGT, '''_BANK_REFUSE_FLAGS + (P.CONTAINS_OTHER,)''', '''_BANK_REFUSE_FLAGS'''),
 "N14_drop_merge_lineage": (TGT, '''        elif P.MERGE_LINEAGE in ov.flags:
            restart = "MERGE_LINEAGE"
''', ''),
 "N15_appearance_floor_not_reacq": (TGT, '''chosen.composite < c.get("reacq_appearance_floor")''', '''chosen.composite < c.get("appearance_floor")'''),
 "N16_reacq_floor_lt_to_le": (TGT, '''chosen.composite < c.get("reacq_appearance_floor")''', '''chosen.composite <= c.get("reacq_appearance_floor")'''),
 "N18_restart_no_reset": (TGT, '''        if restart is not None:
            rv["t0"], rv["box0"] = None, None''', '''        if restart is not None:
            pass'''),
 "N19_not_tracking_no_reset": (TGT, '''            rv["t0"], rv["box0"] = None, None
            rec.update({"status": "RESTART", "why": "NOT_TRACKING_ON_ORIGIN"})''', '''            rec.update({"status": "RESTART", "why": "NOT_TRACKING_ON_ORIGIN"})'''),
 "N20_left_tracklet_not_cancel": (TGT, '''            if self._tracklet != rv["origin"]:
                return cancel("LEFT_ORIGIN_TRACKLET")''', '''            if False:
                return cancel("LEFT_ORIGIN_TRACKLET")'''),
 "N21_any_frame_accept": (TGT, '''if elapsed >= c.get("target_reverify_window_s") and travel_h''', '''if travel_h'''),
 "N22_skip_travel": (TGT, '''and travel_h >= c.get("target_reverify_min_travel_h"):''', ''':'''),
 "N23_elapsed_ge_to_gt": (TGT, '''elapsed >= c.get("target_reverify_window_s")''', '''elapsed > c.get("target_reverify_window_s")'''),
 "N24_travel_ge_to_gt": (TGT, '''travel_h >= c.get("target_reverify_min_travel_h")''', '''travel_h > c.get("target_reverify_min_travel_h")'''),
 "N25_final_spots_ignored": (TGT, '''            if any(spots[k] not in ("CLEAR", "NONE") for k in spots) or spots.get("box0") != "CLEAR":''', '''            if False:'''),
 "N26_final_only_spot": (TGT, '''            if any(spots[k] not in ("CLEAR", "NONE") for k in spots) or spots.get("box0") != "CLEAR":''', '''            if spots.get("spot") not in ("CLEAR", "NONE"):'''),
 "N27_final_only_box0": (TGT, '''            if any(spots[k] not in ("CLEAR", "NONE") for k in spots) or spots.get("box0") != "CLEAR":''', '''            if spots.get("box0") != "CLEAR":'''),
 "N27b_box0_none_ok": (TGT, '''            if any(spots[k] not in ("CLEAR", "NONE") for k in spots) or spots.get("box0") != "CLEAR":''', '''            if any(spots[k] not in ("CLEAR", "NONE") for k in spots):'''),
 "N28_uncheckable_is_clear": (TGT, '''        if not vac.usable:
            return "UNCHECKABLE", None''', '''        if not vac.usable:
            return "CLEAR", None'''),
 "N28b_covered_is_clear": (TGT, '''        if origin_box is not None and _cover(spot, origin_box) > 0.0:''', '''        if False:'''),
 "N28c_covered_half": (TGT, '''        if origin_box is not None and _cover(spot, origin_box) > 0.0:''', '''        if origin_box is not None and _cover(spot, origin_box) > 0.5:'''),
 "N29_spot_lookalike_ignored": (TGT, '''            if st == "LOOKALIKE":
                rec["similarity"] = sim
                return cancel("TARGET_LOOKALIKE_AT_VACATED_SPOT")''', '''            if st == "LOOKALIKE":
                spots[key] = st = "CLEAR"'''),
 "N30_vacated_reacq_floor": (TGT, '''        if sim is not None and sim >= c.get("appearance_floor"):''', '''        if sim is not None and sim >= c.get("reacq_appearance_floor"):'''),
 "N31_vacated_no_others": (TGT, '''        vac = self._ext.extract(image, spot, others=[ob.bbox for ob in res.observations])''', '''        vac = self._ext.extract(image, spot, others=[])'''),
 "N32_watch_even_unverified": (TGT, '''if self._contrast_verified and c.get("target_reverify_enabled"):''', '''if c.get("target_reverify_enabled"):'''),
 "N33_no_clear_on_cut": (TGT, '''            self._reverify = None                             # physical continuity cannot span a cut (F5)
''', ''),
 "N34_no_warp": (TGT, '''                if self._reverify is not None:
                    for key in ("box0", "spot"):''', '''                if False:
                    for key in ("box0", "spot"):'''),
 "N36_run_on_any_tracklet": (TGT, '''                            ev.tracklet_id is not None and ev.tracklet_id == self._reverify["origin"]:''', '''                            ev.tracklet_id is not None:'''),
 "N36b_elsewhere_keeps_watch": (TGT, '''                    else:
                        self._reverify = None                # returned elsewhere: the rule does not apply''', '''                    else:
                        pass'''),
 "N37_run_even_if_contrast_verified": (TGT, '''if self._reverify["phase"] == "WAIT" and not self._contrast_verified and \\''', '''if self._reverify["phase"] == "WAIT" and \\'''),
 "N39_no_conf_recompute": (TGT, '''            if log["reverify"].get("status") == "REVERIFIED":
                conf_trusted = self._confidence_trusted(chosen)''', '''            pass'''),
 "N40_drop_origin_tracklet_check": (TGT, '''if state != TRACKING or chosen is None or chosen.tracklet_id != rv["origin"]:''', '''if state != TRACKING or chosen is None:'''),
 "N42_size_filter_always": (TGT, '''                if cp.position_valid and not self._could_be_target(cp.box, h, t):''', '''                if not self._could_be_target(cp.box, h, t):'''),
 "N45_already_verified_noop": (TGT, '''        if self._contrast_verified and rv["phase"] == "RUN":
            self._reverify = None                            # verified meanwhile (e.g. by ordinary contrast)''', '''        if False:
            self._reverify = None                            # verified meanwhile (e.g. by ordinary contrast)'''),
 "N46_global_unusable_none": (TGT, '''            f = self._feature(cache, res, image, o.index)
            if not f.usable:
                continue''', '''            f = self._feature(cache, res, image, o.index)
            if not f.usable:
                return None'''),
 "N47_travel_from_spot": (TGT, '''        cx0, cy0 = centre(rv["box0"])''', '''        cx0, cy0 = centre(rv["spot"] or rv["box0"])'''),
 "N50_default_on": (CFGF, '''"confusable competitor (R1-G).", False, N,''', '''"confusable competitor (R1-G).", True, N,'''),
 "N51_ignore_flag": (TGT, '''if self._contrast_verified and c.get("target_reverify_enabled"):''', '''if self._contrast_verified:'''),
 "N52_no_expiry": (TGT, '''        if t - rv["t_end"] > c.get("target_reverify_max_s"):''', '''        if False:'''),
 "N53_expiry_x10": (TGT, '''        if t - rv["t_end"] > c.get("target_reverify_max_s"):''', '''        if t - rv["t_end"] > 10 * c.get("target_reverify_max_s"):'''),
 "N53b_expiry_from_t0": (TGT, '''        if t - rv["t_end"] > c.get("target_reverify_max_s"):''', '''        if rv["t0"] is not None and t - rv["t0"] > c.get("target_reverify_max_s"):'''),
 "N54_lost_not_cancel": (TGT, '''        if state == LOST:
            return cancel("LOST")''', '''        if state == LOST and False:
            return cancel("LOST")'''),
 "N56_competitor_restart_not_cancel": (TGT, '''        if why is not None and why != "FEATURE_NOT_USABLE":
            return cancel(why)''', '''        if why is not None and why != "FEATURE_NOT_USABLE":
            rv["t0"], rv["box0"] = None, None
            rec.update({"status": "RESTART", "why": why})
            return rec'''),
 "N57_spot_lookalike_restart_not_cancel": (TGT, '''                rec["similarity"] = sim
                return cancel("TARGET_LOOKALIKE_AT_VACATED_SPOT")''', '''                rec["similarity"] = sim
                rv["t0"], rv["box0"] = None, None
                rec.update({"status": "RESTART", "why": "TARGET_LOOKALIKE_AT_VACATED_SPOT"})
                return rec'''),
 "N58_competitor_only_in_run": (TGT, '''        if why is not None and why != "FEATURE_NOT_USABLE":
            return cancel(why)''', '''        if why is not None and why != "FEATURE_NOT_USABLE" and rv["phase"] == "RUN":
            return cancel(why)'''),
 "N59_no_camera_check": (TGT, '''        if res.camera.status != ESTIMATED:
            restart = "CAMERA_NOT_MEASURED"                  # travel and spots need a measured camera
        elif not chosen.passed:''', '''        if not chosen.passed:'''),
 "N61_no_since_reset": (TGT, '''            self._tracklet_since_t = t       # known others only after target_known_other_min_s of verified tracking
''', ''),
 "N62_already_verified_any_phase": (TGT, '''        if self._contrast_verified and rv["phase"] == "RUN":''', '''        if self._contrast_verified:'''),
 "N63_no_spots_in_wait": (TGT, '''        obox = res.observations[oi].bbox if oi is not None else None
        spots = {}''', '''        obox = res.observations[oi].bbox if oi is not None else None
        if rv["phase"] == "WAIT":
            rec["status"] = "WAITING"
            return rec
        spots = {}'''),
 "N64_no_competitor_when_hidden_local_only": (TGT, '''        own = set(self._target_tracklets) | {origin}
        if i is not None:''', '''        own = set(self._target_tracklets) | {origin}
        if i is None:
            return None
        if i is not None:'''),
}
names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(M)
tests = ["tests_rb/test_reacquisition_f5.py", "tests_rb/test_merge_guard.py"]
if "--target" in sys.argv: tests = ["tests_rb/test_target.py"]
for n in names:
    f, a, b = M[n]
    F = ROOT / f; ORIG = F.read_text()
    cnt = ORIG.count(a)
    if cnt != 1:
        print(n, "BADMATCH", cnt, flush=True); continue
    F.write_text(ORIG.replace(a, b))
    try:
        p = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"] + tests, cwd=ROOT, capture_output=True, text=True, timeout=1500)
        lines = p.stdout.strip().splitlines()
        last = lines[-1] if lines else p.stderr[-300:]
        fail = [l for l in lines if l.startswith("FAILED")][:2]
        print(n, "KILLED" if p.returncode else "SURVIVED", "|", last, "|", fail, flush=True)
    finally:
        F.write_text(ORIG)
