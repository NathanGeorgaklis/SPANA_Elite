import sys, pathlib
P = "player_tracker_rb/population.py"; C = "player_tracker_rb/config.py"
HM = ('       0.2, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population",\n       "NEW_R2 (R2 DEV remediation F3c, review round 1 H3')
FM = ('       0.9, N, "DESIGN_CHOICE", UNV, "BEHAVIOURAL", "population",\n       "NEW_R2 (R2 DEV remediation F3c). Two people')
M = {
 # surviving / unrun mutants from f65c233, re-expressed
 "M01_drop_fill_exclusion_in_emerged": (P, "if region_of.get(i) == k or outsider(i, rg) or i in fills[k]:", "if region_of.get(i) == k or outsider(i, rg):"),
 "M02_fillers_always_count": (P, "n_sep = len((singles - fills[k]) | emerged[k])\n            if emerged[k] and any(near_single_fill(i, rg.ref_wh) for i in singles & fills[k]):\n                n_sep += 1", "n_sep = len(singles | emerged[k])"),
 "M03_drop_emergence_gate": (P, "if emerged[k] and any(near_single_fill(i, rg.ref_wh) for i in singles & fills[k]):", "if any(near_single_fill(i, rg.ref_wh) for i in singles & fills[k]):"),
 "M06_drop_fillers_lineage_keep": (P, "if tr.tid in self._regions[k].born and tr.tid not in self._regions[k].fillers:", "if tr.tid in self._regions[k].born:"),
 "M07_drop_outsider_first_box": (P, "return fb is not None and float(containment(np.asarray([fb]), np.asarray([rg.box]))[0, 0]) < cmin", "return False"),
 "M10_loosen_feet_gap_x3": (P, "and hb[3] - boxes[j][3] <= feet_gap * hh]", "and hb[3] - boxes[j][3] <= 3 * feet_gap * hh]"),
 "M11_abs_feet_gap": (P, "and hb[3] - boxes[j][3] <= feet_gap * hh]", "and abs(hb[3] - boxes[j][3]) <= feet_gap * hh]"),
 "M14_vanish_any_not_all": (P, "if not bool(seen.all()):", "if not bool(seen.any()):"),
 "M16_vanish_loose_iou": (P, "seen = box_iou_matrix(np.asarray(prev), np.asarray(others)).max(axis=1) >= match_min", "seen = box_iou_matrix(np.asarray(prev), np.asarray(others)).max(axis=1) > 0.0"),
 "M19_clear_contained_when_hidden": (P, "                tr.contained_t = t\n", "                tr.contained_t = t\n            else:\n                tr.contained = ()\n"),
 "M20_vanish_check_also_merged": (P, "if i in merged or not tr.contained:", "if not tr.contained:"),
 "M22_fill_min_0_99": (C, FM, FM.replace("0.9, N", "0.99, N")),
 "M23_fill_min_0_75": (C, FM, FM.replace("0.9, N", "0.75, N")),
 "M24_outsider_ignore_first_time": (P, "            if tr.first_time < rg.created_t:\n                return True\n            fb = tr.first_box", "            fb = tr.first_box"),
 "M26_contained_any_nested_host": (P, "now_c = [boxes[j] for j in bodies if j != i and nested_in[j] == i\n", "now_c = [boxes[j] for j in bodies if j != i and nested_in[j] is not None\n"),
 # new rules in 9869c5c
 "N01_near_single_fill_always_true": (P, "return (w * h) / (ref[0] * ref[1]) <= clear_ratio and (w / h) / (ref[0] / ref[1]) <= clear_ratio", "return True"),
 "N02_near_single_fill_area_only": (P, "return (w * h) / (ref[0] * ref[1]) <= clear_ratio and (w / h) / (ref[0] / ref[1]) <= clear_ratio", "return (w * h) / (ref[0] * ref[1]) <= clear_ratio"),
 "N03_near_single_fill_aspect_only": (P, "return (w * h) / (ref[0] * ref[1]) <= clear_ratio and (w / h) / (ref[0] / ref[1]) <= clear_ratio", "return (w / h) / (ref[0] / ref[1]) <= clear_ratio"),
 "N04_near_single_fill_ref_none_true": (P, "            if ref is None:\n                return False\n            w, h = _wh(boxes[i])\n            return (w * h) / (ref[0] * ref[1]) <= clear_ratio", "            if ref is None:\n                return True\n            w, h = _wh(boxes[i])\n            return (w * h) / (ref[0] * ref[1]) <= clear_ratio"),
 "N05_near_single_fill_loose_merged_area": (P, "return (w * h) / (ref[0] * ref[1]) <= clear_ratio and (w / h) / (ref[0] / ref[1]) <= clear_ratio", "return (w * h) / (ref[0] * ref[1]) < c.get(\"merged_area_ratio\")"),
 "N06_check_no_host_displacement": (P, "            prev = [(b[0] + ddx, b[1] + ddy, b[2] + ddx, b[3] + ddy)\n                    for b, since", "            ddx = ddy = 0.0\n            prev = [(b[0] + ddx, b[1] + ddy, b[2] + ddx, b[3] + ddy)\n                    for b, since"),
 "N07_check_no_vertical_displacement": (P, "            prev = [(b[0] + ddx, b[1] + ddy, b[2] + ddx, b[3] + ddy)\n                    for b, since", "            ddy = 0.0\n            prev = [(b[0] + ddx, b[1] + ddy, b[2] + ddx, b[3] + ddy)\n                    for b, since"),
 "N08_drop_hidden_min_filter": (P, "if tr.contained_t - since >= hidden_min_s]", "if True]"),
 "N09_since_never_accumulates": (P, "since_c.append(tr.contained_since[m] if iou[q, m] >= match_min else t)", "since_c.append(t)"),
 "N10_since_always_inherited": (P, "since_c.append(tr.contained_since[m] if iou[q, m] >= match_min else t)", "since_c.append(tr.contained_since[m])"),
 "N11_since_tracking_no_displacement": (P, "moved = np.asarray([(b[0] + ddx, b[1] + ddy, b[2] + ddx, b[3] + ddy) for b in tr.contained])", "moved = np.asarray(list(tr.contained))"),
 "N12_age_uses_now_not_last_obs": (P, "if tr.contained_t - since >= hidden_min_s]", "if t - since >= hidden_min_s]"),
 "N13_hidden_min_1s": (C, HM, HM.replace("0.2, N", "1.0, N")),
 "N14_hidden_min_0": (C, HM, HM.replace("0.2, N", "0.0, N")),
 "N15_hidden_min_0_1": (C, HM, HM.replace("0.2, N", "0.1, N")),
 "N16_hidden_min_0_4": (C, HM, HM.replace("0.2, N", "0.4, N")),
 "N17_strict_age_gt": (P, "if tr.contained_t - since >= hidden_min_s]", "if tr.contained_t - since > hidden_min_s + 1e-6]"),
}
if __name__ == "__main__":
    name = sys.argv[1]
    if name == "list": print("\n".join(M)); sys.exit()
    f, old, new = M[name]
    p = pathlib.Path(f); s = p.read_text()
    assert s.count(old) == 1, (name, s.count(old))
    p.write_text(s.replace(old, new)); print("applied", name)
