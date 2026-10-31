import sys, pathlib
P = "player_tracker_rb/population.py"
C = "player_tracker_rb/config.py"
M = {
 "M01_drop_fill_exclusion_in_emerged": (P, "if region_of.get(i) == k or outsider(i, rg) or i in fills[k]:", "if region_of.get(i) == k or outsider(i, rg):"),
 "M02_fillers_always_count": (P, "n_sep = len((singles - fills[k]) | emerged[k])\n            if emerged[k] and singles & fills[k]:\n                n_sep += 1", "n_sep = len(singles | emerged[k])"),
 "M03_drop_emergence_gate": (P, "if emerged[k] and singles & fills[k]:", "if singles & fills[k]:"),
 "M04_no_filled_memory_in_forget": (P, "t - max(rg.last_evidence_t, rg.last_filled_t) <= keep_region_s", "t - rg.last_evidence_t <= keep_region_s"),
 "M05_never_set_last_filled_t": (P, "            if fills[k]:\n                rg.last_filled_t = t", "            pass"),
 "M06_drop_fillers_lineage_keep": (P, "if tr.tid in self._regions[k].born and tr.tid not in self._regions[k].fillers:", "if tr.tid in self._regions[k].born:"),
 "M07_drop_outsider_first_box": (P, "return fb is not None and float(containment(np.asarray([fb]), np.asarray([rg.box]))[0, 0]) < cmin", "return False"),
 "M08_invert_feet_gap": (P, "and hb[3] - boxes[j][3] <= feet_gap * hh)", "and hb[3] - boxes[j][3] > feet_gap * hh)"),
 "M09_loosen_feet_gap_all_nested": (P, "and hb[3] - boxes[j][3] <= feet_gap * hh)", "and hb[3] - boxes[j][3] <= 1.0 * hh)"),
 "M10_loosen_feet_gap_x3": (P, "and hb[3] - boxes[j][3] <= feet_gap * hh)", "and hb[3] - boxes[j][3] <= 3 * feet_gap * hh)"),
 "M11_abs_feet_gap": (P, "and hb[3] - boxes[j][3] <= feet_gap * hh)", "and abs(hb[3] - boxes[j][3]) <= feet_gap * hh)"),
 "M12_vanish_ignore_duplicates": (P, "others = [boxes[j] for j in range(n) if j != i]", "others = [boxes[j] for j in bodies if j != i]"),
 "M13_skip_union_for_vanished": (P, "                breaks[i] = \"CONTAINED_BODY_VANISHED\"\n                uf.union((\"t\", order_of[tid]), (\"d\", i))", "                breaks[i] = \"CONTAINED_BODY_VANISHED\""),
 "M14_vanish_any_not_all": (P, "if not bool(seen.all()):", "if not bool(seen.any()):"),
 "M15_no_camera_warp_contained": (P, "prev = [camera.warp_box(b) if camera.compensated else b for b in tr.contained]", "prev = list(tr.contained)"),
 "M16_vanish_loose_iou": (P, "seen = box_iou_matrix(np.asarray(prev), np.asarray(others)).max(axis=1) >= match_min", "seen = box_iou_matrix(np.asarray(prev), np.asarray(others)).max(axis=1) > 0.0"),
 "M17_no_vanish_check_at_all": (P, "if not bool(seen.all()):", "if False:"),
 "M18_drop_break_label": (P, "continuity_break=breaks.get(i)))", "continuity_break=None))"),
 "M19_clear_contained_when_hidden": (P, "                tr.contained = tuple(boxes[j] for j in bodies if j != i and nested_in[j] == i\n                                     and hb[3] - boxes[j][3] <= feet_gap * hh)", "                tr.contained = tuple(boxes[j] for j in bodies if j != i and nested_in[j] == i\n                                     and hb[3] - boxes[j][3] <= feet_gap * hh)\n            else:\n                tr.contained = ()"),
 "M20_vanish_check_also_merged": (P, "if i in merged or not tr.contained:", "if not tr.contained:"),
 "M21_fill_min_0_5": (C, "       0.9, N, \"DESIGN_CHOICE\", UNV, \"BEHAVIOURAL\", \"population\",\n       \"NEW_R2 (R2 DEV remediation F3c). Two people", "       0.5, N, \"DESIGN_CHOICE\", UNV, \"BEHAVIOURAL\", \"population\",\n       \"NEW_R2 (R2 DEV remediation F3c). Two people"),
 "M22_fill_min_0_99": (C, "       0.9, N, \"DESIGN_CHOICE\", UNV, \"BEHAVIOURAL\", \"population\",\n       \"NEW_R2 (R2 DEV remediation F3c). Two people", "       0.99, N, \"DESIGN_CHOICE\", UNV, \"BEHAVIOURAL\", \"population\",\n       \"NEW_R2 (R2 DEV remediation F3c). Two people"),
 "M23_fill_min_0_75": (C, "       0.9, N, \"DESIGN_CHOICE\", UNV, \"BEHAVIOURAL\", \"population\",\n       \"NEW_R2 (R2 DEV remediation F3c). Two people", "       0.75, N, \"DESIGN_CHOICE\", UNV, \"BEHAVIOURAL\", \"population\",\n       \"NEW_R2 (R2 DEV remediation F3c). Two people"),
 "M24_outsider_ignore_first_time": (P, "            if tr.first_time < rg.created_t:\n                return True\n            fb = tr.first_box", "            fb = tr.first_box"),
 "M25_fillers_not_excluded_from_singles": (P, "n_sep = len((singles - fills[k]) | emerged[k])", "n_sep = len(singles | emerged[k])"),
 "M26_contained_any_nested_host": (P, "if j != i and nested_in[j] == i\n", "if j != i and nested_in[j] is not None\n"),
}
if __name__ == "__main__":
    name = sys.argv[1]
    if name == "list":
        print("\n".join(M)); sys.exit()
    f, old, new = M[name]
    p = pathlib.Path(f); s = p.read_text()
    assert s.count(old) == 1, (name, s.count(old))
    p.write_text(s.replace(old, new)); print("applied", name)
