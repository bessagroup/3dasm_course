import json, csv, math
R = json.load(open('recomputed.json'))
S = "/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/Lecture20/car_team_run1/workspace/results/"
COLS = ["model","a","b","sd_parameter","coverage","nll","n_train","n_test"]
lines = ["# Check of M1 and M2 (independent recomputation)", "",
         "Tolerance: relative difference < 1e-6 (absolute for zero-valued items). Recomputed from data/car_stopping_distance with train_test_split(row index, train_size=0.75, random_state=1987); script kept in the delegation folder D003.", ""]
allpass = True
for m in ["M1","M2"]:
    with open(S+f"{m}.csv") as f:
        rd = csv.reader(f); hdr = next(rd); rows = list(rd)
    stored = dict(zip(hdr, rows[0])) if rows else {}
    lines.append(f"## {m}")
    ok = (hdr == COLS and len(rows) == 1)
    allpass &= ok
    lines.append(f"- [{'PASS' if ok else 'FAIL'}] column schema and one row: header={hdr}, data rows={len(rows)}")
    r = R[m]
    sdname = "s" if m=="M1" else "c"
    items = [("n_train", r["n_train"], "n_train"), ("n_test", r["n_test"], "n_test"),
             ("a", r["a"], "a"), ("b", r["b"], "b"), ("sd_parameter", r["sd"], sdname),
             ("coverage", r["coverage"], "coverage"), ("nll", r["nll"], "nll")]
    for key, mine, label in items:
        sv = float(stored[key]) if key in stored else float("nan")
        rel = abs(sv-mine)/max(abs(mine),1e-300)
        ok = rel < 1e-6
        allpass &= ok
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {label}: stored (CSV) = {sv!r}, recomputed = {mine!r}, rel diff = {rel:.2e}")
    lines.append("")
lines.insert(3, f"**Overall verdict: {'PASS' if allpass else 'FAIL'}**"); lines.insert(4,"")
open(S+"check.md","w").write("\n".join(lines)+"\n")
print("\n".join(lines))
