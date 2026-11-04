"""Check your Final Project folder (your_Project/) before you zip it. The TAs run this check first on every submission.

    python check_submission.py            # check this folder, including a rerun of the six notebooks
    python check_submission.py --no-run   # skip the rerun (quick)
    python check_submission.py PATH       # check another folder

It checks the files, the structure of the notebooks, the CSV files, the study folders (every agent on Claude Haiku),
the answer labels and length of the report (if pypdf is installed) and, unless --no-run, that every notebook runs top to
bottom in under 10 minutes with RUN_AGENTS = False and RECOMPUTE = False and writes the same CSV files that you submit.
It does not grade anything: passing it only means that the graders will find what they need.
"""
import argparse
import ast
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = {
    1: ["1.1", "1.2", "1.3", "1.4"],
    2: ["2.1.1", "2.1.2", "2.2"],
    3: ["3.1.1", "3.1.2", "3.1.3", "3.2"],
    4: ["4.1", "4.2", "4.3", "4.4"],
    5: ["5.1", "5.2", "5.3"],
    6: ["6.1", "6.2", "6.3"],
}
STUDIES = ["task4_2", "task4_4", "task5_1", "task6"]
CSVS = ["task2_models.csv", "designs_3d.csv", "submission.csv", "designs_3d_agents.csv", "submission_agents.csv"]
REQUIRED = ["report.pdf", "environment.yml", "data", "studies"] + CSVS + [f"task{n}.ipynb" for n in TASKS]
ALLOWED = set(REQUIRED) | {"cache", "check_submission.py", "DELIVERABLES.md", "CLAUDE.md", "report_template", ".ipynb_checkpoints"}
COLS_3D = ["ratio_d", "ratio_pitch", "ratio_top_diameter"]
COLS_7D = ["ratio_area", "ratio_Ixx", "ratio_Iyy", "ratio_J", "ratio_pitch", "ratio_top_diameter", "ratio_shear_modulus"]
GE_MAX = 0.45
PLACEHOLDER = "_Replace this line with your answer._"
TIME_LIMIT = 600  # seconds per notebook
CACHE_MB = 50
MAX_PAGES = 1 + 15 + 3  # cover, at most 15 pages, and a generous allowance for the references

errors, warnings = [], []
err = errors.append
warn = warnings.append


def check_files(root):
    for name in REQUIRED:
        if not (root / name).exists():
            err(f"missing: {name}")
    extra = sorted(p.name for p in root.iterdir() if p.name not in ALLOWED and not p.name.startswith("."))
    if extra:
        warn(f"not graded (remove them or move them into a notebook): {', '.join(extra)}")
    cache = root / "cache"
    if cache.is_dir():
        mb = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) / 2**20
        if mb > CACHE_MB:
            err(f"cache/ holds {mb:.0f} MB; the limit is {CACHE_MB} MB (cache results, not models or large intermediate arrays)")


def local_modules(root):
    """Names a notebook could import from the submission itself."""
    names = {p.stem for p in root.glob("*.py")} | {p.name for p in root.iterdir() if (p / "__init__.py").exists()}
    return names - {"check_submission"}


def check_notebook(root, n):
    import nbformat
    path = root / f"task{n}.ipynb"
    if not path.exists():
        return
    nb = nbformat.read(path, as_version=4)
    tag = path.name
    code = [c for c in nb.cells if c.cell_type == "code"]
    if not code or not re.search(r"^RUN_AGENTS\s*=\s*False\b", code[0].source, re.M) \
            or not re.search(r"^RECOMPUTE\s*=\s*False\b", code[0].source, re.M):
        err(f"{tag}: the first code cell must set RUN_AGENTS = False and RECOMPUTE = False (keep the skeleton's first cell)")

    # One heading per subtask, in order, each section closed by its answer cell
    heads = [(i, m.group(1)) for i, c in enumerate(nb.cells) if c.cell_type == "markdown"
             for m in [re.match(r"##\s+(\d+(?:\.\d+)+)\b", c.source.strip())] if m]
    found = [h for _, h in heads]
    if found != TASKS[n]:
        err(f"{tag}: section headings are {found or 'missing'}; expected exactly {TASKS[n]} in this order (as in the skeleton)")
    bounds = [i for i, _ in heads] + [len(nb.cells)]
    for k, (i, num) in enumerate(heads):
        answers = [c.source for c in nb.cells[i + 1:bounds[k + 1]] if c.cell_type == "markdown"
                   and c.source.strip().startswith(f"**Answer {num}.**")]
        if not answers:
            err(f"{tag}: section {num} has no cell starting with **Answer {num}.**")
        elif PLACEHOLDER in answers[-1] or len(answers[-1].split()) < 12:
            err(f"{tag}: the answer of {num} is still empty")

    # Executed top to bottom, without errors
    counts = [c.execution_count for c in code if c.source.strip()]
    if any(x is None for x in counts):
        err(f"{tag}: not executed (some code cells have no output); run it top to bottom and save it")
    elif counts != list(range(1, len(counts) + 1)):
        err(f"{tag}: cells were not run once, top to bottom, in a fresh kernel (execution counts {counts[:8]}...)")
    if any(o.get("output_type") == "error" for c in code for o in c.get("outputs", [])):
        err(f"{tag}: a cell ends with an error")

    # Self-contained: no imports of your own files
    local = local_modules(root)
    for c in code:
        src = "\n".join(l for l in c.source.splitlines() if not l.lstrip().startswith(("%", "!")))
        if re.search(r"^\s*%run\b|exec\(\s*open\(|sys\.path\.(append|insert)|importlib\.", c.source, re.M):
            err(f"{tag}: runs or imports code from files (%run, exec, sys.path, importlib); put the code in the notebook")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                   [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else []
            for m in mods:
                if m.split(".")[0] in local:
                    err(f"{tag}: imports your own file {m.split('.')[0]}.py; put that code in the notebook")


def check_designs_3d(d, name, x3):
    if list(d.columns) != ["design"] + COLS_3D:
        err(f"{name}: columns {list(d.columns)}; expected {['design'] + COLS_3D}")
    elif sorted(d.design) != ["energy", "sigma_crit"] or len(d) != 2:
        err(f"{name}: two rows, with design = sigma_crit and design = energy")
    else:
        bad = ~d[COLS_3D].apply(lambda s: s.between(x3[s.name].min(), x3[s.name].max())).all(axis=1)
        for design in d.design[bad]:
            err(f"{name}: design {design} is outside the range of the 3-d dataset (Task 1.1)")


def check_submission_7d(s, name, lo7, hi7):
    if list(s.columns) != COLS_7D + ["predicted_sigma_crit"]:
        err(f"{name}: columns {list(s.columns)}; expected {COLS_7D + ['predicted_sigma_crit']}")
    elif len(s) != 5 or not np.isfinite(s.to_numpy(float)).all():
        err(f"{name}: five rows of finite numbers")
    else:
        for i, row in s.iterrows():
            out = [c for c in COLS_7D if not lo7[c] <= row[c] <= hi7[c]]
            if out:
                err(f"{name}: design {i + 1} is outside the bounds in {out} (it would score zero)")


def check_report(root):
    """The answer labels (Answer 1.1. ... Answer 6.3.) in order, and the number of pages."""
    f = root / "report.pdf"
    if not f.exists():
        return
    try:
        from pypdf import PdfReader
    except ImportError:
        warn("report.pdf not checked: install pypdf (pip install pypdf) to check its answer labels and length")
        return
    pages = PdfReader(f).pages
    if len(pages) > MAX_PAGES:
        err(f"report.pdf has {len(pages)} pages; at most a cover page and 15 pages, plus the references")
    text = re.sub(r"\s+", " ", " ".join(page.extract_text() or "" for page in pages))
    labels = [num for n in TASKS for num in TASKS[n]]
    where = [text.find(f"Answer {num}.") for num in labels]
    missing = [num for num, k in zip(labels, where) if k < 0]
    if missing:
        err(f"report.pdf: no answer labelled Answer {missing[0]}. (missing: {', '.join(missing)}); start each answer with its number in bold")
    else:
        if where != sorted(where):
            err("report.pdf: the answers are not in the order of the statement")


def check_csvs(root):
    data = root / "data"
    try:
        x3 = pd.read_csv(data / "supercompressible_3d" / "input.csv")[COLS_3D]
        x7 = pd.read_csv(data / "supercompressible_7d" / "input.csv")[COLS_7D]
    except Exception as e:  # noqa: BLE001
        err(f"cannot read data/ as released: {e}")
        return
    lo7, hi7 = x7.min(), x7.max()
    lo7["ratio_shear_modulus"], hi7["ratio_shear_modulus"] = 0.035, GE_MAX

    for name in ["designs_3d.csv", "designs_3d_agents.csv"]:
        if (root / name).exists():
            check_designs_3d(pd.read_csv(root / name), name, x3)
    for name in ["submission.csv", "submission_agents.csv"]:
        if (root / name).exists():
            check_submission_7d(pd.read_csv(root / name), name, lo7, hi7)
    f = root / "task2_models.csv"
    if f.exists() and len(pd.read_csv(f)) < 2:
        err("task2_models.csv: one row for every model you trained in Task 2")


def check_studies(root):
    for name in STUDIES:
        study = root / "studies" / name
        if not study.is_dir():
            err(f"studies/{name}: missing")
            continue
        if not (study / "PROBLEM_STATEMENT.md").exists():
            err(f"studies/{name}: no PROBLEM_STATEMENT.md")
        runs = sorted(study.glob("runs/*/debug/run_status.json"))
        if not runs:
            err(f"studies/{name}: no completed adda run (runs/*/debug/run_status.json)")
        for f in runs + sorted(study.glob("runs/*/debug/node_models.json")):
            text = json.dumps(json.loads(f.read_text()))
            models = set(re.findall(r'"model_resolved":\s*"([^"]+)"', text))
            other = sorted(m for m in models if "haiku" not in m.lower())
            if other:
                err(f"{f.relative_to(root)}: agents ran on {other}; every adda agent must use Claude Haiku")


def rerun(root):
    """Run the six notebooks in a copy of the folder and compare the CSV files they write with the submitted ones."""
    import nbformat
    from nbclient import NotebookClient
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "submission"
        work.mkdir()
        for p in root.iterdir():
            if p.name in ("studies", "data"):
                (work / p.name).symlink_to(p.resolve())  # read-only use; never rewritten
            elif p.is_dir():
                shutil.copytree(p, work / p.name)
            else:
                shutil.copy(p, work / p.name)
        for n in TASKS:
            path = work / f"task{n}.ipynb"
            if not path.exists():
                continue
            nb = nbformat.read(path, as_version=4)
            t0 = time.time()
            try:
                NotebookClient(nb, timeout=TIME_LIMIT, kernel_name="python3",
                               resources={"metadata": {"path": str(work)}}).execute()
            except Exception as e:  # noqa: BLE001
                err(f"task{n}.ipynb: fails when run again from the top: {str(e).strip().splitlines()[-1][:200]}")
                continue
            dt = time.time() - t0
            print(f"  task{n}.ipynb ran in {dt:.0f} s")
            if dt > TIME_LIMIT:
                err(f"task{n}.ipynb: took {dt:.0f} s; the limit is {TIME_LIMIT} s (cache expensive results with cached())")
        for name in CSVS:
            a, b = root / name, work / name
            if a.exists() and b.exists():
                da, db = pd.read_csv(a), pd.read_csv(b)
                num = da.select_dtypes("number").columns
                same = da.shape == db.shape and list(da.columns) == list(db.columns) and \
                    np.allclose(da[num].to_numpy(float), db[num].to_numpy(float), rtol=1e-6, equal_nan=True)
                if not same:
                    err(f"{name}: running the notebooks again writes a different file; submit the file your notebooks write")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default=".")
    ap.add_argument("--no-run", action="store_true", help="skip running the notebooks again")
    a = ap.parse_args()
    root = Path(a.folder).resolve()
    print(f"Checking {root}")
    check_files(root)
    for n in TASKS:
        check_notebook(root, n)
    check_csvs(root)
    if (root / "studies").is_dir():
        check_studies(root)
    check_report(root)
    if not a.no_run:
        print("Running the six notebooks again (up to 10 minutes each)...")
        rerun(root)
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:  ", e)
    print(f"\n{'PASS' if not errors else f'FAIL: {len(errors)} errors'} ({len(warnings)} warnings)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
