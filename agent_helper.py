"""Run an AI coding agent from a Jupyter notebook: the helper used by the lectures and the homeworks of 3dasm.

Usage in a notebook (the path points to the folder that contains this file):

    import sys; sys.path.insert(0, "../..")
    from agent_helper import set_model, set_workspace, run_agent, results, figures, show
    set_model("sonnet")          # the model you use; whoever re-runs your notebook uses the same one
    set_workspace("agent_work")  # where the task folders go (default: the notebook's own folder)

    TASK_1 = "car_quadratic"
    PROMPT_1 = '''Fit ... and plot it. results.json keys: "w".'''
    run_agent(TASK_1, PROMPT_1)  # a fresh agent, working in its own task folder: agent_work/car_quadratic/
    r = results(TASK_1)          # agent_work/car_quadratic/results.json
    show(*figures(TASK_1))       # every figure this agent made

Every call to run_agent starts a NEW agent (`claude -p`) in its own task folder. It knows only the prompt, the
instructions file CLAUDE.md next to the notebook (which every task folder below it inherits), and the files it can see.
Everything the agent produces goes in its task folder, together with its own report (agent_report.md: read it, its
Assumptions are there) and the transcript of everything it did (agent_transcript.jsonl, recorded by Claude Code, not
by the agent). The helper prints what the agent looked at and ran, the files it made, the code it wrote, and its report.

The task folder belongs to the agent. Running the notebook again does not call the agent again, unless the prompt, the
model or the instructions changed: then the folder is emptied and the agent runs again. To run a prompt again from
scratch, delete its task folder.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STAMP, REPORT, TRANSCRIPT = "agent_stamp.json", "agent_report.md", "agent_transcript.jsonl"
HELPER_VERSION = "2026-09-29"                # part of every fingerprint: changing how agents run invalidates old runs
TOOLS = "Read Glob Grep Write Edit Bash"     # the only tools the agent has: read, write and run code
_model, _workspace, _warned = None, Path("."), False


def set_model(model):
    """Declare the model used for every prompt of this notebook (e.g. "sonnet", "haiku", "opus")."""
    global _model
    _model = model


def set_workspace(folder="."):
    """Declare the folder, relative to the notebook, where the task folders are created."""
    global _workspace
    _workspace = Path(folder)


def _check_task(task):
    if not re.fullmatch(r"[A-Za-z0-9_]+", task):
        raise ValueError(f"task name {task!r}: use only letters, digits and underscores (it becomes a folder name).")


def _instruction_files(task_folder):
    """Every instructions file an agent in `task_folder` would load: in its folder and above, and the personal one."""
    found = []
    for d in [task_folder.resolve(), *task_folder.resolve().parents]:
        found += [d / name for name in ("CLAUDE.md", "AGENTS.md") if (d / name).exists()]
    personal = Path.home() / ".claude" / "CLAUDE.md"
    return found + ([personal] if personal.exists() else [])


def _warn_outside(files):
    """Warn once about instructions files that this notebook does not control."""
    global _warned
    notebook_folder = Path.cwd().resolve()
    outside = [f for f in files if f.parent != notebook_folder and notebook_folder not in f.parents]
    if outside and not _warned:
        _warned = True
        print("WARNING: these instructions files are loaded too, but they are not part of this notebook:\n  "
              + "\n  ".join(str(f) for f in outside)
              + "\nAnyone who re-runs the notebook without them may get different results.\n")


def task_folder(task):
    """The folder of a task: <workspace>/<task>."""
    return _workspace / task


def run_agent(task, prompt):
    """Give `prompt` to a fresh agent working in its own task folder, unless it already ran with this same prompt."""
    _check_task(task)
    if "YOUR PROMPT HERE" in prompt:
        raise ValueError(f"[{task}] write your prompt first.")
    if _model is None:
        raise ValueError("call set_model(...) first, e.g. set_model('sonnet').")
    folder = task_folder(task)
    instructions = _instruction_files(folder)
    _warn_outside(instructions)
    fingerprint = hashlib.sha256("\n".join(
        [HELPER_VERSION, TOOLS, _model, prompt.strip()]
        + [f"{f.name}\n{f.read_text()}" for f in instructions]).encode()).hexdigest()[:16]

    stamp = folder / STAMP
    if stamp.exists():
        done = json.loads(stamp.read_text())
        if done.get("fingerprint") == fingerprint:
            print(f"[{task}] skipped: this prompt already ran ({done['wall_s']:.0f} s, model {done['model']}). "
                  f"To run it again from scratch, delete the folder {folder}.")
            _print_outcome(folder)
            return
        print(f"[{task}] the prompt, the model, the instructions or the helper changed: emptying {folder} and running again.")
        shutil.rmtree(folder)
    elif folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"[{task}] {folder} exists and was not made by run_agent: choose another task name.")

    claude = shutil.which("claude")
    if claude is None:
        raise RuntimeError("Claude Code is not installed or not on the PATH, so this cell cannot run an agent. "
                           "The outputs saved in this notebook are the recorded run.")
    folder.mkdir(parents=True, exist_ok=True)
    # Only these tools exist for the agent (no web, no sub-agents), and no skills or connectors installed on this
    # machine: what the agent can do must not depend on whose computer runs the notebook.
    cmd = [claude, "-p", prompt.strip(), "--model", _model, "--output-format", "stream-json", "--verbose",
           "--tools", TOOLS, "--allowedTools", TOOLS, "--disable-slash-commands", "--strict-mcp-config"]
    # The agent runs `python`: make it the same Python as this notebook, so it has the same packages.
    # Claude Code's automatic memory is switched off: an agent that remembers your other sessions would not be
    # reproducible by anyone else (and would know things your prompt did not tell it).
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
           "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=folder)
    wall = time.time() - t0
    (folder / TRANSCRIPT).write_text(out.stdout)
    final = [e.get("result", "") for e in _events(folder) if e.get("type") == "result"]
    (folder / REPORT).write_text((final[-1] if final else "") + (f"\n\nSTDERR:\n{out.stderr}" if out.stderr else ""))
    if out.returncode != 0:
        raise RuntimeError(f"[{task}] the agent failed after {wall:.0f} s; see {folder / REPORT}")
    stamp.write_text(json.dumps({"wall_s": wall, "model": _model, "fingerprint": fingerprint}, indent=1))
    print(f"[{task}] ran in {wall:.0f} s with model {_model}.")
    _print_outcome(folder)


def _events(folder):
    """The transcript, one event per line (lines that cannot be read are skipped)."""
    events = []
    for line in (folder / TRANSCRIPT).read_text().splitlines() if (folder / TRANSCRIPT).exists() else []:
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    return events


def _actions(folder):
    """(tool, what, full) for every tool the agent called, with paths relative to its task folder."""
    here = folder.resolve()
    ancestors = [here, *here.parents][:-1]           # deepest first; "/" itself is never rewritten

    def relative(text):                              # "/home/me/.../Lecture10/x" -> "../../x": no usernames in outputs
        for i, a in enumerate(ancestors):
            text = text.replace(str(a), "." if i == 0 else "/".join([".."] * i))
        return text

    def rel(path):
        try:
            return os.path.relpath(Path(path).resolve(), here)
        except (TypeError, ValueError):
            return str(path)

    actions = []
    for e in _events(folder):
        message = e.get("message")
        for c in message.get("content", []) if isinstance(message, dict) else []:
            if not (isinstance(c, dict) and c.get("type") == "tool_use"):
                continue
            name, inp = c.get("name", "?"), c.get("input", {}) or {}
            if name in ("Read", "Write", "Edit"):
                what = rel(inp.get("file_path", "?"))
                actions.append((name, what, what))
            elif name in ("Glob", "Grep"):
                what = f"{inp.get('pattern', '?')} in {rel(inp.get('path', here))}"
                actions.append((name, what, what))
            elif name == "Bash":
                full = relative(inp.get("command", "").strip())
                lines = full.splitlines() or [""]
                what = lines[0][:110] + (f"   (+{len(lines) - 1} more lines)" if len(lines) > 1 else "")
                actions.append((name, what, full))
            else:
                actions.append((name, "", ""))
    return actions


def _print_outcome(folder):
    """What the agent did and left in its folder: its actions, its files, its code, what it stored, its report."""
    actions = _actions(folder)
    if actions:
        print("What it did (from the transcript):")
        for name, what, _ in actions:
            print(f"  {name:5} {what}")
        peeks = sorted({m for _, _, full in actions
                        for m in re.findall(r"[\w./-]*(?:\.ipynb|CLAUDE\.md|AGENTS\.md)", full)})
        if peeks:
            print("NOTE: the agent looked at a notebook or an instructions file:\n  " + "\n  ".join(peeks))
        print()
    made = [f for f in files_in(folder) if f.name not in (STAMP, REPORT, TRANSCRIPT)]
    print("Files the agent made:\n  " + ("\n  ".join(str(f.relative_to(folder)) for f in made) or "(none)"))
    for f in made:
        if f.suffix == ".py":
            print(f"\n--- the code it wrote: {f.relative_to(folder)} ---\n{f.read_text().rstrip()}")
    if (folder / "results.json").exists():
        print("\n--- results.json ---\n" + json.dumps(json.loads((folder / "results.json").read_text())))
    report = folder / REPORT
    print("\n--- its report ---\n" + (report.read_text().strip() if report.exists() else "(none)") + "\n")


def files_in(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and ".ipynb_checkpoints" not in p.parts)


def results(task, keys=None):
    """Load the numbers of a task from <workspace>/<task>/results.json.

    With `keys`, check first that the agent stored all of them, and say plainly which ones are missing.
    """
    path = task_folder(task) / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"[{task}] the agent did not write {path}: say so in your prompt "
                                f"(or in your CLAUDE.md), and run the prompt again.")
    values = json.loads(path.read_text())
    missing = [k for k in (keys or []) if k not in values]
    if missing:
        raise KeyError(f"[{task}] results.json has no key {', '.join(repr(k) for k in missing)}: "
                       f"add it to your prompt and run the prompt again.")
    return values


def figures(task):
    """The figures (PNG files) in the task folder."""
    return [f for f in files_in(task_folder(task)) if f.suffix.lower() == ".png"]


def save_results(task, values):
    """Only if you write your own code: store a task's numbers in <workspace>/<task>/results.json."""
    _check_task(task)
    task_folder(task).mkdir(parents=True, exist_ok=True)
    (task_folder(task) / "results.json").write_text(json.dumps(values, indent=1))


def show(*pngs, width=900):
    """Display figures (PNG files) in the notebook."""
    from IPython.display import Image, display
    for p in pngs:
        display(Image(filename=str(p), width=width))
