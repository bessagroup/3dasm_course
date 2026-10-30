"""Write the slideshow style into every lecture notebook (the notebook is the source of truth for its slides).

JupyterLab's RISE reads the notebook metadata "rise" and inserts its "overlay" on every slide: here, a <style> block.
It also reads "slideNumber" there, so every viewer sees the slide numbers (not only those who set it in their settings).
It changes only the slideshow, never the notebook view. Sizes are fractions of the screen height (vh), so the slides
look the same on any screen (tested on a 1920x1080 laptop and an iPad); browser zoom no longer changes slide text.
The script also saves the cells that run an agent in scroll mode, so their long report starts collapsed.

Usage, from the root of the repository:  python Lectures/set_slide_style.py
Edit CSS below and re-run to change the style of all lectures.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = "[data-mime-type='application/vnd.jupyter.stdout']"
ERROR = "[data-mime-type='application/vnd.jupyter.stderr']"
S = ".rise-enabled .reveal"
CSS = [
    # text sizes (22px, 17px and 18px on a 1080-pixel-tall screen)
    f"{S} .jp-Notebook{{--jp-content-font-size1:max(14px,2.05vh);}}",      # markdown
    f"{S} .jp-InputArea{{--jp-code-font-size:max(12px,1.6vh);}}",          # code cells
    f"{S} .jp-OutputArea{{--jp-code-font-size:max(12px,1.7vh);}}",         # text outputs, one step larger
    f"{S} .jp-OutputArea pre{{font-size:var(--jp-code-font-size);}}",       # RISE fixes it at 14px otherwise
    f"{S} .jp-PlaceholderText{{font-size:max(12px,1.6vh);}}",              # first line of a cell with hidden source
    # outputs: collapsed (scroll mode on) at most 55% of the screen, expanded at most 80% (room for a line above/below)
    f"{S} .jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea{{max-height:55vh;overflow-y:auto;}}",
    f"{S} .jp-CodeCell:not(.jp-mod-outputsScrolled) .jp-Cell-outputArea{{max-height:80vh;overflow-y:auto;}}",
    # margins: 7.5% on each side, content centered
    f"{S} .jp-Cell{{max-width:85%;margin-left:auto !important;margin-right:auto !important;}}",
]
OVERLAY = "<style>" + "".join(CSS) + "</style>"
SLIDE_NUMBER = "c/t" # the slide number and the total, bottom right (the pace of a lecture at a glance)


def main():
    notebooks = sorted(p for p in ROOT.glob("Lectures/Lecture*/*.ipynb") if ".ipynb_checkpoints" not in p.parts)
    for path in notebooks:
        raw = path.read_text()
        nb = json.loads(raw)
        nb["metadata"].setdefault("rise", {})["overlay"] = OVERLAY
        nb["metadata"]["rise"]["slideNumber"] = SLIDE_NUMBER
        agents = 0
        for cell in nb["cells"]:
            if cell["cell_type"] == "code" and "run_agent(" in "".join(cell["source"]):
                cell["metadata"]["scrolled"] = True
                agents += 1
        new = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
        if new != raw:
            path.write_text(new)
            print(f"styled {path.relative_to(ROOT)}" + (f" ({agents} agent cells collapsed)" if agents else ""))


if __name__ == "__main__":
    main()
