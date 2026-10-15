"""Report every color in the lecture notebooks that is not in the course palette (nothing is changed).

The convention: color by the distribution a quantity comes from, and by the space it lives in.
  prior red, observation distribution / likelihood blue, posterior green, PPD orange, empirical (data summaries) magenta;
  light shades for the Uniform prior (and Least Squares), dark shades for a Gaussian prior (Ridge, Lasso, BLR, GPs).
  Green lives in the space of the hidden rvs z (the posterior of z or of the weights w). Anything predicted at a new
  input x* lives in y-space and is orange, filled or solid. Dashed means "the noise exists but is left out": the
  credible interval of mu_y|z of noisy data (the PPD without its aleatoric part) has a faint fill and dashed edges. When
  the noise is zero, the band IS the PPD of y* and is drawn filled, never dashed. In a
  decomposition of the PPD variance (law of total variance): total solid orange, epistemic part dashed orange (it only
  exists inside the PPD), aleatoric part blue (the variance of the observation distribution). Draws from a prior are
  shades of red.
Everything else is neutral: ground truth black dotted, data black, ranges and frames gray. In equations, gray marks
a quantity held fixed (the hyperparameters at the first level of Bayesian model selection, Lecture 17). Colors that only
highlight (the variables y, z, x in the first lectures, a term in a derivation, curves compared with each other) use
purple, teal and brown, which carry no Bayes meaning. Colormaps (cmap=...) are not checked.

Usage, from the root of the repository:  python Lectures/check_colors.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PALETTE = {
    "prior": {"red", "r", "#d62728", "#67000d", "#fc9272", "#a50f15", "#fb6a4a"},
    "likelihood": {"blue", "b", "#1f77b4", "rgba(31,119,180"},
    "posterior": {"green", "#82e0aa", "#1e8449", "#3ba768", "rgba(30,132,73"},
    "ppd": {"orange", "#f5b041", "#c0580b", "rgba(192,88,11"},
    "empirical": {"magenta", "m"},
    "highlight": {"purple", "teal", "brown", "saddlebrown"},
    "colormap": {"#2166ac", "#b2182b"},          # the two ends of RdBu, for text that reads a diverging colormap (L7)
    "neutral": {"k", "black", "white", "w", "none", "gray", "grey", "dimgray", "lightgray", "slategray",
                "rgba(0,0,0", "rgba(200,200,200"},
}
ALLOWED = set().union(*PALETTE.values())
GREY = re.compile(r"^(0?\.\d+|#([0-9a-f])\2\2|#([0-9a-f]{2})\3\3)$")      # '0.6', '#999', '#e8e8e8'

# colors in code: keyword arguments, JS styles, CSS, and matplotlib format strings such as 'ro' or 'k--'
CODE = re.compile(r"""\b(?:color|c|mec|mfc|facecolor|edgecolor|ecolor)\s*=\s*["']([^"']+)["']"""
                  r"""|(?:fillStyle|strokeStyle)\s*=\s*["']([^"']+)["']"""
                  r"""|color:\s*(#[0-9a-fA-F]{3,6}|rgba?\([^)]*\)|[a-z]+)"""
                  r"""|\.(?:plot|semilogy|loglog|errorbar)\([^#\n]*?(?<!\[)["']([-:.oXxs*^v<>d]*[rgbcmykw][-:.oXxs*^v<>d]*)["'](?!\])""")
MARKDOWN = re.compile(r"""\\color\{([^}]+)\}|<font color=["']?([#\w]+)""")


def normalize(col):
    col = col.strip().lower().replace(" ", "")
    m = re.match(r"(rgba?\(\d+,\d+,\d+)", col)
    if m:
        return m.group(1)
    if re.fullmatch(r"[-:.oxs*^v<>d]*[rgbcmykw][-:.oxs*^v<>d]*", col) and len(col) <= 4:
        return re.sub(r"[-:.oxs*^v<>d]", "", col)                       # a format string: keep its color letter
    return col


def main():
    notebooks = sorted(p for p in ROOT.glob("Lectures/Lecture*/*.ipynb") if ".ipynb_checkpoints" not in p.parts)
    found = 0
    for path in notebooks:
        for i, cell in enumerate(json.loads(path.read_text())["cells"]):
            pattern = MARKDOWN if cell["cell_type"] == "markdown" else CODE
            for line in "".join(cell["source"]).split("\n"):
                if "cmap" in line:
                    continue
                for m in pattern.finditer(line):
                    raw = next(g for g in m.groups() if g)
                    if raw.startswith("+") or raw.startswith("'+") or "D.col" in raw:
                        continue                                          # a color computed in JS from the palette
                    col = normalize(raw)
                    if col in ALLOWED or GREY.match(col) or col.startswith("#") and col.lstrip("#") in ("000", "111", "222", "333", "444", "555"):
                        continue
                    found += 1
                    print(f"{path.relative_to(ROOT)} [cell {i}]: {raw!r}  |  {line.strip()[:110]}")
    print(f"{found} color(s) outside the palette" if found else "every color is in the palette")


if __name__ == "__main__":
    main()
