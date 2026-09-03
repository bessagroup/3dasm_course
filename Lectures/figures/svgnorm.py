#!/usr/bin/env python3
"""Normalise an SVG so successive Illustrator saves produce readable diffs.

Illustrator rewrites coordinate precision, attribute order and element order on
every save, so a plain `git diff` of two exports can show hundreds of changed
lines when you moved one circle. This rewrites the file into a canonical form:
one element per line, numbers rounded, attributes in a fixed order. Run it after
every export and the diff shows the circle.

    python Lectures/figures/svgnorm.py Lectures/figures/adda_hero.svg

It also refuses to stay quiet about the two export settings that destroy the
round-trip: internal CSS classes, and text converted to outlines.
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET

SVG = "http://www.w3.org/2000/svg"
NUM = re.compile(r"-?\d+\.\d+")
# `d` and `points` are left BYTE-FOR-BYTE ALONE, deliberately. Compact SVG path
# syntax lets arc flags run together with the coordinate that follows them
# ("0 10-9" is flags 1,0 then -9), so rounding path data safely requires a
# command-aware parser, not a tokeniser. Getting that subtly wrong corrupts
# geometry silently, which is far worse than a noisy diff. Consequence: the
# icon paths may still diff noisily. They are also the part nobody hand-edits.
# viewBox joins them: the icons are nested <svg> elements whose viewBox is a
# computed square around each glyph's ink, and rounding it to 2dp rescales the
# glyph by a fraction of a pixel. Harmless to look at, but it means the tool is
# no longer provably pixel-neutral, and that guarantee is the point.
PATH_ATTRS = {"d", "points", "viewBox"}
# id first, then geometry, then paint, then everything else alphabetically
ORDER = ["id", "d", "points", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy",
         "r", "rx", "ry", "width", "height", "viewBox", "transform",
         "fill", "fill-opacity", "fill-rule", "stroke", "stroke-width",
         "stroke-dasharray", "stroke-linecap", "stroke-linejoin", "opacity",
         "font-family", "font-size", "font-weight", "text-anchor"]


def round_nums(v: str, nd: int = 2) -> str:
    def one(m):
        s = f"{float(m.group()):.{nd}f}".rstrip("0").rstrip(".")
        return s if s not in ("", "-") else "0"
    return NUM.sub(one, v)


def key(a: str):
    return (ORDER.index(a), "") if a in ORDER else (len(ORDER), a)


def write(el, out, depth=0, nd=2):
    tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
    attrs = []
    for a in sorted(el.attrib, key=key):
        name = a.split("}")[-1] if "}" in a and "xlink" not in a else a
        raw = el.attrib[a]
        val = raw if name in PATH_ATTRS else round_nums(raw, nd)
        attrs.append(f'{name}="{val}"')
    pad = "  " * depth
    head = f"{pad}<{tag}" + ("" if not attrs else " " + " ".join(attrs))
    kids = list(el)
    text = (el.text or "").strip()
    if not kids and not text:
        out.append(head + "/>")
        return
    if not kids and text:
        out.append(f"{head}>{text}</{tag}>")
        return
    out.append(head + ">")
    if text:
        out.append("  " * (depth + 1) + text)
    for k in kids:
        write(k, out, depth + 1, nd)
    out.append(f"{pad}</{tag}>")


def main(path: str, nd: int = 2) -> int:
    raw = open(path, encoding="utf-8").read()
    warn = []
    if re.search(r"<style", raw) or re.search(r'class="st\d', raw):
        warn.append("internal CSS classes found — re-export with "
                    "Styling: Presentation Attributes, or diffs are meaningless")
    if "<text" not in raw:
        warn.append("no <text> elements — the type looks outlined; "
                    "re-export with Fonts: SVG to keep it editable")
    if "<filter" in raw or "<image" in raw:
        warn.append("filters or raster images present — an Illustrator effect "
                    "was applied; that region is no longer editable as vectors")

    ET.register_namespace("", SVG)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(re.sub(r"<!--.*?-->", "", raw, flags=re.S))
    lines: list[str] = []
    write(root, lines, 0, nd)
    body = "\n".join(lines) + "\n"
    body = body.replace("<svg", f'<svg xmlns="{SVG}"', 1) if "xmlns=" not in lines[0] else body
    open(path, "w", encoding="utf-8").write(body)

    print(f"normalised {path}  ({len(lines)} lines, {nd} dp)")
    for w in warn:
        print("  WARNING:", w)
    return 1 if warn else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2))
