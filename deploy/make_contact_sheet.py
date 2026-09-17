#!/usr/bin/env python3
"""Build one labelled contact sheet from the two comparison series."""
import glob
import json
import sys
from PIL import Image, ImageDraw, ImageFont

OUT = "/tmp/sd_demo"
GALLERY = "/home/bianbu/image-generation/output/ui"  # proxy writes PNGs here
CELL = 384
PAD = 10
LABEL_H = 34


def load_font(size):
    """The default PIL bitmap font is latin-1 only, so it raises on CJK labels.
    Pick any installed CJK face instead."""
    for pat in (
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-*.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc",
        "/usr/share/fonts/**/Noto*CJK*.tt[cf]",
    ):
        for path in sorted(glob.glob(pat, recursive=True)):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    raise SystemExit("no CJK font found; labels would fail to render")

rows = [
    ("A) 同 seed=42，不同步数", [("steps1", "1 步"), ("steps2", "2 步"), ("steps4", "4 步"), ("steps8", "8 步")]),
    ("B) 同步数=8，不同种子", [("seed42", "seed 42"), ("seed43", "seed 43"), ("seed44", "seed 44")]),
]

cols = max(len(r[1]) for r in rows)
W = PAD + cols * (CELL + PAD)
H = PAD + len(rows) * (LABEL_H + CELL + PAD)
sheet = Image.new("RGB", (W, H), (18, 18, 22))
draw = ImageDraw.Draw(sheet)
title_font = load_font(20)
cap_font = load_font(17)

y = PAD
for title, items in rows:
    draw.text((PAD, y + 6), title, fill=(230, 230, 240), font=title_font)
    y += LABEL_H
    x = PAD
    for tag, label in items:
        meta = json.load(open("%s/%s.json" % (OUT, tag)))
        img = Image.open("%s/%s" % (GALLERY, meta["id"])).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
        sheet.paste(img, (x, y))
        text = "%s  ·  %.2fs" % (label, meta["elapsed"])
        # Dark plaque behind the caption so it stays readable on snow/bright art.
        tw = draw.textlength(text, font=cap_font)
        draw.rectangle([x + 4, y + CELL - 30, x + 12 + tw, y + CELL - 6], fill=(0, 0, 0))
        draw.text((x + 8, y + CELL - 28), text, fill=(255, 235, 120), font=cap_font)
        x += CELL + PAD
    y += CELL + PAD

path = "%s/contact_sheet.png" % OUT
sheet.save(path)
print("wrote", path, sheet.size)

# Same seed + same params must be byte-identical; print the check for the pair
# that appears in both series.
import hashlib
a = open("%s/%s" % (GALLERY, json.load(open("%s/steps8.json" % OUT))["id"]), "rb").read()
b = open("%s/%s" % (GALLERY, json.load(open("%s/seed42.json" % OUT))["id"]), "rb").read()
print("steps8 sha256 :", hashlib.sha256(a).hexdigest()[:16])
print("seed42 sha256 :", hashlib.sha256(b).hexdigest()[:16])
print("identical     :", a == b)
