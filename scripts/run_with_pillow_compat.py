#!/usr/bin/env python
import runpy
import sys
from PIL import Image
import numpy as np

for _name, _value in {"int": int, "float": float, "bool": bool}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)

_aliases = {
    "LINEAR": "BILINEAR",
    "CUBIC": "BICUBIC",
    "ANTIALIAS": "LANCZOS",
}
for old, new in _aliases.items():
    if not hasattr(Image, old) and hasattr(Image, new):
        setattr(Image, old, getattr(Image, new))

if len(sys.argv) < 2:
    raise SystemExit("usage: run_with_pillow_compat.py SCRIPT [args...]")
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
