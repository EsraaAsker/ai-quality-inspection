#!/usr/bin/env python3
"""Download the MVTec AD metal_nut subset into the standard directory layout.

The official MVTec archive URL has changed over time; this helper defaults to a
public mirror of the subset. Use --source official with a current archive URL if
one is available to you.
"""
import argparse, json, urllib.request
from pathlib import Path

MIRROR_API = "https://huggingface.co/api/datasets/MSherbinii/mvtec-ad-metal-nut/tree/main?recursive=true&expand=false"
MIRROR_BASE = "https://huggingface.co/datasets/MSherbinii/mvtec-ad-metal-nut/resolve/main/"

p = argparse.ArgumentParser(); p.add_argument("--output", default="data/mvtec_ad")
p.add_argument("--source", choices=["mirror", "official"], default="mirror")
p.add_argument("--url", help="Official .tar.xz URL when --source official is selected")
args = p.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

if args.source == "official":
    if not args.url: raise SystemExit("--url is required with --source official")
    archive = out.parent / "mvtec_ad.tar.xz"
    print(f"Downloading {args.url} -> {archive}"); urllib.request.urlretrieve(args.url, archive)
    import tarfile
    with tarfile.open(archive, "r:xz") as tf: tf.extractall(out.parent)
else:
    items = json.load(urllib.request.urlopen(MIRROR_API))
    files = [x["path"] for x in items if x.get("type") == "file" and x["path"].startswith("metal_nut/")]
    for i, rel in enumerate(files, 1):
        dest = out / rel; dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists(): urllib.request.urlretrieve(MIRROR_BASE + rel + "?download=true", dest)
        if i % 25 == 0: print(f"Downloaded {i}/{len(files)} files")
print(f"Dataset ready at {out / 'metal_nut'}")
