"""Stage 2 (only when two sources need to match): measure the colour difference on drawings present in both clips, fit a colour map, and test it on pairs held out of the fit.
Error is CIE76 delta E; under about 5 reads as the same colour at a glance. edit.json: "color_fit": {"from": "G", "to": "T", "out": "grade.npy", "pairs": {"name": [from_start, from_end, to_start, to_end], ...}}
usage: color_fit.py edit.json"""
import subprocess
import sys

import cv2
import numpy as np

from lib import Project

P = Project(sys.argv[1]); cf = P.cfg["color_fit"]; W, H = 192, 82
pa, pb = P.cfg["sources"][cf["from"]]["path"], P.cfg["sources"][cf["to"]]["path"]


def read(path, a, b, crop=None):
    vf = f"fps=30,{crop + ',' if crop else ''}scale={W}:{H}:flags=area"
    o = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(b - a), "-i", path, "-map", "0:v:0", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(o, np.uint8).reshape(-1, H, W, 3)


lab = lambda x: cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32) * np.array([100 / 255, 1, 1])


def feats(x):
    x = x.astype(np.float64) / 255
    b, g, r = x[:, 0], x[:, 1], x[:, 2]
    return np.c_[np.ones(len(x)), b, g, r, b * b, g * g, r * r, b * g, b * r, g * r]


data = {}
crop_to = P.cfg["sources"][cf["to"]]["vf"].split(",")[0] if "crop" in P.cfg["sources"][cf["to"]]["vf"] else None
norm = lambda m: (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-6)
gray = lambda f: cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (0, 0), 2).ravel().astype(np.float32)
for k, (ga, gb, ta, tb) in cf["pairs"].items():
    gf, tf = read(pa, ga, gb), read(pb, ta, tb, crop_to)
    c = norm(np.array([gray(f) for f in gf[::4]])) @ norm(np.array([gray(f) for f in tf[::2]])).T / (W * H)
    ps, qs = [], []
    for i in range(c.shape[0]):
        j = int(c[i].argmax())
        if c[i, j] > .8:   # same drawing
            ps.append(cv2.GaussianBlur(gf[i * 4], (0, 0), 2.5).reshape(-1, 3)); qs.append(cv2.GaussianBlur(tf[j * 2], (0, 0), 2.5).reshape(-1, 3))
    print(k, "best match", round(float(c.max()), 2), "frames used", len(ps), flush=True)
    if ps:
        data[k] = (np.concatenate(ps), np.concatenate(qs))


def fit(keys):
    X = np.concatenate([feats(data[k][0][::3]) for k in keys]); Y = np.concatenate([data[k][1][::3].astype(np.float64) / 255 for k in keys])
    return np.linalg.solve(X.T @ X + 1e-5 * len(X) * np.eye(X.shape[1]), X.T @ Y)


apply = lambda m, p: np.clip(feats(p) @ m * 255, 0, 255)
de = lambda a, b: float(np.median(np.linalg.norm(lab(a) - lab(b), axis=1)))
bb, aa = [], []
for k in data:
    m = fit([q for q in data if q != k]); p, q = data[k]; bb.append(de(p, q)); aa.append(de(apply(m, p), q)); print(k, "held out: delta E before", round(bb[-1], 1), "after", round(aa[-1], 1))
print("median delta E before", round(float(np.median(bb)), 1), "after", round(float(np.median(aa)), 1))
np.save(P.work / cf["out"], fit(list(data)))
print("saved", P.work / cf["out"], '- set it as sources[...]["grade"]["matrix"]')
