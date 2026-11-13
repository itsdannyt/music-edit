"""Place a word on a real surface: the word is warped onto a quad drawn on one frame, and the surface is tracked by homography (feature points on the surface, the subject's matte excluded) so it stays fixed to that surface as the camera moves. fx.py bakes it into the picture, so it sits under grain and behind the subject's matte.
edit.json fx.wall_text: [{"name","src","t","ref","word","vp":[x,y],"x0","x1","y_top","y_bot","surface":[x0,y0,x1,y1],"from","to"}]. Picture coordinates. y_top, y_bot are the word's height at x0; its other edge follows lines through the vanishing point vp.
usage: wall_text.py edit.json   writes work/walls/{name}.npz and work/sheets/walls.png"""
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np

from lib import Project, cadence_smooth
from wall_lib import composite

P = Project(sys.argv[1]); W, ACT, TOP = P.w, P.act, P.top; OUT = P.work / "walls"; OUT.mkdir(exist_ok=True); cells = []
F = os.path.expanduser("~/.cache/hyperframes/fonts/poppins/900-normal-*.woff2")
CH = glob.glob(os.path.expanduser("~/.cache/hyperframes/chrome/chrome-headless-shell/*/*/chrome-headless-shell"))[0]
cap = cv2.VideoCapture(str(P.work / "picture_raw.mp4"))


def frame(i):
    cap.set(1, i); ok, f = cap.read(); return f[TOP:TOP + ACT].copy()


def shoot(body, face, weight, extra=""):
    files = glob.glob(os.path.expanduser(f"~/.cache/hyperframes/fonts/{face}/{weight}-normal-*.woff2"))
    css = "".join(f"@font-face{{font-family:w{i};src:url('file://{p}')}}" for i, p in enumerate(files))
    fam = ",".join(f"w{i}" for i, _ in enumerate(files)); h = OUT / "w.html"; png = OUT / "w.png"
    h.write_text(f"<!doctype html><meta charset=utf-8><style>{css}html,body{{margin:0;background:#000}}div{{font:{weight} 600px {fam};color:#fff;padding:100px;white-space:nowrap;letter-spacing:-.01em;{extra}}}</style><div>{body}</div>")
    subprocess.run([CH, "--headless", "--disable-gpu", "--allow-file-access-from-files", "--hide-scrollbars", "--window-size=5200,2400", "--virtual-time-budget=3000", f"--screenshot={png}", f"file://{h}"], capture_output=True)
    a = cv2.imread(str(png), 0); h.unlink(); png.unlink(); return a


def word_alpha(word, face="poppins", weight=900, extra=""):
    """One word, or a list: then each word gets its own mask in the same layout, so words can light one at a time."""
    words = [word] if isinstance(word, str) else word
    masks = [shoot(" ".join(f'<span style="color:{"#fff" if k == j else "transparent"}">{x}</span>' for k, x in enumerate(words)), face, weight, extra) for j in range(len(words))]
    u = np.max(masks, 0); xs, ys = np.where(u.max(0) > 60)[0], np.where(u.max(1) > 60)[0]
    return np.array([m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] for m in masks])


for w in P.cfg.get("fx", {}).get("wall_text", []):
    c = P.clip(w["src"], w["t"]); ms = P.matte(c, "sam") if w.get("matte", True) else np.zeros((c["e"] - c["a"], ACT // 2, W // 2), np.uint8); ref = w["ref"]; a0, e0 = w["from"], w["to"]; sc = 2; WS, HS = W // sc, ACT // sc
    if "stand" in w:      # letters upright on the ground. Ground points: the base line and where the shadow of the top lands.
        st = w["stand"]; yh = st["horizon"]; base = np.float32(st["base"]); hpx = (base[:, 1] - yh) / st["cam_m"] * st["height_m"]; svp = np.float32(st["shadow_vp"]); D = st["shadow_per_m"] * st["height_m"]; tips = []
        for b_ in base:
            y_ = yh + 1 / (1 / (b_[1] - yh) - D); tips.append([b_[0] + (svp[0] - b_[0]) * (y_ - b_[1]) / (svp[1] - b_[1]), y_])
        quad = np.vstack([base, np.float32(tips)])
    elif "quad" in w:
        quad = np.float32(w["quad"])
    else:
        vp = np.float64(w["vp"]); ray = lambda y, x: y + (vp[1] - y) * (x - w["x0"]) / (vp[0] - w["x0"])
        quad = np.float32([[w["x0"], w["y_top"]], [w["x1"], ray(w["y_top"], w["x1"])], [w["x1"], ray(w["y_bot"], w["x1"])], [w["x0"], w["y_bot"]]])
    H_ = {i: np.eye(3) for i in range(a0, e0)}; errs = [0.]
    if w.get("track", True):
        gray = lambda i: cv2.cvtColor(cv2.resize(frame(i), (WS, HS), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        G = {i: gray(i) for i in range(a0, e0)}; lk = dict(winSize=(25, 25), maxLevel=3, criteria=(3, 30, .01)); H_ = {ref: np.eye(3)}; errs = []; x0, y0, x1, y1 = [v // sc for v in w["surface"]]
        for seq, st in ((range(ref + 1, e0), 1), (range(ref - 1, a0 - 1, -1), -1)):
            H = np.eye(3)
            for i in seq:
                p = i - st; mask = np.zeros((HS, WS), np.uint8); mask[y0:y1, x0:x1] = 255
                mask[cv2.dilate((cv2.resize(ms[p - c["a"]], (WS, HS)) > 128).astype(np.uint8), np.ones((15, 15))) > 0] = 0
                A = cv2.goodFeaturesToTrack(G[p], 300, .01, 6, mask=mask); B, s1, _ = cv2.calcOpticalFlowPyrLK(G[p], G[i], A, None, **lk); Bk, s2, _ = cv2.calcOpticalFlowPyrLK(G[i], G[p], B, None, **lk)
                g = (s1[:, 0] == 1) & (s2[:, 0] == 1) & (np.abs(A - Bk).reshape(-1, 2).max(1) < .7); h, inl = cv2.findHomography(A[g], B[g], cv2.RANSAC, 1.5); H = h @ H; H_[i] = H / H[2, 2]
                errs.append(float(np.median(np.linalg.norm((cv2.perspectiveTransform(A[g], h) - B[g]).reshape(-1, 2), axis=1)[inl[:, 0] == 1])) * sc)
    S_ = np.diag([sc, sc, 1.]); Si = np.diag([1 / sc, 1 / sc, 1.])
    C = np.array([cv2.perspectiveTransform(quad[None], (S_ @ H_[i] @ Si).astype(np.float32))[0] for i in range(a0, e0)])
    C = cadence_smooth(C, [gray(i) for i in range(a0, e0)]) if w.get("track", True) else C; SH_ = None
    if "stand" in w:      # height follows the base line's own scale, so the letters keep their real height as the camera moves
        bL, bR, tL, tR = C[:, 0], C[:, 1], C[:, 2], C[:, 3]; sc_ = np.linalg.norm(bR - bL, axis=1) / np.linalg.norm(base[1] - base[0]); up = lambda b_, h_: b_ - np.c_[np.zeros(len(b_)), h_ * sc_]
        SH_ = np.stack([tL, tR, bR, bL], 1); C = np.stack([up(bL, hpx[0]), up(bR, hpx[1]), bR, bL], 1)
    alpha = word_alpha(w["word"], w.get("font", "poppins"), w.get("weight", 900), w.get("css", "")); np.savez(OUT / f"{w['name']}.npz", corners=C, alpha=alpha, a=a0, **({} if SH_ is None else dict(shadow=SH_)))
    print(w["name"], "frames", a0, e0, "track error median px", round(float(np.median(errs)), 3), "worst", round(max(errs), 3), "corners x", C[..., 0].min().round(), C[..., 0].max().round(), "y", C[..., 1].min().round(), C[..., 1].max().round(), "word aspect", round(alpha.shape[2] / alpha.shape[1], 2), flush=True)
    r = np.float32([[0, 0], [alpha.shape[2], 0], [alpha.shape[2], alpha.shape[1]], [0, alpha.shape[1]]])
    for i in w.get("preview", (a0, ref, e0 - 1)):
        f = frame(i); cells.append(cv2.resize(cv2.cvtColor(np.zeros((1, 1), np.uint8), cv2.COLOR_GRAY2BGR), (960, 403)))
        cells[-1] = cv2.resize(composite(f, w, C[i - a0], alpha, i, ms[i - c["a"]], None if SH_ is None else SH_[i - a0]), (960, 403))
if cells:
    n = 3; cells += [np.zeros_like(cells[0])] * (-len(cells) % n); cv2.imwrite(str(P.work / "sheets/walls.png"), np.vstack([np.hstack(cells[i:i + n]) for i in range(0, len(cells), n)]))
