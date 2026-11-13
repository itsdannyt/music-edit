"""Stage 3 measurements, from the raw picture.
flow:   at every cut, whole-frame motion leaving and entering (px/frame) and each side's action point. Push transitions aim at the action point; side whips are only allowed where the footage really pans.
track:  each object in edit.json "objects", by dense alignment (ECC) from the frame its box was drawn on out to both ends of the shot. Text and object effects pin to these.
usage: measure.py edit.json flow|track|all"""
import json
import subprocess
import sys

import cv2
import numpy as np

from lib import Project, cadence_smooth

P = Project(sys.argv[1]); what = sys.argv[2]; C = P.clips
SW, SH = 480, 204; SCALE = P.w / SW
if what in ("flow", "all"):
    cuts = [c["a"] for c in C[1:]]; N = 3; need = set()
    for f in cuts:
        need |= set(range(f - N - 1, f + N + 1))
    store = {}
    for i, f in P.frames(crop=True):
        if i in need:
            store[i] = cv2.cvtColor(cv2.resize(f, (SW, SH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    yy, xx = np.mgrid[0:SH, 0:SW].astype(np.float32)

    def side(fr):
        dx, dy, en = [], [], np.zeros((SH, SW), np.float32)
        for a, b in zip(fr[:-1], fr[1:]):
            fl = cv2.calcOpticalFlowFarneback(store[a], store[b], None, .5, 3, 21, 3, 5, 1.2, 0)
            mx, my = float(np.median(fl[..., 0])), float(np.median(fl[..., 1]))
            dx.append(mx); dy.append(my); en += (fl[..., 0] - mx) ** 2 + (fl[..., 1] - my) ** 2
        t = float(en.sum()) + 1e-6
        return dict(dx=round(float(np.mean(dx)) * SCALE, 1), dy=round(float(np.mean(dy)) * SCALE, 1), fx=round(float((en * xx).sum() / t) * SCALE), fy=round(float((en * yy).sum() / t) * SCALE))
    out = {}
    prev = {c["e"]: c for c in C}; at = {c["a"]: c for c in C}
    for f in cuts:
        la, ln = prev[f], at[f]; nl, nn = min(N, la["e"] - la["a"] - 1), min(N, ln["e"] - ln["a"] - 1)
        out[f] = dict(out=side(list(range(f - nl - 1, f))), inc=side(list(range(f, f + nn + 1))))
    json.dump(out, open(P.work / "flow.json", "w")); print("flow:", len(out), "cuts")
if what in ("track", "all"):
    CC_FLOOR = .5    # on the anime test, scores split into 0.59+ on tracked objects and 0.32 or less on failures; the floor sits in that gap. Below it a frame holds.
    SW2, SH2 = P.w // 2, P.act // 2
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(P.work / "picture_raw.mp4"), "-vf", f"crop={P.w}:{P.act}:0:{P.top},scale={SW2}:{SH2},format=gray", "-f", "rawvideo", "-"], stdout=subprocess.PIPE); F = []
    while True:
        b = p.stdout.read(SW2 * SH2)
        if len(b) < SW2 * SH2:
            break
        F.append(np.frombuffer(b, np.uint8).reshape(SH2, SW2).astype(np.float32))
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5); out = {}
    for name, o in P.cfg.get("objects", {}).items():
        c = P.clip(o["src"], o["t"]); a, e, ref, box = c["a"], c["e"], o["ref"], o["box"]
        if o.get("follow_matte"):      # a moving, deforming subject defeats ECC; its SAM cutout already knows where it is on every frame
            ms = P.matte(c); cen = []
            for q in ms:
                ys, xs = np.nonzero(q > 128); cen.append((xs.mean() * P.w / q.shape[1], ys.mean() * P.act / q.shape[0]))
            cen = np.array(cen); r = np.c_[cen - cen[ref - a], np.zeros(len(cen))]
            sm = cadence_smooth(r, [F[i] for i in range(a, e)])
            out[name] = dict(a=a, e=e, ref=ref, box=box, t=sm.round(3).tolist())
            print(f"{name:22s} from matte | move ({r[0, 0]:7.1f},{r[0, 1]:7.1f}) to ({r[-1, 0]:7.1f},{r[-1, 1]:7.1f}) | max step {np.abs(np.diff(r[:, :2], axis=0)).max():5.1f}")
            continue
        if o.get("tracker", "lk") == "lk":     # feature points, checked forward and back, so one bad point or a passing shadow cannot drag the anchor. Translation only: text should not tilt on tracker noise.
            cx, cy, bw, bh = [v / 2 for v in box]; U = lambda k: np.clip(F[k], 0, 255).astype(np.uint8)
            lkp = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))

            def seed(k, x_, y_):
                m_ = np.zeros((SH2, SW2), np.uint8); m_[max(0, int(y_ - bh / 2)):int(y_ + bh / 2), max(0, int(x_ - bw / 2)):int(x_ + bw / 2)] = 255
                return cv2.goodFeaturesToTrack(U(k), 80, .01, 4, mask=m_)
            T = {ref: (0., 0., 0.)}; lost = 0; used = []
            for seq, step in ((range(ref + 1, e), 1), (range(ref - 1, a - 1, -1), -1)):
                x_, y_ = cx, cy; p0 = seed(ref, x_, y_)
                for i in seq:
                    if p0 is None or len(p0) < 6:
                        p0 = seed(i - step, x_, y_)
                    if p0 is None or len(p0) < 6:
                        lost += 1; T[i] = (x_ - cx) * 2, (y_ - cy) * 2, 0.; continue
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(U(i - step), U(i), p0, None, **lkp); pb, sb, _ = cv2.calcOpticalFlowPyrLK(U(i), U(i - step), p1, None, **lkp)
                    good = (st[:, 0] == 1) & (sb[:, 0] == 1) & (np.abs(p0 - pb).reshape(-1, 2).max(1) < 1.)
                    if good.sum() < 6:
                        lost += 1; p0 = None; T[i] = (x_ - cx) * 2, (y_ - cy) * 2, 0.; continue
                    d = np.median((p1 - p0).reshape(-1, 2)[good], 0); x_ += float(d[0]); y_ += float(d[1]); p0 = p1[good].reshape(-1, 1, 2); used.append(int(good.sum()))
                    T[i] = (x_ - cx) * 2, (y_ - cy) * 2, 0.
            r = np.array([T[i] for i in range(a, e)])
            sm = cadence_smooth(r, [F[i] for i in range(a, e)])
            out[name] = dict(a=a, e=e, ref=ref, box=box, t=sm.round(3).tolist())
            print(f"{name:18s} lk lost {lost:3d}/{e - a - 1:3d} points med {int(np.median(used)) if used else 0:3d} | move ({r[0, 0]:7.1f},{r[0, 1]:7.1f}) to ({r[-1, 0]:7.1f},{r[-1, 1]:7.1f}) | max step {np.abs(np.diff(r[:, :2], axis=0)).max():5.1f}")
            continue
        cx, cy, bw, bh = [v / 2 for v in box]
        m = np.zeros((SH2, SW2), np.uint8); m[max(0, int(cy - bh / 2)):int(cy + bh / 2), max(0, int(cx - bw / 2)):int(cx + bw / 2)] = 255
        T = {ref: (0., 0., 0.)}; lost = 0; ccs = []
        for seq in (range(ref + 1, e), range(ref - 1, a - 1, -1)):
            W_ = np.eye(2, 3, dtype=np.float32); last = (0., 0., 0.)
            for i in seq:
                try:
                    cc, N_ = cv2.findTransformECC(F[ref], F[i], W_.copy(), cv2.MOTION_EUCLIDEAN, crit, m, 5)
                except cv2.error:
                    cc, N_ = 0., W_
                if cc >= CC_FLOOR:
                    W_ = N_; pt = W_ @ np.array([cx, cy, 1.])
                    last = (float(pt[0] - cx) * 2, float(pt[1] - cy) * 2, float(np.degrees(np.arctan2(W_[1, 0], W_[0, 0])))); ccs.append(cc)
                else:
                    lost += 1
                T[i] = last
        r = np.array([T[i] for i in range(a, e)])
        sm = cadence_smooth(r, [F[i] for i in range(a, e)])
        out[name] = dict(a=a, e=e, ref=ref, box=box, t=sm.round(3).tolist())
        print(f"{name:22s} lost {lost:3d}/{e - a - 1:3d} cc med {np.median(ccs) if ccs else 0:.3f} | move ({r[0, 0]:7.1f},{r[0, 1]:7.1f}) to ({r[-1, 0]:7.1f},{r[-1, 1]:7.1f}) turn {r[0, 2]:5.1f}..{r[-1, 2]:5.1f} | max step {np.abs(np.diff(r[:, :2], axis=0)).max():5.1f}")
    json.dump(out, open(P.work / "tracks.json", "w"))
    print("Many lost frames or a big max step: use a smaller box on something rigid, or set the place entry's object to null so the text holds still.")
