"""Stage 3 picture: transitions, carry-over overlays, aura, blade effect, glow, accents, softening and grain. Writes the HyperFrames project's assets/picture.mp4. Cuts never move here.
Every strength in edit.json "fx" is a starting value for the eye. Defaults are the ones that survived 17 drafts of review.
usage: fx.py edit.json"""
import json
import subprocess
import sys
from collections import Counter

import cv2
import numpy as np

from lib import Project

P = Project(sys.argv[1]); C = P.clips; X = P.cfg.get("fx", {})
W, H, TOP, ACT, FPS = P.w, P.h, P.top, P.act, P.fps
OUT = P.hf / "assets" / "picture.mp4"; OUT.parent.mkdir(parents=True, exist_ok=True)
drop = round(P.song["video_drop"] * FPS)
HITS = {float(k): v for k, v in X.get("hits", {}).items()}      # second: colour split px. The big bar-line moments only: beat-by-beat kicks got rejected
TICKS = X.get("ticks", [])                                      # half-beat accents measured in the song
GRAIN, BLOOM = X.get("grain", 3.5), X.get("bloom", .25)         # .25 was the strongest tested that blew out no sampled frame past 30% white
HALO = X.get("halation", 0)                                     # warm glow around lights on every frame. On the Drive edit, 1.5 added at most 10% near-white on any sampled frame
clip_at = {f: c for c in C for f in range(c["a"], c["e"])}
cuts = [c["a"] for c in C[1:]]; lens = {c["a"]: c["e"] - c["a"] for c in C}; lens_prev = {c["e"]: c["e"] - c["a"] for c in C}
hitf = {round(t * FPS) for t in HITS}
FLOW = {int(k): v for k, v in json.load(open(P.work / "flow.json")).items()}

# Carry-overs: "out" keeps the outgoing subject while the next shot arrives behind it; "in" brings the incoming subject early, over the outgoing picture. The other shot's frames play reversed so it is continuous when it takes over.
carry, need, rep = {}, set(), {}
for cc in X.get("carry", []):
    c = P.clip(cc["src"], cc["t"]); ms = P.matte(c, cc.get("matte", "sam")); n = cc["n"]; f = c["a"] if cc["kind"] == "in" else c["e"]
    use = ms[1:n + 1] if cc["kind"] == "in" else ms[-n:]; ar = (use > 128).mean(axis=(1, 2))
    rep[f"{cc['kind']} {cc['src']}{cc['t']}"] = (round(float(ar.min()) * 100, 1), round(float(ar.max()) * 100, 1))
    carry[f] = (cc["kind"], c, n, ms); need |= set(range(f - n, f + n + 1))
print("carry subject area % min, max:", rep)

# Transitions. Explicit "transitions" {second: kind} wins. Otherwise: build cycle before the drop, push cycle after it, with hard cuts where a shot has text behind its subject (a warped frame cannot be cut out, so the word would pop in front) and around 0.2 s shots (a transition on each end leaves no clean frame).
behind_cuts = set()
for b in P.cfg.get("behind", []):
    c = P.clip(b["src"], b["t"]); behind_cuts |= {c["a"], c["e"]}
trans = {}
explicit = {round(float(k) * FPS): v for k, v in X.get("transitions", {}).items()}
BUILD_T = X.get("build_transitions", ["glitch", "suck", "glitch", "suck", "whip_v", "glitch"]); BATTLE_T = X.get("battle_transitions", ["suck", "spiral", "suck", "suck"])
nb = nd = 0
for f in cuts:
    w = min(3, lens[f] // 3, lens_prev[f] // 3)
    if f in explicit:
        k_ = explicit[f]
        if k_:
            trans[f] = (k_, min(w, 2) if f >= drop else w)
        continue
    if X.get("transitions_explicit_only") or f in carry or f in hitf or f in behind_cuts or w < 2 or f < round(X.get("clean_until", 2.0) * FPS):
        continue
    if f < drop:
        trans[f] = (BUILD_T[nb % len(BUILD_T)], w); nb += 1
    elif lens[f] > 6 and lens_prev[f] > 6:
        trans[f] = (BATTLE_T[nd % len(BATTLE_T)], min(w, 2)); nd += 1
if not X.get("transitions_explicit_only") and not explicit:
    # Side whips only where footage pans: sideways speed = |dx| - |dy|; the largest gap in the sorted speeds separates panning shots, so no threshold is picked. One whip per panning shot, on its faster side, travelling its way.
    sides = []
    for f in trans:
        if f >= drop:
            for name, shot in (("out", next(c for c in C if c["e"] == f)), ("inc", next(c for c in C if c["a"] == f))):
                v = FLOW[f][name]; sides.append((abs(v["dx"]) - abs(v["dy"]), f, v["dx"], shot["src"], shot["t"]))
    if len(sides) > 1:
        sides.sort(reverse=True); cutoff = int(np.argmax([sides[i][0] - sides[i + 1][0] for i in range(len(sides) - 1)])); best = {}
        for sp, f, dx, s_, t_ in sides[:cutoff + 1]:
            if sp > 0:
                best.setdefault((s_, t_), (f, dx))
        for f, dx in best.values():
            trans[f] = ("whip_r" if dx > 0 else "whip_l", trans[f][1])
mont = set(range(drop, C[-1]["e"])); hit_ = {round(t * FPS) + k for t in HITS for k in range(3)}
touched = {f + o for f, (_, w) in trans.items() for o in range(-w, w)} | {i for f, (k, c, n, m) in carry.items() for i in range(f - n, f)}
print("after the drop: frames", len(mont), "clean", len(mont - touched - hit_), round(len(mont - touched - hit_) / max(1, len(mont)) * 100), "%   (carry-overs count as touched, though the fight stays visible)")
print("transitions", len(trans), "of", len(cuts), "cuts", dict(Counter(v[0] for v in trans.values())))
for f, (_, w) in trans.items():
    need |= set(range(f - w, f + w))

# Softening: two sources meet halfway in measured detail, so the sharp shots stay sharper. Only when "soften_src" is set.
store, lap = {}, {}
sharp = X.get("soften_src"); sig = 0.
for i, f in P.frames(crop=True):
    if i in need:
        store[i] = f.copy()
    if sharp and i % 5 == 0 and i < C[-1]["e"] - 10:
        lap.setdefault(clip_at[i]["src"], []).append(float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()))
if sharp and len(lap) == 2:
    lt = float(np.median(lap[sharp])); lg = float(np.median([v for k, v in lap.items() if k != sharp][0])); target = float(np.sqrt(lt * lg))
    sample = next(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for i, f in P.frames(crop=True) if clip_at[i]["src"] == sharp and i > 30)
    sig = .3
    for s in np.arange(.3, 3.01, .1):
        sig = float(s)
        if cv2.Laplacian(cv2.GaussianBlur(sample, (0, 0), s), cv2.CV_32F).var() * (lt / cv2.Laplacian(sample, cv2.CV_32F).var()) <= target:
            break
    print("detail", sharp, round(lt), "other", round(lg), "target", round(target), "softening sigma", round(sig, 1))


def blend(fg, bg, al):
    sh = np.clip(cv2.GaussianBlur(al, (0, 0), 18) * .45, 0, 1)
    b = bg.astype(np.float32) * (1 - sh)[:, :, None]
    return (fg.astype(np.float32) * al[:, :, None] + b * (1 - al[:, :, None])).clip(0, 255).astype(np.uint8)


def preroll(shot, n):
    """The n source frames just before a shot's in-point, in playing order, so an "out" carry's background runs forward straight into the shot. None if a source cut is in the way."""
    s = P.cfg["sources"][shot["src"]]; t0 = shot["t"] - n / FPS
    raw = json.load(open(P.work / f"cuts_{shot['src'].lower()}.json"))["cuts"]
    if t0 < 0 or any(t0 <= q["t"] <= shot["t"] for q in raw):
        return None
    o = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", s["path"], "-map", "0:v:0", "-frames:v", str(n + 1), "-vf", f"fps={FPS}," + s["vf"], "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(o, np.uint8).reshape(-1, ACT, W, 3)
    if len(fr) > 1 and np.abs(fr[0][::8, ::8].astype(int) - fr[1][::8, ::8].astype(int)).mean() < .3:   # seeking can repeat the first frame
        fr = fr[1:]
    return fr[:n] if len(fr) >= n else None


comp = {}
CARRY_M = {}     # the cut-out carried in over each frame: words baked on a surface sit behind it
for f, (kind, c, n, ms) in carry.items():
    pre = preroll(next(x for x in C if x["a"] == f), n)      # both kinds: the arriving shot's own frames from just before its in-point, in playing order
    pm = P.work / "mattes_sam" / f"pre_{c['a']}_{n}.npy"; pmask = np.load(pm) if kind == "in" and pm.exists() else None      # matte_pre.py
    forward = pre is not None and (kind == "out" or pmask is not None)
    print(kind, "carry at", f, "plays forward" if forward else "REVERSED: move the in-point later and run matte_pre.py")
    for i in range(f - n, f):
        b = f + (f - i); q = i - (f - n)
        if kind == "out":
            comp[i] = blend(store[i], pre[q] if pre is not None else store[b], cv2.GaussianBlur(cv2.resize(ms[i - c["a"]], (W, ACT)).astype(np.float32) / 255, (0, 0), 1.1))
        elif forward:
            CARRY_M[i] = pmask[q]; comp[i] = blend(pre[q], store[i], cv2.GaussianBlur(cv2.resize((pmask[q] > 128).astype(np.uint8) * 255, (W, ACT)).astype(np.float32) / 255, (0, 0), 1.1))
        else:
            comp[i] = blend(store[b], store[i], cv2.GaussianBlur(cv2.resize(ms[b - c["a"]], (W, ACT)).astype(np.float32) / 255, (0, 0), 1.1))


def warp(im, M):
    return cv2.warpAffine(im, M, (W, ACT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def radial(im, s0, s1, a0=0., a1=0., n=6, c=(W / 2, ACT / 2)):
    acc = np.zeros_like(im, dtype=np.float32)
    for q in np.linspace(0, 1, n):
        acc += warp(im, cv2.getRotationMatrix2D(c, a0 + (a1 - a0) * q, s0 + (s1 - s0) * q))
    return (acc / n).astype(np.uint8)


PUSH, TRAIL, FEATHER, BACK = X.get("push", .55), .12, .4, .55     # zoom at the cut, radial streak, soft edge on the new shot, brightness of the old shot behind it
_y, _x = np.mgrid[0:ACT, 0:W].astype(np.float32)
EDGE = np.minimum.reduce([_x, W - 1 - _x, _y, ACT - 1 - _y])


def max_angle(s):
    """Largest rotation about the frame centre at zoom s that shows no edge of the picture (geometry of the rotated rectangle)."""
    best_ = 0.
    for d in np.arange(0, 30, .25):
        r = np.radians(d)
        if (W / 2 * np.cos(r) + ACT / 2 * np.sin(r)) <= s * W / 2 and (W / 2 * np.sin(r) + ACT / 2 * np.cos(r)) <= s * ACT / 2:
            best_ = float(d)
    return best_


def push(kind, f, im, o, w):
    """"Suck you in": zoom speed mirrors around the cut, so the push never reverses, and the new shot blooms out of the old one at the point the eye was on, then settles. Needs no mirrored edges."""
    k = 1 - (abs(o + .5) - .5) / w
    spiral = kind == "spiral"; centre = (W / 2, ACT / 2)
    Fo = centre if spiral else (FLOW[f]["out"]["fx"], FLOW[f]["out"]["fy"]); Fi = centre if spiral else (FLOW[f]["inc"]["fx"], FLOW[f]["inc"]["fy"])
    s_ = 1 + PUSH * k * k; ang = max_angle(s_) if spiral else 0.
    if o < 0:
        return radial(im, s_, s_ * (1 + TRAIL * k), ang * .7, ang, n=16, c=Fo)
    peak = 1 + PUSH
    back = radial(store[f - 1], peak * (1 + TRAIL * (o + 1)), peak * (1 + TRAIL * (o + 2)), max_angle(peak), max_angle(peak), c=Fo)
    back = (back.astype(np.float32) * BACK).astype(np.uint8)
    pos = (Fi[0] + (Fo[0] - Fi[0]) * k * k, Fi[1] + (Fo[1] - Fi[1]) * k * k)
    M = cv2.getRotationMatrix2D(Fi, -ang, 1 / s_); M[0, 2] += pos[0] - Fi[0]; M[1, 2] += pos[1] - Fi[1]
    fg = cv2.warpAffine(im, M, (W, ACT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    soft = np.clip(EDGE / max(1., FEATHER * ACT * k * k), 0, 1)
    al = cv2.warpAffine(soft * soft * (3 - 2 * soft), M, (W, ACT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None]
    return radial((fg.astype(np.float32) * al + back.astype(np.float32) * (1 - al)).astype(np.uint8), 1., 1 + TRAIL * k, n=16, c=pos)


def glitch(im, k, seed):
    r = np.random.default_rng(seed); out = im.copy(); y = 0
    while y < ACT:
        h = int(r.integers(14, 90))
        if r.random() < .6:
            out[y:y + h] = np.roll(im[y:y + h], int(r.normal(0, .14) * W * k), axis=1)
        y += h
    px = int(34 * k); out[..., 2] = np.roll(out[..., 2], -px, axis=1); out[..., 0] = np.roll(out[..., 0], px, axis=1)
    return out


def make(kind, im, o, w, seed):
    k = 1 - (abs(o + .5) - .5) / w; side = -1 if o < 0 else 1
    if kind in ("whip_r", "whip_l", "whip_v"):
        d = 1 if kind == "whip_r" else -1; vert = kind == "whip_v"
        size = max(3, int(.3 * (ACT if vert else W) * k)) | 1
        x = cv2.filter2D(im, -1, np.ones((size, 1) if vert else (1, size), np.float32) / size, borderType=cv2.BORDER_REFLECT)
        off = -side * d * .42 * k * k * (ACT if vert else W)
        return warp(x, np.float32([[1, 0, 0 if vert else off], [0, 1, off if vert else 0]]))
    return glitch(im, k, seed)


for f, (kind, w) in trans.items():
    for o in range(-w, w):
        if f + o not in comp:
            comp[f + o] = push(kind, f, store[f + o], o, w) if kind in ("suck", "spiral") else make(kind, store[f + o], o, w, f + o)
split = {}
for t, amp in HITS.items():
    f = round(t * FPS)
    for k in range(7):
        split[f + k] = round(amp * (1 if k < 2 else .6 if k < 4 else .3))
for t in TICKS:
    f = round(t * FPS)
    for k in range(2):
        split.setdefault(f + k, 4)
rng = np.random.default_rng(9)

# Object effect: a blade (any long lit object) glows, its marks burn and a glint runs along it on the beat. Geometry is measured on the object's ref frame and carried by its track.
BL = X.get("blade")
if BL:
    BS = P.clip(BL["src"], BL["t"]); TR = json.load(open(P.work / "tracks.json"))[BL["object"]]; OB = P.cfg["objects"][BL["object"]]["box"]
    S0, S1 = BS["a"] + 3, BS["e"] - 3
    GUARD, HW, BLEN = np.float32(BL["base"]), float(BL["half_width"]), float(BL["length"]); ANG = np.radians(BL["angle"])
    _T = ((_x - GUARD[0]) * -np.cos(ANG) + (_y - GUARD[1]) * np.sin(ANG)) / BLEN
    _V = (_x - GUARD[0]) * -np.sin(ANG) + (_y - GUARD[1]) * -np.cos(ANG)
    BEATF = round(P.song["beat"] * FPS); SWEEPS = [(round(t * FPS), a) for t, a in BL["sweeps"]]
    GLOW, IDLE, MIST, DIM, TAU = np.float32(BL.get("glow", [255, 215, 90])), BL.get("idle", .35), BL.get("mist", .85), BL.get("dim", .8), BL.get("tau", 14.)
    _m = np.random.default_rng(16)

    def _fbm(h, w):
        n = np.zeros((h, w), np.float32)     # long along the blade, so it reads as wisps instead of specks
        for (sy, sx), amp in (((110, 420), 1.), ((60, 220), .5), ((30, 110), .2)):
            n += amp * cv2.resize(_m.normal(0, 1, (h // sy + 3, w // sx + 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        return n
    _rot = cv2.getRotationMatrix2D((W / 2 + 150, ACT / 2 + 150), -BL["angle"], 1.)
    _A, _B = [cv2.warpAffine(_fbm(ACT + 300, W + 300), _rot, (W + 300, ACT + 300), borderMode=cv2.BORDER_REFLECT) for _ in "ab"]
    _lo, _hi = np.percentile(_A + _B, 45), np.percentile(_A + _B, 99)

    def blade_fx(a, i):
        d = TR["t"][i - TR["a"]]; M = cv2.getRotationMatrix2D((float(OB[0]), float(OB[1])), -d[2], 1.); M[0, 2] += d[0]; M[1, 2] += d[1]
        war = lambda x: cv2.warpAffine(x, M, (W, ACT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        T, V = war(_T), war(_V); blade = np.clip(T / .04, 0, 1) * np.clip((HW - np.abs(V)) / 10, 0, 1)
        g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        K = cv2.GaussianBlur(np.clip((cv2.GaussianBlur(g, (0, 0), 16) - g) / 26, 0, 1) * blade * np.clip((HW - 22 - np.abs(V)) / 8, 0, 1), (0, 0), 1.)
        lit, glint = np.zeros((ACT, W), np.float32) + IDLE, np.zeros((ACT, W), np.float32)
        for fs, amp in SWEEPS:
            front = (i - fs) / BEATF; since = (front - T) * BEATF
            lit += amp * np.where(since >= 0, np.exp(-np.clip(since, 0, None) / TAU), 0)
            glint += amp * np.exp(-((T - front) / .035) ** 2) * blade * float(-.2 < front < 1.2)
        lit = np.clip(lit, 0, 1.2); fade = min(1., (i - S0 + 1) / 4, (S1 - i) / 4)
        # A nearly white blade would swallow added light, so pull the blade down and let the marks burn on it.
        x = a.astype(np.float32) * (1 - (blade * DIM * np.clip(lit, 0, 1) * fade)[..., None] * np.float32([.25, .75, 1.]))
        x += (K * lit * fade)[..., None] * np.float32([255, 250, 225]) * 1.5
        x += (cv2.GaussianBlur(K * lit * fade, (0, 0), 9) * 1.5)[..., None] * GLOW
        x += cv2.GaussianBlur(glint * .8, (0, 0), 2.5)[..., None] * np.float32([255, 245, 225])
        k = i - S0
        n = _A[150 - min(150, k * 3):, 150 + min(150, k * 4):][:ACT, :W] + _B[150 - min(150, k * 2):, 150 - min(150, k * 2):][:ACT, :W]
        mist = np.clip((n - _lo) / (_hi - _lo), 0, 1) ** 1.3
        out = np.clip(np.abs(V) - HW, 0, None); fall = np.exp(-out / 95.) * np.clip(out / 12, .0, 1) * np.clip(T / .12, 0, 1)
        x += (mist * fall * MIST * fade)[..., None] * np.float32([255, 228, 190])
        return np.clip(x, 0, 255).astype(np.uint8)
else:
    S0 = S1 = -1

AURA = {P.clip(a["src"], a["t"])["a"]: a for a in X.get("aura", [])}; aura_cache = {}     # glow around a subject, only on shots whose cutout holds steady
INVERT = {round(t * FPS) + k for t in X.get("strobe", []) for k in (0, 2, 4)}              # rapid flashes: warn before posting
KICK = {round(t * FPS) + k: amp for t in HITS for k, amp in enumerate((.14, .09, .05))}


from wall_lib import composite as wall_composite
WALL = []     # words placed on a surface by wall_text.py
for w_ in X.get("wall_text", []):
    z = np.load(P.work / "walls" / f"{w_['name']}.npz"); c_w = P.clip(w_["src"], w_["t"]); WALL.append(dict(a=int(z["a"]), C=z["corners"], al=z["alpha"], S=z["shadow"] if "shadow" in z else None, c=c_w, ms=P.matte(c_w, "sam") if w_.get("matte", True) else np.zeros((c_w["e"] - c_w["a"], ACT // 2, W // 2), np.uint8), cfg=w_))
enc = P.encoder(OUT)
for i, f in P.frames():
    a = comp[i].copy() if i in comp else f[TOP:TOP + ACT].copy()
    c_ = clip_at[i]
    if c_["src"] == sharp and sig and i not in comp:
        a = cv2.GaussianBlur(a, (0, 0), sig)
    if S0 <= i < S1 and i not in comp:
        a = blade_fx(a, i)
    if c_["a"] in AURA and i not in comp:
        if c_["a"] not in aura_cache:
            aura_cache.clear(); aura_cache[c_["a"]] = P.matte(c_, AURA[c_["a"]].get("matte", "sam"))
        m_ = cv2.resize(aura_cache[c_["a"]][i - c_["a"]], (W, ACT)).astype(np.float32) / 255
        edge = np.clip(cv2.GaussianBlur(m_, (0, 0), 16) * 1.6 - m_ * 1.2, 0, 1)
        a = np.clip(a.astype(np.float32) + edge[..., None] * np.array(AURA[c_["a"]].get("color", [255, 190, 70]), np.float32) * .7, 0, 255).astype(np.uint8)
    if i >= drop and BLOOM:
        sm_ = cv2.resize(np.clip(a.astype(np.float32) - 175, 0, None), (W // 4, ACT // 4), interpolation=cv2.INTER_AREA)
        a = np.clip(a.astype(np.float32) + cv2.resize(cv2.GaussianBlur(sm_, (0, 0), 12), (W, ACT)) * BLOOM, 0, 255).astype(np.uint8)
    if HALO:
        g_ = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (W // 4, ACT // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
        h_ = np.clip(g_ - cv2.GaussianBlur(g_, (0, 0), 20) - 40, 0, None) / 215 * np.clip((g_ - 150) / 50, 0, 1)     # brighter than its surroundings, so lamps glow and a bright sky does not
        b_ = cv2.resize(cv2.GaussianBlur(h_, (0, 0), 5) * .8 + cv2.GaussianBlur(h_, (0, 0), 18) * 1.4, (W, ACT))
        a = np.clip(a.astype(np.float32) + b_[..., None] * np.float32([.25, .55, 1.]) * 255 * HALO, 0, 255).astype(np.uint8)     # warm: film halation bleeds toward red
    for w_ in WALL:
        q = i - w_["a"]
        if 0 <= q < len(w_["C"]):
            m_w = w_["ms"][i - w_["c"]["a"]]; m_w = np.maximum(m_w, cv2.resize(CARRY_M[i], m_w.shape[::-1])) if i in CARRY_M else m_w
            a = wall_composite(a, w_["cfg"], w_["C"][q], w_["al"], i, m_w, None if w_["S"] is None else w_["S"][q])
    if i in KICK and i not in comp:
        a = radial(a, 1.0, 1.0 + KICK[i], n=6)
    if i in INVERT:
        a = 255 - a
    px = split.get(i)
    if px:
        a[..., 2] = np.roll(a[..., 2], -px, axis=1); a[..., 0] = np.roll(a[..., 0], px, axis=1)
    n_ = cv2.resize(rng.normal(0, GRAIN, (ACT // 2, W // 2)).astype(np.float32), (W, ACT), interpolation=cv2.INTER_LINEAR)[..., None]
    g = f.copy(); g[TOP:TOP + ACT] = np.clip(a.astype(np.float32) + n_, 0, 255).astype(np.uint8)
    enc.stdin.write(g.tobytes())
enc.stdin.close(); assert enc.wait() == 0
print("picture written:", OUT)
