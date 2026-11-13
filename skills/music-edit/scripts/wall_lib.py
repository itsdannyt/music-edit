"""Draws words placed by wall_text.py into a frame. Shared by wall_text.py (preview) and fx.py (final picture).
Three kinds of word: invert (default), paint (flat colour lit by the surface it is on), stand (solid letters standing on the ground: rise from their base, have a back edge toward the camera's vanishing point, cast a shadow, and sit behind the subject's matte)."""
import cv2
import numpy as np


def warp(alpha, corners, size):
    h, w = alpha.shape
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(corners))
    return cv2.warpPerspective(alpha.astype(np.float32) / 255, M, size, flags=cv2.INTER_LINEAR)


def glitch_state(w, i, lit):
    """Same timing as the on-screen text's glitch: 0.17 s in four steps (5 frames at 30 fps), run backward to leave. Returns opacity, slice shift and colour split in px."""
    n, ins = w.get("glitch_frames", 5), w.get("glitch_px", 26)
    p = min(1., (i - lit + 1) / n if w.get("glitch_in", True) else 1., (w["to"] - i) / n if w.get("glitch_out", True) else 1.)
    if not w.get("glitch_in", True) and i < lit:
        p = 0.
    return (0., 0., 0.) if p <= 0 else (p, (1 - p) * ins * 2, (1 - p) * ins)


def slices(al, i, shift):
    """Slide horizontal bands of the word sideways by a frame-seeded amount, so a render is repeatable."""
    if shift < .5:
        return al
    rng = np.random.default_rng(i); out = al.copy(); ys = np.where(al.max(1) > 0)[0]
    if not len(ys):
        return al
    cuts = np.linspace(ys[0], ys[-1] + 1, 8).astype(int)
    for y0, y1 in zip(cuts[:-1], cuts[1:]):
        out[y0:y1] = np.roll(al[y0:y1], int(rng.uniform(-shift, shift)), axis=1)
    return out


def composite(a, w, corners, alpha, i, matte, shadow=None):
    ACT, W = a.shape[:2]; size = (W, ACT); out = a.astype(np.float32)
    occ = 1 - cv2.GaussianBlur(cv2.resize(matte, size).astype(np.float32) / 255, (0, 0), 1.1)
    lits = w.get("lit", [w["from"]]); ramp = lambda j, n: min(1., max(0., (i - lits[j] + 1) / n))
    if "stand" in w:
        s = w["stand"]; n = s.get("rise_frames", 3); ease = lambda t: 1 - (1 - t) ** 3; live = [j for j in range(len(alpha)) if ramp(j, n) > 0]
        base = corners[[3, 2]]
        if shadow is not None and live:
            m = np.zeros((ACT, W), np.float32)
            for j in live:
                r = ease(ramp(j, n)); sb = shadow[[3, 2]]; tips = sb + (shadow[[0, 1]] - sb) * r
                m = np.maximum(m, warp(alpha[j], [tips[0], tips[1], sb[1], sb[0]], size))
            m = cv2.GaussianBlur(m, (0, 0), s.get("shadow_blur", 1.2)) * occ
            out *= 1 - m[..., None] * (1 - np.float32(s["shadow_rgb_ratio"]))
        vp = np.float32(s["vp"])
        for j in live:
            r = ease(ramp(j, n)); top = base + (corners[[0, 1]] - base) * r; q = np.float32([top[0], top[1], base[1], base[0]]); col = np.float32(w["paint"][j])
            fade = glitch_state(w, i, lits[j])[0] if w.get("ent") == "glitch" else 1.
            for t in np.linspace(s["depth"], 0, s.get("depth_steps", 6), endpoint=False):
                al = warp(alpha[j], q + (vp - q) * t, size)[..., None] * occ[..., None] * fade
                out = out * (1 - al) + col * s["edge"] * al
            g_ = glitch_state(w, i, lits[j]) if w.get("ent") == "glitch" else None
            face = warp(alpha[j], q, size)
            if g_:
                face = slices(face, i, g_[1]) * g_[0]
            al = cv2.GaussianBlur(face, (0, 0), .6)[..., None] * occ[..., None]
            out = out * (1 - al) + col * al
        return np.clip(out, 0, 255).astype(np.uint8)
    for j in range(len(alpha)):
        glitch = glitch_state(w, i, lits[j]) if w.get("ent") == "glitch" else None
        dim = w.get("dim", 0.); k = (glitch[0] if glitch else dim + (1 - dim) * ramp(j, 2))
        if k <= 0:
            continue
        raw = warp(alpha[j], corners, size)
        if glitch:
            raw = slices(raw, i, glitch[1]); chan = [np.roll(raw, int(round(glitch[2] * d)), axis=1) * occ * k for d in (1, 0, -1)]     # BGR split: blue one way, red the other
        else:
            chan = [raw * occ * k] * 3
        al = chan[1]
        if "paint" in w:
            g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32); light = np.clip(g / max(1., float(np.median(g[al > .5])) if (al > .5).any() else 1.), .65, 1.2)
            top = np.float32(w["paint"][j]) * light[..., None]
        else:
            top = 255. - a.astype(np.float32)
        top = np.clip(top, 0, 255)
        for c_, al_ in enumerate(chan):
            out[..., c_] = out[..., c_] * (1 - al_) + top[..., c_] * al_
    return out.astype(np.uint8)
