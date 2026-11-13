"""Shared core for the music-edit tools. Everything reads one edit.json (see references/edit-json.md)."""
import glob
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

SAM_SHA = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
HOME = Path.home() / ".music-edit"


class Project:
    def __init__(self, path):
        self.path = Path(path)
        self.cfg = json.load(open(path))
        f = self.cfg["format"]
        self.w, self.h, self.fps = f["width"], f["height"], f["fps"]
        self.top, self.act = f["picture_top"], f["picture_height"]
        self.sw, self.sh = self.w // 2, self.act // 2     # mattes are stored at half size
        self.work = Path(self.cfg["work"])
        self.hf = Path(self.cfg["project"])
        self.drafts = Path(self.cfg["drafts"])
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / "sheets").mkdir(exist_ok=True)
        self._plan = None

    @property
    def song(self):
        return self.cfg["song"]

    @property
    def plan(self):
        if self._plan is None:
            self._plan = json.load(open(self.work / "plan.json"))
        return self._plan

    @property
    def clips(self):
        return self.plan["clips"]

    @property
    def origin(self):
        return self.song["origin"]

    def clip(self, src, t):
        return next(c for c in self.clips if c["src"] == src and c["t"] == t)

    def clip_at(self, f):
        return next(c for c in self.clips if c["a"] <= f < c["e"])

    def frames(self, path=None, crop=False):
        """Yield (index, frame) from a full-frame video. crop=True returns the picture rows only."""
        path = path or self.work / "picture_raw.mp4"
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
        i, n = 0, self.w * self.h * 3
        try:
            while True:
                b = p.stdout.read(n)
                if len(b) < n:
                    return
                f = np.frombuffer(b, np.uint8).reshape(self.h, self.w, 3)
                yield i, (f[self.top:self.top + self.act] if crop else f)
                i += 1
        finally:
            p.stdout.close()
            p.kill()
            p.wait()

    def encoder(self, out, crf=13):
        return subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.w}x{self.h}", "-r", str(self.fps), "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    # ---- mattes
    def sam_settings(self, c):
        for s in self.cfg.get("sam", []):
            if s["src"] == c["src"] and s["t"] == c["t"]:
                return s
        return {}

    def matte(self, c, kind="sam"):
        """Frames of one shot's matte, half size, uint8. kind 'sam' = SAM 2 with cleanup; 'other' = a second model's soft matte, e.g. matte_isnet.py."""
        name = f"{c['a']}_{c['e']}.npy"
        if kind == "other":
            return np.load(self.work / "mattes_other" / name)
        out = []
        for m in np.load(self.work / "mattes_sam" / name):
            # SAM's own 0 level as a hard edge: a soft edge is what reads as see-through and glitching. Gaps closed, flecks under 2% of the main shape dropped.
            b = cv2.morphologyEx((m > 128).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(b)
            keep = np.zeros_like(b)
            if n > 1:
                big = st[1:, cv2.CC_STAT_AREA].max()
                for q in range(1, n):
                    if st[q, cv2.CC_STAT_AREA] >= .02 * big:
                        keep[lab == q] = 1
            out.append(keep * 255)
        out = np.array(out)
        sub = self.sam_settings(c).get("subtract_other")
        if sub is not None:   # SAM fills enclosed holes (a hair loop); where a second model is sure it is background, take it out
            out[np.load(self.work / "mattes_other" / name) < sub] = 0
        return out


def grade(bgr, matrix, chroma=1.6):
    """Colour map fitted by color_fit.py on drawings present in both sources, held to what those shots can vouch for. Colours absent from the fit (reds, purples) get only a capped chroma gain, because the polynomial is unreliable outside its data. Smooths colour, not detail, so the gain doesn't raise chroma blocks on low-res sources."""
    x = bgr.astype(np.float32) / 255
    b, g, r = x[..., 0], x[..., 1], x[..., 2]
    F = np.stack([np.ones_like(b), b, g, r, b * b, g * g, r * r, b * g, b * r, g * r], -1)
    out = np.clip(F @ matrix.astype(np.float32), 0, 1)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1:] = 128 + (lab[..., 1:] - 128) * chroma
    lab[..., 0] = np.clip((lab[..., 0] - 12) * 1.1, 0, 255)
    keep = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255
    a = lab[..., 1] - 128
    w = np.clip((a - 12) / 25, 0, 1)[..., None]
    o = out * (1 - w) + keep * w
    res = (o * 255).astype(np.uint8)
    lr = cv2.cvtColor(res, cv2.COLOR_BGR2LAB)
    lr[..., 1:] = cv2.GaussianBlur(lr[..., 1:], (0, 0), 3)
    return cv2.cvtColor(lr, cv2.COLOR_LAB2BGR)


def cadence_smooth(r, frames):
    """Smooth a per-frame track over the source's own frames only. 24 fps footage in a 30 fps edit repeats about every 5th frame; smoothing across those makes a pinned thing keep moving while the picture pauses. A repeat: under a quarter of the shot's median frame change. r: array with frames on axis 0."""
    r = np.asarray(r, np.float64); n = len(r); d = np.array([0.] + [float(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)).mean()) for i in range(1, n)])
    med = np.median(d[1:]) if n > 1 else 0.; idx = [0] + [i for i in range(1, n) if d[i] >= .25 * med]
    u = r[idx].reshape(len(idx), -1); pad = np.pad(u, ((2, 2), (0, 0)), mode="edge"); k = np.array([1, 2, 3, 2, 1]) / 9.
    sm = np.stack([np.convolve(pad[:, j], k, "valid") for j in range(u.shape[1])], 1); last = np.searchsorted(idx, np.arange(n), side="right") - 1
    return sm[last].reshape(r.shape)


def node_env():
    """HyperFrames needs node 22+. Use the newest nvm install when the shell's node is missing."""
    env = dict(os.environ)
    found = sorted(glob.glob(str(Path.home() / ".nvm/versions/node/v*/bin")), key=lambda p: [int(x) for x in Path(p).parent.name[1:].split(".")])
    if found:
        env["PATH"] = found[-1] + os.pathsep + env["PATH"]
    return env
