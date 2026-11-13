"""Cut out the frames just BEFORE a shot, so an "in" carry-over plays its subject forward into the cut instead of reversed.
For each fx.carry entry of kind "in": takes n source frames before the in-point plus a few after, seeds SAM 2 with the box of the shot's own matte on its first frame, and propagates backward. Writes work/mattes_sam/pre_{a}_{n}.npy (frames in playing order) and work/sheets/mattes_pre.png.
usage: ~/.music-edit/venv/bin/python -I matte_pre.py edit.json   (run from the scripts folder, after matte_sam.py)"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

cfg = json.load(open(sys.argv[1])); A = Path(cfg["work"]); OUT = A / "mattes_sam"
HOME = Path.home() / ".music-edit"; ck = HOME / "model/sam2.1_hiera_tiny.pt"
assert hashlib.sha256(ck.read_bytes()).hexdigest() == "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69", "SAM 2 model checksum mismatch"
clips = json.load(open(A / "plan.json"))["clips"]; f_ = cfg["format"]; FPS = f_["fps"]
W, ACT, SW, SH = f_["width"], f_["picture_height"], f_["width"] // 2, f_["picture_height"] // 2
device = "mps" if torch.backends.mps.is_available() else "cpu"
pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", str(ck), device=device)
tmp = A / "sam_tmp"; cells = []; AFTER = 10
for cc in cfg.get("fx", {}).get("carry", []):
    if cc["kind"] != "in":
        continue
    c = next(x for x in clips if x["src"] == cc["src"] and x["t"] == cc["t"]); n = cc["n"]; s = cfg["sources"][c["src"]]; t0 = c["t"] - n / FPS
    cuts = json.load(open(A / f"cuts_{c['src'].lower()}.json"))["cuts"]
    assert t0 >= 0 and not any(t0 <= q["t"] <= c["t"] for q in cuts), ("a source cut sits inside the pre-roll; move the in-point later", cc)
    m0 = np.load(OUT / f"{c['a']}_{c['e']}.npy")[0] > 128; ys, xs = np.where(m0); box = np.array([xs.min() * 2, ys.min() * 2, xs.max() * 2, ys.max() * 2], np.float32)
    o = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", s["path"], "-map", "0:v:0", "-frames:v", str(n + AFTER + 1), "-vf", f"fps={FPS}," + s["vf"], "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    F = np.frombuffer(o, np.uint8).reshape(-1, ACT, W, 3)
    if len(F) > 1 and np.abs(F[0][::8, ::8].astype(int) - F[1][::8, ::8].astype(int)).mean() < .3:      # seeking can repeat the first frame
        F = F[1:]
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    for i, f in enumerate(F):
        cv2.imwrite(str(tmp / f"{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 98])
    with torch.inference_mode():
        st = pred.init_state(video_path=str(tmp)); pred.add_new_points_or_box(st, frame_idx=n, obj_id=1, box=box); L = {}
        for i, _, lg in pred.propagate_in_video(st, start_frame_idx=n, reverse=True):
            L[i] = lg[0, 0].float().cpu().numpy()
    ms = np.array([(cv2.resize(cv2.GaussianBlur(np.clip(L[i] * 2 + .5, 0, 1), (0, 0), .8), (SW, SH)) * 255).astype(np.uint8) for i in range(n)])
    np.save(OUT / f"pre_{c['a']}_{n}.npy", ms); ar = (ms > 128).mean(axis=(1, 2)) * 100
    print(f"{c['src']}{c['t']} pre {n} frames, area {ar.min():.1f}-{ar.max():.1f}%, first shot-frame area {m0.mean() * 100:.1f}%", flush=True)
    for j in (0, n // 2, n - 1):
        al = cv2.resize(ms[j], (W, ACT)).astype(np.float32)[..., None] / 255
        x = cv2.resize((F[j] * al + np.array([200, 0, 255], np.float32) * (1 - al)).astype(np.uint8), (480, 204)); cv2.putText(x, f"{c['t']} pre {j}", (4, 16), 0, .45, (0, 255, 0), 1); cells.append(x)
    pred.reset_state(st)
shutil.rmtree(tmp, ignore_errors=True)
if cells:
    cv2.imwrite(str(A / "sheets/mattes_pre.png"), np.vstack([np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)]))
