"""Cut subjects out with SAM 2: a box on one frame picks the subject and SAM 2 follows it to both ends of the shot. Writes work/mattes_sam/{a}_{e}.npy and a magenta check sheet to look at before using any cutout.
Flicker = matte change the picture's own motion cannot explain, as a share of the matte.
usage: ~/.music-edit/venv/bin/python -I matte_sam.py edit.json [SHOT_FILTER e.g. T:8.25]
Uses edit.json "sam": [{"src","t","ref","box":[x0,y0,x1,y1],"name"}]. ref is a video frame in that shot; the box is in picture coordinates on that frame."""
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

cfg = json.load(open(sys.argv[1])); A = Path(cfg["work"]); OUT = A / "mattes_sam"; OUT.mkdir(exist_ok=True); (A / "sheets").mkdir(exist_ok=True)
HOME = Path.home() / ".music-edit"; ck = HOME / "model/sam2.1_hiera_tiny.pt"
assert hashlib.sha256(ck.read_bytes()).hexdigest() == "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69", "SAM 2 model checksum mismatch"
clips = json.load(open(A / "plan.json"))["clips"]; f_ = cfg["format"]
W, H, TOP, ACT, SW, SH = f_["width"], f_["height"], f_["picture_top"], f_["picture_height"], f_["width"] // 2, f_["picture_height"] // 2
want = sys.argv[2] if len(sys.argv) > 2 else None
shots = [s for s in cfg.get("sam", []) if not want or want == f"{s['src']}:{s['t']}"]
device = "mps" if torch.backends.mps.is_available() else "cpu"
pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", str(ck), device=device)
yy, xx = np.mgrid[0:SH // 2, 0:SW // 2].astype(np.float32); cells = []; tmp = A / "sam_tmp"; rep = []
for s in shots:
    c = next(x for x in clips if x["src"] == s["src"] and x["t"] == s["t"]); a, e = c["a"], c["e"]
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(A / "picture_raw.mp4"), "-vf", f"select='between(n,{a},{e - 1})',crop={W}:{ACT}:0:{TOP}", "-fps_mode", "passthrough", "-q:v", "1", "-start_number", "0", str(tmp / "%05d.jpg")], check=True)
    n = e - a; F = [cv2.imread(str(tmp / f"{i:05d}.jpg")) for i in range(n)]
    with torch.inference_mode():
        st = pred.init_state(video_path=str(tmp))
        pred.add_new_points_or_box(st, frame_idx=s["ref"] - a, obj_id=1, box=np.array(s["box"], np.float32))
        L = {}
        for rev in (False, True):
            for i, _, lg in pred.propagate_in_video(st, start_frame_idx=s["ref"] - a, reverse=rev):
                L[i] = lg[0, 0].float().cpu().numpy()
    ms = np.array([(cv2.resize(cv2.GaussianBlur(np.clip(L[i] * 2 + .5, 0, 1), (0, 0), .8), (SW, SH)) * 255).astype(np.uint8) for i in range(n)])
    np.save(OUT / f"{a}_{e}.npy", ms)
    gs = [cv2.cvtColor(cv2.resize(f, (SW // 2, SH // 2)), cv2.COLOR_BGR2GRAY) for f in F]; ch = []
    for i in range(1, n):
        fl = cv2.calcOpticalFlowFarneback(gs[i], gs[i - 1], None, .5, 3, 21, 3, 5, 1.2, 0)
        w_ = cv2.remap(cv2.resize(ms[i - 1], (SW // 2, SH // 2)), xx + fl[..., 0], yy + fl[..., 1], cv2.INTER_LINEAR) > 128
        now = cv2.resize(ms[i], (SW // 2, SH // 2)) > 128
        ch.append(float((w_ ^ now).sum() / max(1, now.sum())))
    ar = (ms > 128).mean(axis=(1, 2))
    r = dict(src=c["src"], t=c["t"], a=a, e=e, name=s.get("name", ""), area=round(float(np.median(ar)) * 100, 1), amin=round(float(ar.min()) * 100, 1), amax=round(float(ar.max()) * 100, 1), flick=round(float(np.median(ch)) * 100, 1) if ch else 0, fmax=round(float(max(ch)) * 100, 1) if ch else 0)
    rep.append(r); print(r, flush=True)
    for j in (0, n // 2, n - 1):
        al = cv2.resize(ms[j], (W, ACT)).astype(np.float32)[..., None] / 255
        x = cv2.resize((F[j] * al + np.array([200, 0, 255], np.float32) * (1 - al)).astype(np.uint8), (480, 204))
        cv2.putText(x, f"{c['src']}{c['t']} f{a + j}", (4, 16), 0, .45, (0, 255, 0), 1); cells.append(x)
    pred.reset_state(st)
shutil.rmtree(tmp, ignore_errors=True)
json.dump(rep, open(A / "matte_report.json", "w"))
if cells:
    cv2.imwrite(str(A / "sheets/mattes.png"), np.vstack([np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)]))
    print("LOOK at", A / "sheets/mattes.png", "- a good score does not prove a clean cutout")
