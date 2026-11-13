"""Optional second opinion for anime: isnet-anime soft mattes, written to work/mattes_other/{a}_{e}.npy. Used to subtract the holes SAM fills (a hair loop), via "subtract_other" in edit.json's sam entry. Its alpha can dip see-through for whole frames, so it should not be the only matte behind text.
usage: ~/.music-edit/venv/bin/python -I matte_isnet.py edit.json SRC:T [SRC:T ...]     (needs setup.sh --isnet)"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

cfg = json.load(open(sys.argv[1])); A = Path(cfg["work"]); OUT = A / "mattes_other"; OUT.mkdir(exist_ok=True)
f_ = cfg["format"]; W, H, TOP, ACT = f_["width"], f_["height"], f_["picture_top"], f_["picture_height"]; SW, SH = W // 2, ACT // 2
clips = json.load(open(A / "plan.json"))["clips"]
want = {(s.split(":")[0], float(s.split(":")[1])) for s in sys.argv[2:]}
sess = ort.InferenceSession(str(Path.home() / ".music-edit/model/isnet-anime.onnx"), providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]); inp = sess.get_inputs()[0].name
MEAN = np.array([.485, .456, .406], np.float32)
starts = {c["a"]: c for c in clips if (c["src"], c["t"]) in want}
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(A / "picture_raw.mp4"), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
i, cur = 0, None
while True:
    b = p.stdout.read(W * H * 3)
    if len(b) < W * H * 3:
        break
    if i in starts:
        cur = (starts[i], [])
    if cur:
        f = np.frombuffer(b, np.uint8).reshape(H, W, 3)[TOP:TOP + ACT]
        x = cv2.cvtColor(cv2.resize(f, (1024, 1024), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN
        m = sess.run(None, {inp: x.transpose(2, 0, 1)[None]})[0][0, 0]; m = (m - m.min()) / (m.max() - m.min() + 1e-6)
        cur[1].append((cv2.resize(m, (SW, SH)) * 255).astype(np.uint8))
        if i == cur[0]["e"] - 1:
            np.save(OUT / f"{cur[0]['a']}_{cur[0]['e']}.npy", np.array(cur[1])); print("wrote", cur[0]["src"], cur[0]["t"]); cur = None
    i += 1
