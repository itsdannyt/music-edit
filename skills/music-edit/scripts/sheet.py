"""Look at frames. usage: sheet.py VIDEO OUT.png FRAMES COLS WIDTH [--grid --top 132 --act 816]
FRAMES is comma separated, or a:b:step. --grid draws picture coordinates, for choosing boxes and text spots."""
import subprocess
import sys

import cv2
import numpy as np

a = sys.argv; v, out, spec, cols, cw = a[1], a[2], a[3], int(a[4]), int(a[5])
grid = "--grid" in a; top = int(a[a.index("--top") + 1]) if "--top" in a else 0; act = int(a[a.index("--act") + 1]) if "--act" in a else None
if ":" in spec:
    s, e, st = [int(x) for x in spec.split(":")]; want = list(range(s, e, st))
else:
    want = [int(x) for x in spec.split(",")]
W, H = [int(x) for x in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", v], capture_output=True, text=True).stdout.strip().split(",")[:2]]
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", v, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE); got = {}; i = 0
while i <= max(want):
    b = p.stdout.read(W * H * 3)
    if len(b) < W * H * 3:
        break
    if i in want:
        got[i] = np.frombuffer(b, np.uint8).reshape(H, W, 3)[top:top + (act or H)].copy()
    i += 1
p.kill(); cells = []
for f in want:
    if f not in got:
        continue
    x = got[f]
    if grid:
        for gx in range(0, x.shape[1], 240):
            cv2.line(x, (gx, 0), (gx, x.shape[0]), (0, 255, 255), 1); cv2.putText(x, str(gx), (gx + 4, 30), 0, 1, (0, 255, 255), 2)
        for gy in range(0, x.shape[0], 204):
            cv2.line(x, (0, gy), (x.shape[1], gy), (0, 255, 255), 1); cv2.putText(x, str(gy), (4, gy + 30), 0, 1, (0, 255, 255), 2)
    cv2.putText(x, f"f{f} {f / 30:.2f}s", (x.shape[1] - 460, x.shape[0] - 20), 0, 1.4, (0, 255, 0), 3)
    cells.append(cv2.resize(x, (cw, int(cw * x.shape[0] / x.shape[1])), interpolation=cv2.INTER_AREA))
while len(cells) % cols:
    cells.append(np.zeros_like(cells[0]))
cv2.imwrite(out, np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]))
print(out)
