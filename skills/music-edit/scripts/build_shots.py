"""Stage 1: cut the shots to the song. Writes work/plan.json, work/picture_raw.mp4 and work/mix.wav, plus a rough cut to look at. Real time only: no retiming.
usage: build_shots.py edit.json [--plan-only]"""
import json
import subprocess
import sys

import numpy as np

from lib import Project, grade

P = Project(sys.argv[1]); cfg = P.cfg; W, H, TOP, ACT, FPS = P.w, P.h, P.top, P.act, P.fps
song = cfg["song"]; BEAT, DROP = song["beat"], song["video_drop"]
fr = lambda v: round(v * FPS)
vt = lambda k: DROP + k * BEAT
cuts = {k: [0.] + [q["t"] for q in json.load(open(P.work / f"cuts_{k.lower()}.json"))["cuts"]] + [json.load(open(P.work / f"cuts_{k.lower()}.json"))["end"]] for k in cfg["sources"]}
plan, bad, x = [], [], float(cfg["start_beat"])
for s, t, n in cfg["shots"]:
    a, e = max(0, fr(vt(x))), fr(vt(x + n)); x += n
    lim = next(q for q in cuts[s] if q > t)
    if t + (e - a) / FPS > lim - 1 / FPS:
        bad.append((s, t, round((e - a) / FPS, 2), round(lim, 2)))
    plan.append(dict(src=s, t=t, a=a, e=e))
ov = []
for k in cfg["sources"]:
    r = sorted((c["t"], c["t"] + (c["e"] - c["a"]) / FPS) for c in plan if c["src"] == k)
    ov += [(k, round(p[0], 2), round(q[0], 2)) for p, q in zip(r[:-1], r[1:]) if q[0] < p[1] - .02]
print("shots", len(plan), "end", plan[-1]["e"] / FPS, "s; before the drop", sum(c["a"] < fr(DROP) for c in plan), "after", sum(c["a"] >= fr(DROP) for c in plan))
print("runs across a source cut (fix these):", bad)
print("footage used twice:", ov)
json.dump(dict(fps=FPS, clips=plan), open(P.work / "plan.json", "w"))
if bad:
    sys.exit(1)
if "--plan-only" in sys.argv:
    sys.exit(0)
matrix = {k: np.load(v["grade"]["matrix"]) for k, v in cfg["sources"].items() if v.get("grade")}
enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", str(P.work / "picture_raw.mp4")], stdin=subprocess.PIPE)
blank = np.zeros((H, W, 3), np.uint8); end = plan[-1]["e"]; fade = cfg.get("fade_out_frames", 8)
for c in plan:
    n = c["e"] - c["a"]; s = cfg["sources"][c["src"]]
    o = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{c['t']:.3f}", "-i", s["path"], "-map", "0:v:0", "-frames:v", str(n + 1), "-vf", f"fps={FPS}," + s["vf"], "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    f = np.frombuffer(o, np.uint8).reshape(-1, ACT, W, 3)
    if len(f) > 1 and np.abs(f[0][::8, ::8].astype(int) - f[1][::8, ::8].astype(int)).mean() < .3:   # seeking can repeat the first frame
        f = f[1:]
    assert len(f) >= n, (c, len(f))
    for j, z in enumerate(f[:n]):
        if c["src"] in matrix:
            z = grade(z, matrix[c["src"]], s["grade"].get("chroma", 1.6))
        left = end - (c["a"] + j)
        blank[TOP:TOP + ACT] = (z * (left / fade)).astype(np.uint8) if left < fade else z
        enc.stdin.write(blank.tobytes())
enc.stdin.close(); assert enc.wait() == 0
dur = end / FPS
subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(song["origin"]), "-t", str(dur), "-i", song["path"], "-map", "0:a:0", "-af", f"aresample=44100,alimiter=limit={song.get('limit', .68)}:attack=5:release=50:level=false,afade=t=in:d=0.04,afade=t=out:st={dur - .12}:d=0.12", "-c:a", "pcm_s16le", str(P.work / "mix.wav")], check=True)
rough = P.work / "rough_cut.mp4"
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(P.work / "picture_raw.mp4"), "-i", str(P.work / "mix.wav"), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "25", "-preset", "medium", "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", str(rough)], check=True)
print("rough cut:", rough)
