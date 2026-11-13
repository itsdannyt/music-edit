"""Stage 3 text: writes the cut-out layers that sit above the text, then the HyperFrames composition. Each block pins to a tracked object per shot and follows the slow push, but never rides the bar-line zooms. Cut-out pixels come from the finished picture, so they match it exactly.
usage: text.py edit.json     (run after fx.py, because the layers are cut from its picture)"""
import html
import json
import subprocess
import sys

import cv2
import librosa
import numpy as np

from lib import Project

P = Project(sys.argv[1]); cfg = P.cfg; FPS = P.fps; W, H, TOP, ACT = P.w, P.h, P.top, P.act
TOTAL = P.clips[-1]["e"] / FPS; ORIGIN = P.origin; vt = lambda t: t - ORIGIN
TX = cfg.get("text", {}); BLOCKS = TX.get("blocks", [])
TR = json.load(open(P.work / "tracks.json")) if (P.work / "tracks.json").exists() else {}
OBJ = cfg.get("objects", {})
PUSH_N, PUSH_AMT = cfg.get("cam", {}).get("push_first", 0), cfg.get("cam", {}).get("push", .05)
PUSH = P.clips[:PUSH_N]

# ---- cut-out layers
LAYERS = []
for b in cfg.get("behind", []):
    c = P.clip(b["src"], b["t"]); LAYERS.append(dict(a=b.get("from", c["a"]), e=b.get("to", c["e"]), ca=c["a"], ce=c["e"], ms=P.matte(c, b.get("matte", "sam"))))
enc = {}
for i, f in P.frames(P.hf / "assets/picture.mp4"):
    for l in LAYERS:
        if l["a"] <= i < l["e"]:
            if l["a"] not in enc:
                enc[l["a"]] = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-crf", "14", "-b:v", "0", "-auto-alt-ref", "0", str(P.hf / "assets" / f"cut_{l['a']}.webm")], stdin=subprocess.PIPE)
            k = i - l["ca"] if i >= l["ca"] else l["ca"] - i      # frames before the shot belong to its carry-over, which plays reversed
            al = cv2.GaussianBlur(cv2.resize(l["ms"][k], (W, ACT)).astype(np.float32) / 255, (0, 0), 1.1)
            out = np.zeros((H, W, 4), np.uint8); out[..., :3] = f; out[TOP:TOP + ACT, :, 3] = (al * 255).astype(np.uint8)
            enc[l["a"]].stdin.write(out.tobytes())
for e in enc.values():
    e.stdin.close(); assert e.wait() == 0
json.dump([dict(a=l["a"], e=l["e"], ca=l["ca"], ce=l["ce"]) for l in LAYERS], open(P.work / "cut_layers.json", "w"))
CUTHTML = "\n".join(f'        <video id="cut{l["a"]}" class="clip cut" src="assets/cut_{l["a"]}.webm" muted playsinline data-layout-allow-occlusion data-start="{int(l["a"] / FPS * 1e4) / 1e4:.4f}" data-duration="{(l["e"] - l["a"]) / FPS:.4f}" data-track-index="{6 + n_}"></video>' for n_, l in enumerate(LAYERS))     # start rounded DOWN: a start rounded up (14.4667 for 14.46667) makes the renderer show the previous frame, so the cut-out lags the picture by one frame


def pose(block, f):
    """Screen position, scale and turn of a block at frame f: the object's own move, then the slow push the picture gets."""
    es = block["place"]; hit = [e for e in es if e["at"] <= f]; e_ = hit[-1] if hit else es[0]
    x, y, r = float(e_["x"]), float(e_["y"]), float(e_.get("tilt", 0))
    ob = e_.get("object")
    if ob:
        t = TR[ob]; bx = OBJ[ob]["box"]; hold = e_.get("hold_after", 10 ** 6)
        d = t["t"][min(f, t["e"] - 1, hold) - t["a"]] if f >= t["a"] else t["t"][0]
        th = np.radians(d[2]); cx, cy = bx[0], bx[1] + TOP; dx, dy = x - cx, y - cy
        x = cx + dx * np.cos(th) - dy * np.sin(th) + d[0]; y = cy + dx * np.sin(th) + dy * np.cos(th) + d[1]; r += d[2]
    s = 1.
    for c in PUSH:
        if c["a"] <= f < c["e"]:
            s = 1 + PUSH_AMT * (f - c["a"]) / (c["e"] - c["a"]); x, y = W / 2 + (x - W / 2) * s, H / 2 + (y - H / 2) * s
    return round(x, 1), round(y, 1), round(s, 4), round(r, 2)


FONTS = TX.get("fonts", {"fa": "Archivo Black", "fl": "League Gothic"})
RED = set(TX.get("red", [])); BURST_PX = TX.get("burst_px", 165)
LEAD = {"left": .073, "drop": .069, "pop": .049, "slam": .05, "type": .02, "neon": .03, "glitch": .03}      # words land on the voice: entrances run ahead of it by their time to read
ENT = {"left": ('{opacity:0,x:-60,"--t":0.8}', 'opacity:1,x:0,"--t":0,duration:0.2,ease:"expo.out"'), "drop": ('{opacity:0,y:-42}', 'opacity:1,y:0,duration:0.22,ease:"back.out(1.6)"'),
       "pop": ('{opacity:0,scale:0.72}', 'opacity:1,scale:1,duration:0.18,ease:"back.out(2)"'), "slam": ('{opacity:0,scale:1.14,"--c":0.8}', 'opacity:1,scale:1,"--c":0.15,duration:0.14,ease:"expo.out"'),
       "type": ('{opacity:0}', 'opacity:1,duration:0.02,ease:"none"'),
       "glitch": ('{opacity:0,x:-16,"--c":1.6}', 'opacity:1,x:0,"--c":0,duration:0.17,ease:"steps(4)"'),
       "neon": ('{opacity:0}', 'opacity:1,duration:0.08,ease:"power2.out"')}
LAG, SPREAD = 6, 34
AUDIO_LAG = .010     # measured with work/qa/qa_sync.py's correlation: the render's audio runs 10 ms after the vocal stem the word times come from
F50 = {"neon": 1, "slam": 1, "glitch": 4, "type": 1, "left": 1, "drop": 1, "pop": 1}     # frames after an entrance starts until the word is half visible, read from each tween at 30 fps (glitch: steps(4) over .17 s plus its 16 px slide; pixels showed half brightness one frame after the .5 opacity step)
F0 = {"glitch": 1}   # frames an entrance stays fully hidden after it starts


def word_start(t, ent, cut):
    """First frame of the entrance, so the word is half visible on the last frame at or before the voice (a word never shows after the voice; at most one frame early), and fully hidden before the shot's cut."""
    voice = int((t + AUDIO_LAG) * FPS + 1e-6)
    return max(voice - F50[ent], cut - F0.get(ent, 0) - 1) / FPS


rows_html, js = [], []
JS0 = []
for n, B in enumerate(BLOCKS):
    JS0.append(len(js))
    gid = f"g{n:02d}"; ent, times, end = B["ent"], B["times"], B["end"]; echo = B.get("echo")
    cut_ = B["place"][0]["at"]; a, b = round(min(word_start(t_, ent, cut_) for t_ in times), 4), round(vt(end), 3); body, j, bursts = [], 0, []
    for words, font, px in B["rows"]:
        sp = []
        for w in words:
            cls = "w" + (" red" if w in RED else "")
            ghost = "".join(f'<span class="er" id="{gid}w{j}g{k}"><span class="s">{w}</span></span>' for k in (1, 2, 3)) if echo and echo["word"] == j else ""
            burst = f'<span class="e" id="{gid}w{j}e"><span class="s">{html.escape(w)}</span></span>' if font == "fl" and px >= BURST_PX else ""
            if burst:
                bursts.append(j)
            sp.append(f'<span class="{cls}" id="{gid}w{j}"><span class="s">{html.escape(w)}</span>{burst}{ghost}</span>'); j += 1
        body.append(f'<div class="row {font}" style="font-size:{px}px">{" ".join(sp)}</div>')
    assert j == len(times), (n, j)
    rows_html.append(f'<div id="{gid}" class="clip line" data-start="{a}" data-duration="{round(b - a, 3)}" data-track-index="2"><div class="trk" id="{gid}t"><div class="pos" style="transform:translate(-50%,-50%)"><div class="in" id="{gid}i">{"".join(body)}</div></div></div></div>')
    for k, t in enumerate(times):
        frm, to = ENT[ent]; js.append(f'tl.fromTo("#{gid}w{k}",{frm},{{{to}}},{round(word_start(t, ent, cut_), 4)});')
    for k in bursts:
        js.append(f'tl.fromTo("#{gid}w{k}e",{{opacity:.8,scale:1}},{{opacity:0,scale:1.28,duration:0.28,ease:"power2.out"}},{round(word_start(times[k], ent, cut_) + .03, 3)});')
    if B.get("out") == "glitch":     # a line with time to spare leaves the way a glitch word arrives
        js.append(f'tl.to("#{gid} .w",{{opacity:0,x:16,"--c":1.6,duration:0.17,ease:"steps(4)"}},{round(b - 0.17, 3)});')
    else:
        js.append(f'tl.to("#{gid}i",{{opacity:0,duration:0.06,ease:"power2.in"}},{round(b - 0.07, 3)});')
    js.append(f'tl.set("#{gid}i",{{opacity:0}},{round(b - 0.005, 3)});')
    f0, f1 = round(a * FPS), round(b * FPS) + 1; last = None
    for f in range(f0, f1):
        now = pose(B, f)
        if now != last:
            js.append(f'tl.set("#{gid}t",{{x:{now[0]},y:{now[1]},scale:{now[2]},rotation:{now[3]}}},{int((a if f == f0 else f / FPS) * 1e4) / 1e4:.4f});'); last = now     # rounded DOWN: a time rounded up lands after its frame, so the text froze every third frame and then jumped
    if echo:   # outlines of the word follow the voice's own level, one echo later each
        s0, s1 = echo["song_from"], echo["song_to"]; off = int(s0); y, sr = librosa.load(cfg["song"]["path"], sr=22050, mono=True, offset=off, duration=int(np.ceil(s1)) - off)
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256)); fr = librosa.fft_frequencies(sr=sr, n_fft=2048); e = 20 * np.log10(S[(fr >= 300) & (fr <= 3400)].sum(0) + 1e-6)
        n0, n1 = round(vt(s0) * FPS), round(vt(s1) * FPS)
        lv = np.array([e[min(int((f / FPS + ORIGIN - off) * sr / 256), len(e) - 1)] for f in range(n0, n1)]); lv = (lv - lv.min()) / (lv.max() - lv.min() + 1e-6)
        wi = echo["word"]
        for fr_ in range(n0, n1):
            for k in (1, 2, 3):
                q = fr_ - n0 - k * LAG; l_ = lv[q] if q >= 0 else 0.
                js.append(f'tl.set("#{gid}w{wi}g{k}",{{opacity:{(.3 + .6 * l_) * (1.05 - .25 * k) if q >= 0 else 0:.2f},x:{(-1) ** k * SPREAD * (.6 + .4 * k):.0f}}},{fr_ / FPS:.4f});')
        for k in (1, 2, 3):
            js.append(f'tl.set("#{gid}w{wi}g{k}",{{opacity:0}},{n1 / FPS:.4f});')

# Blocks with fx "invert" live in the main composition on a difference-blend layer: white letters invert the picture behind them, under the cut-out layers.
JS0.append(len(js)); inv_html, inv_js, keep_html, keep_js = [], [], [], []
for n, B in enumerate(BLOCKS):
    part = js[JS0[n]:JS0[n + 1]]; h_ = rows_html[n].replace('class="clip line"', 'class="clip line ' + B.get("fx", "") + '"', 1)
    (inv_html if B.get("fx") == "invert" else keep_html).append(h_); (inv_js if B.get("fx") == "invert" else keep_js).extend(part)
rows_html, js = keep_html, keep_js
cam = []
for c in PUSH:
    cam.append(f'tl.fromTo("#frame,#frame2",{{scale:1}},{{scale:{1 + PUSH_AMT},duration:{(c["e"] - c["a"]) / FPS - .01:.3f},ease:"none",immediateRender:false}},{c["a"] / FPS:.4f});')
    cam.append(f'tl.set("#frame,#frame2",{{scale:1}},{c["e"] / FPS:.4f});')
for t0, p_, o in cfg.get("cam", {}).get("hits", []):   # bar lines only: [second, zoom, flash opacity]
    for k in range(12):
        cam.append(f'tl.set("#frame",{{scale:{1 + p_ * np.exp(-k / FPS / .12):.4f}}},{t0 + k / FPS:.4f});')
    cam.append(f'tl.set("#frame",{{scale:1}},{t0 + 12 / FPS:.4f});')
    cam.append(f'tl.set("#flash",{{opacity:{o}}},{t0:.4f});tl.to("#flash",{{opacity:0,duration:0.18,ease:"power2.out"}},{t0 + .001:.4f});')
fa, fl = FONTS["fa"], FONTS["fl"]; NL = chr(10)
captions = f"""<template>
  <style>
    #root {{ position: absolute; inset: 0; }}
    .line {{ position: absolute; inset: 0; overflow: visible; }}
    .trk {{ position: absolute; left: 0; top: 0; width: 0; height: 0; }}
    .pos {{ position: absolute; left: 0; top: 0; white-space: nowrap; text-align: center; }}
    .row {{ line-height: 0.98; color: #fff; }}
    .fa {{ font-family: "{fa}"; letter-spacing: 0.05em; margin: 0.12em 0; }}
    .fl {{ font-family: "{fl}"; letter-spacing: 0.015em; }}
    .w {{ position: relative; display: inline-block; opacity: 0; --t: 0; --c: 0; transform-origin: 50% 60%;
      text-shadow: calc(var(--c) * -9px) 0 0 rgba(255,40,60,.8), calc(var(--c) * 9px) 0 0 rgba(40,220,255,.8),
        calc(var(--t) * -26px) 0 0 rgba(255,255,255,.4), calc(var(--t) * -52px) 0 0 rgba(255,255,255,.2),
        0 3px 22px rgba(0,0,0,.85), 0 1px 3px rgba(0,0,0,.9); }}
    .s {{ display: inline-block; transform: skewX(-9deg); }}
    .red {{ color: #FF3341; }}
    .e {{ position: absolute; left: 0; top: 0; opacity: 0; color: transparent; text-shadow: none; -webkit-text-stroke: 3px #fff; }}
    .er {{ position: absolute; left: 0; top: 0; opacity: 0; color: rgba(255,255,255,.2); text-shadow: none; -webkit-text-stroke: 2px #fff; }}
    {TX.get("css", "")}
  </style>
  <div id="root" data-composition-id="captions" data-width="{W}" data-height="{H}">
    {NL.join("    " + r for r in rows_html)}
  </div>
  <script>
    const tl = gsap.timeline({{ paused: true }});
    {NL.join("    " + j for j in js)}
    window.__timelines["captions"] = tl;
  </script>
</template>
"""
import re as _re
CAPCSS = _re.search(r"<style>(.*?)</style>", captions, _re.S).group(1).replace("#root { position: absolute; inset: 0; }", "")
INVHTML = NL.join(inv_html); INVJS = NL.join(inv_js)
(P.hf / "compositions").mkdir(exist_ok=True); (P.hf / "compositions" / "captions.html").write_text(captions)
index = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width={W}, height={H}" />
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/gsap.min.js"></script>
    <style>
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      html, body {{ width: {W}px; height: {H}px; overflow: hidden; background: #000; }}
      #root {{ position: relative; width: 100%; height: 100%; overflow: hidden; }}
      #frame {{ position: absolute; inset: 0; }}
      #captions {{ z-index: 3; }}
      #frame2 {{ position: absolute; inset: 0; z-index: 5; pointer-events: none; }}
      .cut {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }}
      #picture {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }}
      #flash {{ position: absolute; inset: 0; opacity: 0; background: #E8F7FF; }}
      #inv {{ position: absolute; inset: 0; z-index: 2; mix-blend-mode: difference; }}
      {CAPCSS}
      #barT, #barB {{ position: absolute; left: 0; right: 0; background: #000; z-index: 10; }}
      #barT {{ top: 0; height: {TOP}px; }} #barB {{ bottom: 0; height: {H - TOP - ACT}px; }}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="{round(TOTAL, 3)}" data-width="{W}" data-height="{H}">
      <div id="frame">
        <video id="picture" class="clip" src="assets/picture.mp4" muted playsinline data-start="0" data-duration="{round(TOTAL, 3)}" data-track-index="0"></video>
        <div id="flash"></div>
      </div>
      <audio id="mix" class="clip" src="assets/mix.wav" data-start="0" data-duration="{round(TOTAL, 3)}" data-track-index="5"></audio>
      <div id="frame2">
{CUTHTML}
      </div>
      <div id="inv">{INVHTML}</div>
      <div id="barT"></div><div id="barB"></div>
      <div id="captions" data-composition-id="captions" data-composition-src="compositions/captions.html" data-start="0" data-duration="{round(TOTAL, 3)}" data-track-index="2" data-track-kind="captions" data-width="{W}" data-height="{H}"></div>
    </div>
    <script>
      window.__timelines = window.__timelines || {{}};
      const tl = gsap.timeline({{ paused: true }});
      tl.set("#frame", {{ scale: 1 }}, 0);
      {INVJS}
      {NL.join("      " + j for j in cam)}
      window.__timelines["main"] = tl;
    </script>
  </body>
</html>
"""
(P.hf / "index.html").write_text(index)
subprocess.run(["cp", str(P.work / "mix.wav"), str(P.hf / "assets" / "mix.wav")], check=True)
print("blocks", len(BLOCKS), "layers", [(l["a"], l["e"]) for l in LAYERS], "total", TOTAL)
