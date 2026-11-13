"""Final checks. Numbers report what is true; they do not replace watching at full speed.
text:  renders text alone. Edges of frame and bars; per cut-out layer, share of small and big words hidden; any frame where text pops in front of a cut-out subject.
beats: every cut against the grid, and where the hits' attacks land relative to the cuts, in the final file's own audio. Target: attack 0-45 ms after the cut (sound may lead by 15 ms at most, ATSC IS-191).
file:  frames, bars, near-white frames, audio peak, optionally diff against the previous draft.
usage: audit.py edit.json text|beats|file|all [DRAFT_NUMBER] [--compare PREVIOUS_NUMBER]"""
import json
import re
import shutil
import subprocess
import sys

import cv2
import numpy as np
import scipy.signal as sg
import soundfile as sf

from lib import Project, node_env

P = Project(sys.argv[1]); what = sys.argv[2]; a_ = sys.argv
num = int(a_[3]) if len(a_) > 3 and a_[3].isdigit() else None
name = P.cfg.get("name", "edit"); W, H, TOP, ACT, FPS = P.w, P.h, P.top, P.act, P.fps
full = P.drafts / f"{name}_draft_{num:02d}.mp4" if num else None


def text_audit():
    T = P.work / "audit_tmp"

    def render(css):
        shutil.rmtree(T, ignore_errors=True); (T / "compositions").mkdir(parents=True)
        for j in P.hf.glob("*.json"):
            shutil.copy(j, T)
        (T / "assets").symlink_to(P.hf / "assets")
        h = (P.hf / "index.html").read_text(); h = re.sub(r'<video .*?</video>', '', h, flags=re.S); h = re.sub(r'<audio id="mix".*?</audio>', '', h, flags=re.S)
        h = h.replace('<div id="barT"></div>', '').replace('<div id="barB"></div>', '').replace('<div id="flash"></div>', ''); h = re.sub(r'tl\.(set|to)\("#flash"[^;]*;', '', h)
        h = h.replace("</style>", css + "</style>", 1); (T / "index.html").write_text(h); (T / "compositions/captions.html").write_text((P.hf / "compositions/captions.html").read_text().replace("<style>", "<style>\n" + css, 1))
        subprocess.run(f'npx -y hyperframes@latest render --fps {FPS} --quality draft --output "{T}/t.mp4"', shell=True, cwd=T, env=node_env(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(T / "t.mp4"), "-vf", "format=gray", "-f", "rawvideo", "-"], stdout=subprocess.PIPE); fr = []
        while True:
            b = p.stdout.read(W * H)
            if len(b) < W * H:
                break
            fr.append(np.frombuffer(b, np.uint8).reshape(H, W)[TOP:TOP + ACT] > 70)
        return fr
    small, big = render(".fl,.hv{visibility:hidden!important}"), render(".fa{visibility:hidden!important}")
    shutil.rmtree(T, ignore_errors=True)
    edge = []
    for i, (s, b) in enumerate(zip(small, big)):
        u = s | b
        if u.any():
            xs, ys = np.where(u.any(0))[0], np.where(u.any(1))[0]
            if xs.min() < 24 or xs.max() > W - 24 or ys.min() < 2 or ys.max() > ACT - 3:
                edge.append(round(i / FPS, 2))
    print("text frames at frame or bar edges:", edge)
    layers = json.load(open(P.work / "cut_layers.json")) if (P.work / "cut_layers.json").exists() else []
    active = {f for l in layers for f in range(l["a"], l["e"])}; pop = []
    kinds = {(P.clip(b["src"], b["t"])["a"]): b.get("matte", "sam") for b in P.cfg.get("behind", [])}
    for l in layers:
        c = P.clip_at(l["ca"]); m = P.matte(c, kinds.get(l["ca"], "sam")); ca = l["ca"]; bad = []; bs = []
        for i in range(l["a"], l["e"]):
            k = i - ca if i >= ca else ca - i; mm = cv2.resize(m[k], (W, ACT)) > 128; s, b = small[i], big[i]
            if s.sum() > 80 and (s & mm).sum():
                bad.append((round(i / FPS, 2), round(float((s & mm).sum() / s.sum()) * 100)))
            if b.sum() > 800:
                bs.append(float((b & mm).sum() / b.sum()))
        v = np.array(bs) if bs else np.zeros(1)
        print(f"layer {l['a']}-{l['e']}: small words hidden on {len(bad)} frames {bad[:6]} | big words hidden min {v.min() * 100:.0f}% med {np.median(v) * 100:.0f}% max {v.max() * 100:.0f}%, biggest frame jump {np.abs(np.diff(v)).max() * 100 if len(v) > 1 else 0:.0f}%")
        for i in range(ca, l["ce"]):
            if i not in active and (small[i] | big[i]).sum() > 80 and (((small[i] | big[i]) & (cv2.resize(m[i - ca], (W, ACT)) > 128)).sum() > 40):
                pop.append(round(i / FPS, 2))
    print("frames where text pops in front of a cut-out subject:", pop)
    print("Pass: no edges, small words 0 frames, no pops, no big frame jump. Big words 5-25% hidden stay readable and still read as behind.")


def beats_audit():
    song = P.song; bt, drop = song["beat"], song["video_drop"]; cuts = [c["a"] for c in P.clips[1:]]
    off = [abs((f / FPS - drop) / (bt / 2) - round((f / FPS - drop) / (bt / 2))) * (bt / 2) for f in cuts]
    print(f"cuts {len(cuts)}: off the half-beat grid by more than a frame: {sum(o > 1 / FPS for o in off)}, largest {max(off) * 1000:.0f} ms. Gaps or overlaps: {sum(P.clips[i]['e'] != P.clips[i + 1]['a'] for i in range(len(P.clips) - 1))}")
    tmp = P.work / "audit_audio.wav"; subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(full or P.work / "mix.wav"), "-vn", "-ac", "1", "-ar", "44100", str(tmp)], check=True)
    y, sr = sf.read(tmp); tmp.unlink(); end = len(y) / sr
    for label, lo, hi in (("low 30-150", 30, 150), ("high 2-8k", 2000, 8000)):
        so = sg.butter(4, [lo, hi], "band", fs=sr, output="sos"); e = np.abs(sg.sosfilt(so, y)); k = int(.002 * sr); e = np.convolve(e, np.ones(k) / k, "same"); r = []
        for t in np.arange(drop, end - .2, bt):
            s0, s1 = int((t - .025) * sr), int((t + .06) * sr); seg = e[s0:s1]; pk = int(seg.argmax()); pre = np.median(e[int((t - .05) * sr):s0])
            if seg[pk] >= pre * 2.5:
                r.append(((s0 + np.argmax(seg[:pk + 1] >= pre + (seg[pk] - pre) * .5)) / sr - t) * 1000)
        if r:
            print(f"{label}: attacks after the cut, ms: n {len(r)} median {np.median(r):.1f} IQR {np.percentile(r, 25):.0f} to {np.percentile(r, 75):.0f}")
    print("Median outside 0-45 ms: change song.origin by whole frames and rerun build_shots.py (mix only changes) and text.py. A reading at the ends of the search window usually means a different event: check by cross-correlating old and new audio.")
    so = sg.butter(4, [30, 150], "band", fs=sr, output="sos"); e = np.abs(sg.sosfilt(so, y)); k = int(.003 * sr); e = np.convolve(e, np.ones(k) / k, "same")
    level = lambda t: float(e[int((t - .03) * sr):int((t + .07) * sr)].max())
    ref = np.median([level(t) for t in np.arange(drop, end - .2, bt)])      # a typical kick after the drop
    weak = [round(f / FPS, 2) for f in cuts if f / FPS >= drop and level(f / FPS) < .5 * ref]
    print("cuts after the drop with no clear kick under them (tell the editor; kick-free gaps are often deliberate):", weak)


def file_audit():
    w, h = 480, 270

    def rd(q):
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(q), "-vf", f"scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
        return np.frombuffer(p.stdout.read(), np.uint8).reshape(-1, h, w)
    b = rd(full); ph = rd(P.drafts / f"{name}_draft_{num:02d}_phone.mp4"); t = max(1, round(TOP * h / H)); bt_ = max(1, round((H - TOP - ACT) * h / H))
    print("frames", len(b), len(ph), "expected", P.clips[-1]["e"], "| bars max (full top, bottom, phone top, bottom):", b[:, :t - 1].max() if t > 1 else 0, b[:, -bt_ + 1:].max() if bt_ > 1 else 0, ph[:, :t - 1].max() if t > 1 else 0, ph[:, -bt_ + 1:].max() if bt_ > 1 else 0)
    print("over 70% white at", [round(float(i) / FPS, 2) for i in np.where((b[:, t + 1:h - bt_ - 1] > 235).mean(axis=(1, 2)) > .7)[0]])
    a = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", str(full), "-map", "0:a:0", "-f", "s16le", "-ac", "2", "-"], capture_output=True).stdout, np.int16)
    print("audio peak", round(float(np.abs(a).max()) / 32768, 3), "(1.0 would mean clipping)")
    if "--compare" in a_:
        pv = P.drafts / f"{name}_draft_{int(a_[a_.index('--compare') + 1]):02d}.mp4"; o = rd(pv); d = np.abs(o[:len(b)].astype(np.int16) - b[:len(o)]).mean(axis=(1, 2)); ch = np.where(d > .6)[0]
        print("changed frames vs previous:", len(ch), f"from {ch.min() / FPS:.2f} to {ch.max() / FPS:.2f} s" if len(ch) else "")


if what in ("text", "all"):
    text_audit()
if what in ("beats", "all"):
    beats_audit()
if what in ("file", "all") and full:
    file_audit()
