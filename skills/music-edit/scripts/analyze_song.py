"""Map a song before cutting: tempo, drop candidates, a beat-by-beat table of kicks and snares, and the origin that puts cuts on the attack of the hit.
usage: analyze_song.py SONG OUT.json [--drop SONG_SECONDS] [--video-drop VIDEO_SECONDS] [--fps 30]

Why attack, not peak: a detector's peak sits 30-60 ms after the hit starts. A grid set from peaks leaves the picture trailing every hit by 1-2 frames.
Filters are causal here, because zero-phase filters ring before the attack and smear the timing."""
import argparse
import json

import librosa
import numpy as np
import scipy.signal as sg

ap = argparse.ArgumentParser()
ap.add_argument("song"); ap.add_argument("out"); ap.add_argument("--drop", type=float); ap.add_argument("--video-drop", type=float); ap.add_argument("--fps", type=float, default=30)
a = ap.parse_args()
y, sr = librosa.load(a.song, sr=44100, mono=True)


def band(lo, hi, zero_phase=False):
    so = sg.butter(4, [lo, min(hi, sr / 2 - 1)], "band", fs=sr, output="sos")
    e = np.abs(sg.sosfiltfilt(so, y) if zero_phase else sg.sosfilt(so, y))
    k = int(.002 * sr)
    return np.convolve(e, np.ones(k) / k, "same")


low, high = band(30, 130, True), band(2000, 8000)
d = np.maximum(np.diff(low, prepend=low[0]), 0)
k = int(.008 * sr)
d = np.convolve(d, np.ones(k) / k, "same"); d /= np.percentile(d, 99.9)
kp, _ = sg.find_peaks(d, height=.25, distance=int(.09 * sr))
kicks = kp / sr
# Tempo: tracker's estimate, then a constant grid fitted to the kicks. Dance tracks hold tempo, and tracked beats jitter by up to 50 ms.
guess = float(np.atleast_1d(librosa.feature.tempo(y=y, sr=sr))[0])
best = None
for per in np.arange(60 / guess * .97, 60 / guess * 1.03, .0002):
    for ph in np.arange(0, per, .002):
        e = np.abs(kicks - (np.round((kicks - ph) / per) * per + ph))
        sc = (e < .03).sum() - e[e < .03].sum() * 5
        if best is None or sc > best[0]:
            best = (sc, per, ph)
per, ph = best[1], best[2]
print(f"tempo {60 / per:.2f} BPM, beat {per:.4f} s, kicks {len(kicks)}, on grid {int((np.abs(kicks - (np.round((kicks - ph) / per) * per + ph)) < .03).sum())}")
# Drop candidates: biggest rise in low-band level between the 4 s before and the 4 s after a grid beat.
grid = np.arange(np.ceil((4 - ph) / per), np.floor((len(y) / sr - 4 - ph) / per)) * per + ph
ratio = [(low[int(t * sr):int((t + 4) * sr)].mean() / (low[int((t - 4) * sr):int(t * sr)].mean() + 1e-6), float(t)) for t in grid]
cands, seen = [], []
for r, t in sorted(ratio, reverse=True):
    if all(abs(t - s) > 8 for s in seen):
        cands.append((round(t, 3), round(float(r), 2))); seen.append(t)
    if len(cands) == 5:
        break
print("drop candidates (song seconds, level rise):", cands)
drop = a.drop or cands[0][0]
# Sections of a song can sit off each other's grid, so lock the grid to the kicks around the drop itself. Bass wiggles also register as kicks, so the fit is weighted by hit strength: the real kicks decide.
i0, i1 = int(max(0, drop - 1) * sr), int((drop + 15) * sr)
ld = d[i0:i1] / (np.percentile(d[i0:i1], 99.9) + 1e-9)
lp, lh = sg.find_peaks(ld, height=.25, distance=int(.09 * sr))
near, st = (lp + i0) / sr, lh["peak_heights"]
if len(near) >= 4:
    best = None
    for per_ in np.arange(per * .99, per * 1.01, .0002):
        for ph_ in np.arange(0, per_, .002):
            e = near - (np.round((near - ph_) / per_) * per_ + ph_)
            sc = float((st * np.exp(-(e / .015) ** 2)).sum())
            if best is None or sc > best[0]:
                best = (sc, per_, ph_)
    per, ph = best[1], best[2]
    on = np.abs(near - (np.round((near - ph) / per) * per + ph)) < .03
    print(f"around the drop: beat {per:.4f} s ({60 / per:.2f} BPM), {st[on].sum() / st.sum() * 100:.0f}% of kick strength on the grid")
    grid = np.arange(np.ceil((4 - ph) / per), np.floor((len(y) / sr - 4 - ph) / per)) * per + ph
drop = float(np.round((drop - ph) / per) * per + ph)
# Attack of the hits near the drop, relative to the kick grid.
sos = sg.butter(4, [2000, 8000], "band", fs=sr, output="sos")
offs = []
for t in np.arange(drop, min(drop + 12, len(y) / sr - 1), per):
    s0, s1 = int((t - .09) * sr), int((t + .03) * sr)
    seg = high[s0:s1]; pk = int(seg.argmax()); pre = np.median(high[int((t - .16) * sr):int((t - .1) * sr)])
    if seg[pk] >= pre * 2.5:
        offs.append((s0 + np.argmax(seg[:pk + 1] >= pre + (seg[pk] - pre) * .5)) / sr - t)
att = float(np.median(offs)) if offs else 0.
print(f"drop on grid at song {drop:.3f}. Hit attack starts {att * 1000:.0f} ms from it ({len(offs)} hits measured).")
out = dict(period=per, phase=ph, bpm=60 / per, drop_song=drop, attack_offset=att, drop_candidates=cands)
if a.video_drop is not None:
    # Sync limits (ATSC IS-191): sound may lead picture by 15 ms at most and lag by 45 ms at most. Aim the attack just after the cut, inside that.
    lag = .25 / a.fps
    out["origin"] = round(drop + att - a.video_drop - lag, 5)
    print(f"origin for video drop at {a.video_drop}: {out['origin']} (attack lands {lag * 1000:.0f} ms after the cut)")
rows = []
for t in grid:
    i, j = int((t - .03) * sr), int((t + .07) * sr); h = t + per / 2; i2, j2 = int((h - .03) * sr), int((h + .07) * sr)
    rows.append(dict(song=round(float(t), 3), kick=round(float(low[i:j].max()), 3), snare=round(float(high[i:j].max()), 3), half_kick=round(float(low[i2:j2].max()), 3), half_snare=round(float(high[i2:j2].max()), 3)))
out["beats"] = rows
json.dump(out, open(a.out, "w"), indent=0)
print("beat table written:", a.out, "- use it to find kick-free gaps, and half-beats that carry a real accent")
