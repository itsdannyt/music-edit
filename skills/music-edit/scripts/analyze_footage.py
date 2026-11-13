"""Index a source clip's shots and make contact sheets so shots can be chosen by eye.
usage: analyze_footage.py edit.json SOURCE_KEY     (writes work/cuts_KEY.json and work/sheets/KEY_NN.png)"""
import json
import sys

import cv2
import numpy as np
from scenedetect import AdaptiveDetector, SceneManager, open_video

from lib import Project

P = Project(sys.argv[1]); key = sys.argv[2]; path = P.cfg["sources"][key]["path"]
v = open_video(path); sm = SceneManager(); sm.add_detector(AdaptiveDetector()); sm.detect_scenes(v)
sc = [(a.get_seconds(), b.get_seconds()) for a, b in sm.get_scene_list()]
cap = cv2.VideoCapture(path); end = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
print(f"{key}: {cap.get(3):.0f}x{cap.get(4):.0f}, {cap.get(cv2.CAP_PROP_FPS):.2f} fps, {end:.1f} s, {len(sc)} shots. Check the real resolution: file names lie.")
json.dump(dict(cuts=[dict(t=round(a, 3)) for a, _ in sc[1:]], end=round(end, 3)), open(P.work / f"cuts_{key.lower()}.json", "w"))
cells = []
for a, b in sc:
    cap.set(cv2.CAP_PROP_POS_MSEC, (a + b) / 2 * 1000)
    ok, f = cap.read()
    if ok:
        x = cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1])))
        cv2.putText(x, f"{a:.2f}-{b:.2f}", (4, 18), 0, .55, (0, 255, 0), 2); cells.append(x)
for s in range(0, len(cells), 48):
    part = cells[s:s + 48]
    while len(part) % 6:
        part.append(np.zeros_like(cells[0]))
    cv2.imwrite(str(P.work / "sheets" / f"{key}_{s // 48:02d}.png"), np.vstack([np.hstack(part[i:i + 6]) for i in range(0, len(part), 6)]))
print("sheets in", P.work / "sheets")
