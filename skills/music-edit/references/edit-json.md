# edit.json

One file describes the whole edit. `example-edit.json` beside this file is a complete real one. Times are seconds, `at`/`ref`/`from`/`to` are video frames, and boxes and `x, y` are pixels. Boxes are in **picture coordinates** (row 0 is the top of the picture, under the bar). Text `x, y` is in **full-frame coordinates**, so add `picture_top` to a picture row.

```jsonc
{
  "name": "anime",                       // draft files: {name}_draft_NN.mp4
  "work": "/abs/work",                   // measurements, mattes, sheets
  "project": "/abs/hyperframes-project",
  "drafts": "/abs/drafts",
  "format": {"width": 1920, "height": 1080, "fps": 30, "picture_top": 132, "picture_height": 816},
  "song": {
    "path": "/abs/song.mp3",
    "origin": 63.66133,     // song second at video 0:00, from analyze_song.py
    "beat": 0.4,            // seconds; from analyze_song.py
    "video_drop": 13.2,     // where the drop lands in the video
    "limit": 0.68           // limiter level; lower it if a hot master clips
  },
  "sources": {
    "T": {"path": "...", "vf": "crop=1920:816:0:132"},     // ffmpeg filters that produce exactly width x picture_height
    "G": {"path": "...", "vf": "scale=1920:816:flags=lanczos,unsharp=5:5:0.6:5:5:0",
          "grade": {"matrix": "/abs/work/grade.npy", "chroma": 1.6}}
  },
  "color_fit": {"from": "G", "to": "T", "out": "grade.npy", "pairs": {"name": [from_start, from_end, to_start, to_end]}},
  "start_beat": -33,        // beat of the first cut relative to the drop; the first frame is clamped to 0
  "shots": [["T", 34.85, 5], ["G", 83.1, 4]],             // source, in-point, length in beats (0.5 = half beat)
  "fade_out_frames": 8,

  "sam": [{"src": "T", "t": 8.25, "ref": 288, "box": [440, 0, 1560, 816], "name": "him", "subtract_other": 38}],
          // box is x0, y0, x1, y1 on frame ref. subtract_other is optional, see craft.md.
  "objects": {"blade": {"src": "T", "t": 16.45, "ref": 230, "box": [800, 400, 900, 400]}},
          // tracked things. Here box is centre x, centre y, width, height.

  "fx": {
    "hits": {"13.2": 12},          // second: colour split in px, plus a radial kick. Bar lines only.
    "ticks": [14.6],               // small split on half-beat accents
    "grain": 3.5, "bloom": 0.25, "push": 0.55,
    "soften_src": "T",             // optional, the sharper source in a two-source edit
    "strobe": [13.0],              // inverted flashes
    "clean_until": 2.0,            // no transitions before this second
    "build_transitions": ["glitch", "suck", "glitch", "suck", "whip_v", "glitch"],
    "battle_transitions": ["suck", "spiral", "suck", "suck"],
    "transitions": {"14.0": "suck"},       // optional: fix a cut's transition, or null for a hard cut
    "transitions_explicit_only": false,    // true: only the map above gets transitions
    "carry": [{"kind": "out", "src": "T", "t": 12.55, "n": 6, "matte": "sam"}],    // "in" or "out"; n frames; matte sam or other
    "aura": [{"src": "T", "t": 8.25, "matte": "sam", "color": [255, 190, 70]}],    // glow around a subject, BGR
    "blade": {"src": "T", "t": 16.45, "object": "blade", "base": [1058, 412], "angle": 11, "half_width": 120, "length": 1100,
              "sweeps": [[7.2, 1.0], [8.0, 0.6]]}                                  // [second, strength]; each runs base to tip in one beat
  },

  "behind": [{"src": "T", "t": 8.25, "from": 268, "to": 300, "matte": "sam"}],
          // shots whose subject sits above the text. from and to default to the whole shot. from can reach back into an "in" carry-over.
  "cam": {"push_first": 3, "push": 0.05, "hits": [[13.2, 0.16, 0.85]]},            // slow zoom on the first shots; [second, zoom, flash] on bar lines

  "text": {
    "fonts": {"fa": "Archivo Black", "fl": "League Gothic"},    // fa = small wide type, fl = tall big type
    "red": ["HEART"],             // words drawn in red
    "burst_px": 165,              // big words this size or more shoot off an outline
    "blocks": [{
      "ent": "drop",              // left, drop, pop or slam
      "rows": [[["MY"], "fa", 46], [["HEART"], "fl", 240]],     // words, font, size in px
      "times": [70.28, 71.1],     // one SONG second per word, when it is sung
      "end": 71.65,               // song second when the block leaves
      "place": [                  // the last entry with at <= frame applies
        {"at": 192, "x": 620, "y": 345, "tilt": -11, "object": null},
        {"at": 204, "x": 620, "y": 345, "tilt": -11, "object": "blade", "hold_after": 240}
      ],
      "echo": {"word": 2, "song_from": 88.5, "song_to": 89.58}  // optional: outlines of that word follow the voice
    }]
  }
}
```

## Notes

- Shots are identified by `src` and `t` everywhere. Two shots with the same in-point in one source would collide, and footage used twice is flagged anyway.
- A shot's frames change when any shot before it changes length, so check `at`, `ref`, `from` and `to`, and rerun `matte_sam.py` for shots that moved (mattes are named by frame range).
- Hard cuts at `behind` shots and carry-over cuts come automatically, unless `transitions` names the cut.
- For vertical edits, set the format to 1080x1920 with `picture_top` 0 and height 1920, and use `vf` filters that crop the action into frame.
