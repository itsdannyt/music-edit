---
name: music-edit
description: Make a TikTok-style music edit (AMV, movie edit, anime edit, fan edit, sports edit) by cutting scenes from video clips to a song, with cuts on the beat, cut-out overlays between shots, push-in transitions, lyric text pinned to objects, and big words behind characters. Use this whenever someone wants to cut footage to a song, make an edit or AMV, sync shots to a beat drop, or asks for "a music edit like the last one", even if they only say "here's a song and two clips". Not for vlogs or spoken videos, or for motion graphics with no source footage.
---

# Music Edit

Turn clips and a song into a finished, beat-locked edit, one reviewed draft at a time. The tools handle measuring and rendering. Your job is the editing judgment: which shots tell the story, where a cutout helps, and when an effect gets in the way.

Everything about one edit lives in a single `edit.json`. Changing the edit means changing that file and rerunning from the stage it affects. Read [references/edit-json.md](references/edit-json.md) before writing one, and [references/example-edit.json](references/example-edit.json) to see a complete real edit (lyrics replaced with placeholders).

Read [references/taste.md](references/taste.md) before showing anything. It holds what 49 drafts across two edits taught about what looks good and what gets rejected. [references/craft.md](references/craft.md) explains how to use each effect well, and is worth reading before Stage 3.

## Setup (once per machine)

The tools need their own Python with PyTorch and SAM 2. **Ask before running this for the first time**, because it downloads about 1.3 GB:

```bash
bash ~/.claude/skills/music-edit/scripts/setup.sh
```

It installs to `~/.music-edit`. `--whisper` adds lyric timing and `--isnet` adds the optional anime matte; ask before each. HyperFrames renders the final video, so load the `hyperframes` skill's rules before editing compositions by hand.

Run every tool with `~/.music-edit/venv/bin/python` from the `scripts/` folder. Below, `PY` means that Python and `E` means the path to `edit.json`.

## How to work with the editor

Work in the editor's three stages and get a reaction after each: **shots and story, then color, then text and effects.** Shot changes are cheap early and expensive after effects are built on them, so don't race ahead.

- Send the phone copy after every draft with the file-sending tool, with a one-line note on what changed.
- Stills and numbers can't tell you how an edit feels. Say plainly that you haven't watched it at full speed, and treat the editor's eye and ear as final. You can't hear the song at all.
- When they name a timestamp, look at those exact frames before answering. Find the mechanism, then fix that, and check the fix did what you predicted.
- Change the smallest thing that answers the note, then compare with the last draft (`audit.py E file N --compare N-1`) and report what moved. Editors notice when something they already approved changes.
- Every strength is a starting value. Say so, and offer to dial it.
- Warn about rapid flashing if the edit has strobes and might be posted.
- Quote song lyrics in chat only as far as you need to identify a line.

## Stage 0: Map the material

Ask for the clips, the song, the lyrics if they want text, the length, and the aspect ratio. Then create a project with `npx -y hyperframes@latest init <dir> --non-interactive` (needs node 22+) and a work folder, and write the top of `edit.json`.

```bash
$PY analyze_song.py SONG work/song_map.json --drop SONG_SECONDS --video-drop VIDEO_SECONDS
$PY analyze_footage.py $E KEY     # once per source; writes shot lists and contact sheets
```

- `analyze_song.py` gives tempo, drop candidates and a table of kicks and snares per beat, plus the `origin` that aligns the drop. Leave `--drop` off first to see the candidates, then listen for the real drop with the editor, because the candidates are only rises in bass level. A low "kick strength on the grid" means a looser beat, so check the grid against the audit before relying on it.
- Check each source's real resolution and frame rate. File names lie: a "4K" clip here was 640 wide.
- Letterboxed sources keep their bars inside the frame. Measure the picture rows and fix `picture_top` and `picture_height`, so bars stay black and still.
- Look through the contact sheets yourself before proposing shots. For lyrics, `lyrics_align.py` gives word times to check against the song.

## Stage 1: Shots and story

Write `shots` as `[source, in-point, beats]`, then:

```bash
$PY build_shots.py $E     # refuses shots that run across a source cut; writes work/rough_cut.mp4
```

- Tell a story with the shots. Before the drop, build tension and let it breathe. After it, go to full action, with cuts on the beat or on half-beats that carry a real accent in the beat table.
- The line "does this make sense?" matters more than pace. Does a character hold the sword the right way, does anything appear from nowhere, is a shot on a tail end of the source's own transition? Check the first and last frames of every shot with `sheet.py`.
- Use real time only. Slowed clips and repeated frames read as laggy, unless the source itself is slow-motion.
- Prefer the sharpest version when the same shot exists in two clips.

## Stage 2: Color (only when sources differ)

If two clips contain the same drawings or scenes, measure rather than eyeball: set `color_fit` pairs, run `color_fit.py`, and set the source's `grade`. Report delta E before and after on the held-out pairs. Under about 5 reads as the same color. Set `fx.soften_src` to the sharper source if the detail gap is obvious.

## Stage 3: Effects and text

Choose objects and spots by looking at frames with coordinates:

```bash
$PY sheet.py work/picture_raw.mp4 work/sheets/grid.png 30,84,130 3 660 --grid --top TOP --act HEIGHT
```

Then cut out subjects and check them before using any:

```bash
~/.music-edit/venv/bin/python -I matte_sam.py $E     # writes work/sheets/mattes.png
```

Look at the magenta sheet. A clean score doesn't prove a clean cutout, and rough edges or kept background are worse than no overlay. Then build, in this order:

```bash
$PY measure.py $E all     # motion at cuts; tracks the objects
$PY fx.py $E              # transitions, overlays, glow, object effects
$PY text.py $E            # cut-out layers, text, composition
$PY render.py $E N        # numbered draft and phone copy
```

Which step to rerun: text position or wording needs `text.py`; effects need `fx.py` then `text.py`, because layers are cut from the finished picture; new shots need all of it, plus any `sam` entries on shots whose frames moved.

## Final check

Before calling a draft final, and any time text or timing changes:

```bash
$PY audit.py $E all N --compare N-1
```

It reports text at edges, words hidden or popping through characters, cuts off the grid, where the hits land against the cuts, weak cuts, bars, near-white frames and audio peak. Fix what is wrong and report the rest as what's still there. If the editor asks for the "final check", tell them plainly which cuts won't feel like a hit and why, such as kick-free gaps.

## Finishing up

When it's final, offer to keep `edit.json` in the project folder, since it rebuilds the whole edit. Add anything new the editor taught you to `references/taste.md`, so the next edit starts smarter.
