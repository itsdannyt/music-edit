# Craft notes: using each effect well

## Beat sync

Set the grid from the **attack** of the hit, not a detector's peak. A peak sits 30-60 ms after the hit starts, which leaves the picture trailing every hit by 1-2 frames. `analyze_song.py` does this. Sync limits (ATSC IS-191) allow sound to lead by 15 ms at most and lag by 45 ms at most, so aim for the hit landing just after the cut. Use a causal filter to measure attacks, because a zero-phase filter rings before the attack and makes it look earlier than it is.

Fit a constant grid to the kicks when the track holds tempo, because beat trackers jitter by up to 50 ms. Use half-beat cuts only where the beat table shows a real accent, and put a held close-up in a kick-free gap.

## Keeping the fight readable

Count clean frames after the drop: `fx.py` prints the share. Past edits landed around 75-80%. A 0.2 s shot has 6 frames, so a transition on each end leaves none clean; those get hard cuts. Cap transitions at 2 frames a side after the drop. Test glow by measuring: choose the strongest level at which no sampled frame goes over 30% white.

## Cutouts

Draw the box tight on the frame where the subject is largest and clearest. Check `work/sheets/mattes.png` at the start, middle and end of each shot. SAM 2 handles low-res footage and small subjects that isnet and GrabCut lose. Known weaknesses:

- It fills enclosed holes, such as a hair loop. Run `matte_isnet.py` on that shot and set `subtract_other` (38 worked). Where the second model is near zero, the hole comes out.
- It can pick up a neighboring object at shot ends, or fail on huge close-ups with soft edges. Try another box or skip the shot.
- A grid-like pattern in a soft area means low confidence. The hard-edge cleanup in `lib.py` usually removes it. Check anyway.

## Carry-overs

Good: a subject that doesn't land on another subject or across a face, with a background that reads behind it. Standing in front of a looming enemy works well as a story beat. Carry 6-8 frames on long shots and 3-4 next to 0.2 s shots, so the shot behind still reads. A carried cut replaces that cut's transition. Frames from the other shot play reversed so it is continuous when it takes over, and this hasn't been noticed in review.

## Text behind subjects

Three things stop it glitching, and all three matter:

1. **Hard matte.** Soft alpha lets the word show through on some frames. The anime model dipped to 53% see-through on a sleeve for four frames, compared with 1.5% from SAM 2.
2. **The layer covers the whole time the word is on that shot.** Warped transition frames can't be cut out, so shots in `behind` automatically get hard cuts at their ends. If a shot has a zoom punch, start the layer after it with `from`.
3. **Small words stay fully clear.** A chopped small word reads as a mistake; only big words tuck behind.

Aim for 5-25% of the big text hidden. Under that it doesn't read as behind; over that the word gets hard to read. A subject that drifts changes the share, so check the audit and the last frames. Don't cut out the very first text on screen.

## Pinned text

Choose something rigid, such as an ear, a guard or a logo, not hair or a mouth. `measure.py` prints lost frames and the largest step. A big jump means the object really moved or tracking failed; look at it. If a track can't hold, set `object` to null and the text holds still, which is better than sliding. Long blocks cross cuts, so give them a new `place` entry per long shot and let short shots inherit. When an object whips away just as a word lands, use `hold_after` so it stays readable. Keep the same `x, y` when you only want pinning with no jump at the cut.

## Transitions

- `suck`: pushes toward each shot's measured action point. The default after the drop.
- `spiral`: the same with a turn, limited by geometry so no picture edge shows. Use for variety.
- `glitch`: tension, before the drop. It hides action, so skip it after.
- `whip_r`, `whip_l`, `whip_v`: only where the footage actually pans. `fx.py` chooses side whips from measured motion.

## Object effects

Effects should come from the object and the story. The blade effect darkens the blade so its marks can burn, sweeps a glint along it for exactly one beat so it crosses the middle as a word lands, and adds drifting mist for a character whose style is mist. For a new object, measure its base point, angle, half width and length on the tracked `ref` frame, and test on the raw shot before a full build. For other effects, add a new function next to `blade_fx` in `fx.py`, and keep it pinned to a track.

## Flash and punch levels

Keep full-strength flashes for the drop and the climax, and halve them on mid-song bar lines. Movie flashes in the source plus added glow can white out a whole frame, so start a shot just past its flash if it does.
