# music-edit

A Claude Code skill for TikTok-style music edits: movie edits, anime edits, AMVs and sports edits. Give it a song and a few clips. It cuts the shots on the beat, adds cut-out overlays and push-in transitions, pins lyric text to objects, and puts big words behind characters. You review one draft at a time.

## Start here

Open **Claude Code** (the Code tab in the Claude desktop app, or `claude` in a terminal) and paste:

> Read https://github.com/itsdannyt/music-edit and follow its setup steps, then guide me through making my first music edit.

That's it. Claude installs everything, asks before any big download, then asks for your song and clips.

**You need:** a Mac (Apple Silicon is best), Claude Code on a paid plan, and about 3 GB free disk space. Regular Claude chat won't work, because the skill runs programs on your computer.

---

## Setup steps (for Claude)

Do these in order. Explain each step in a sentence. **Ask the user before you install anything or download anything large.**

1. **Check the tools.** Run `uname -sm`, `git --version`, `ffmpeg -version`, `node --version`, and look for `python3.12`, `python3.11` or `python3.10`.
   - Needed: macOS, git, ffmpeg, Node 22 or newer, and Python 3.10–3.12. Newer Python versions may not support PyTorch yet.
   - For anything missing, offer the Homebrew command (`brew install ffmpeg`, `brew install python@3.12`, `brew install node`). If Homebrew is missing too, point the user to https://brew.sh and wait for them to install it.

2. **Install the skill.**
   ```bash
   rm -rf /tmp/music-edit-src && git clone --depth 1 https://github.com/itsdannyt/music-edit /tmp/music-edit-src
   mkdir -p ~/.claude/skills
   cp -R /tmp/music-edit-src/skills/music-edit ~/.claude/skills/
   ```
   If `~/.claude/skills/music-edit` already exists, this is an update. Say so before copying over it.

3. **Install the HyperFrames skills**, which render the video:
   ```bash
   npx -y hyperframes@latest skills update
   ```

4. **Run the skill's one-time setup.** It downloads about 1.3 GB (PyTorch and the SAM 2 cut-out model) into `~/.music-edit`. Ask first.
   ```bash
   bash ~/.claude/skills/music-edit/scripts/setup.sh
   ```
   If the user wants lyrics on screen, offer `--whisper` for lyric timing (about 1.6 GB more on first use). For anime footage, offer `--isnet` (about 170 MB). Ask before each.

5. **Start the edit.** Read `~/.claude/skills/music-edit/SKILL.md` and follow it from **Stage 0**. You don't need to restart Claude Code. Ask the user for:
   - the song (an audio file)
   - the clips (video files on their computer)
   - the lyrics, if they want text
   - the length, and the aspect ratio (9:16 for TikTok and Reels)
   - a folder to work in

   Tell the user that each draft gets sent as a phone-sized copy for review, and that their notes steer the next draft.

## Updating

Paste the same line again. Step 2 replaces the skill with the latest version, and the setup steps are skipped if they're already done.

## What's inside

- `skills/music-edit/SKILL.md`: the workflow Claude follows
- `skills/music-edit/references/`: taste notes learned from real reviews, effect craft, and the `edit.json` format with a full example
- `skills/music-edit/scripts/`: the tools for song analysis, shot building, cut-outs, effects, text, rendering and the audit

Use footage and music you have the rights to post.
