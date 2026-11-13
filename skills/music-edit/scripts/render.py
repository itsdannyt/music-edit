"""Render a numbered draft: project check, delivery render, exact mix muxed on, and a phone copy small enough to send.
usage: render.py edit.json DRAFT_NUMBER [--draft-quality]"""
import subprocess
import sys

from lib import Project, node_env

P = Project(sys.argv[1]); n = int(sys.argv[2]); env = node_env(); fps = P.fps
P.drafts.mkdir(parents=True, exist_ok=True); name = P.cfg.get("name", "edit")
chk = subprocess.run("npx -y hyperframes@latest check", shell=True, cwd=P.hf, env=env, capture_output=True, text=True)
print("\n".join(l for l in chk.stdout.splitlines() if "error(s)" in l)[:400])
if chk.returncode:
    print(chk.stdout[-1500:]); sys.exit("project check failed")
q = "draft" if "--draft-quality" in sys.argv else "delivery"; raw = P.work / f"render_{n:02d}.mp4"
subprocess.run(f'npx -y hyperframes@latest render --fps {fps} --quality {q} --output "{raw}"', shell=True, cwd=P.hf, env=env, check=True, stdout=subprocess.DEVNULL)
full, phone = P.drafts / f"{name}_draft_{n:02d}.mp4", P.drafts / f"{name}_draft_{n:02d}_phone.mp4"
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-i", str(P.hf / "assets/mix.wav"), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", str(full)], check=True)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(full), "-c:v", "libx264", "-crf", "25", "-preset", "medium", "-c:a", "copy", "-movflags", "+faststart", str(phone)], check=True)
raw.unlink()
print(full, f"{full.stat().st_size / 1e6:.1f} MB"); print(phone, f"{phone.stat().st_size / 1e6:.1f} MB")
