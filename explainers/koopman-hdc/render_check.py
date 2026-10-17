#!/usr/bin/env python3
"""
Render a ManimGX scene and build a contact sheet of frames to LOOK at.

    python render_check.py FILE.py SceneName                 # 360p video + contact sheet
    python render_check.py FILE.py SceneName --frames 9      # more frames in the sheet
    python render_check.py FILE.py SceneName --at 12 30 41   # frames at chosen seconds, no video
    python render_check.py FILE.py SceneName --still         # final frame only (fast layout check)
    python render_check.py FILE.py SceneName --draft         # MANIM_DRAFT=1: coarse surfaces, fast
    python render_check.py FILE.py SceneName --res 1920x1080 --fps 60   # final quality

Before rendering, if FILE.py has an `if __name__ == "__main__":` block it is run
as a preflight (e.g. revolve_lib's check()); a non-zero exit aborts the render.

Frames come from `manimgx inspect -t`, which runs the scene without encoding and draws
only the frames asked for, so --at and --still never render a video. For a 2D scene
ManimGX also checks the layout (text cut off by the frame, texts overlapping, a line or
fill through text, text too small) at every play's end, or at the --at frames; the
problems are printed and outlined in red on the pictures. 3D scenes get the pictures
only: look at them yourself.

Outputs go to ./render/<SceneName>/ : the .mp4, sheet.png (six frames per sheet, more
go to sheet-2.png, ...), storyboard.png (the frame at the end of every play), render.log.
Exit code is non-zero if rendering failed; the tail of the log is printed.
"""
import argparse, os, re, shutil, subprocess, sys, time

VENV_PY = os.path.expanduser("~/.manimgx-venv/bin/python")    # made by setup_env.sh


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def manimgx_python():
    """An interpreter that can import manimgx: this one, or setup_env.sh's venv."""
    for py in (sys.executable, VENV_PY):
        if os.path.exists(py) and run([py, "-c", "import manimgx"]).returncode == 0:
            return py
    print("[render] manimgx is not installed: run  bash scripts/setup_env.sh"); sys.exit(1)


def tail(log, n=15):
    return "\n".join(l[:300] for l in log.strip().splitlines()[-n:])


def report(text):
    """manimgx's layout check, as it printed it (problems are numbered like the red
    outlines on the pictures)."""
    keep = ("layout:", "  [", "  note:", "      [")
    return "\n".join("[render] " + l if l.startswith("layout:") else l
                     for l in text.splitlines() if l.startswith(keep))


def sheets(out, stem):
    """stem.png, stem-2.png, ... as written."""
    found = [f for f in os.listdir(out) if re.fullmatch(re.escape(stem) + r"(-\d+)?\.png", f)]
    found.sort(key=lambda f: int(re.search(r"-(\d+)\.png$", f).group(1)) if "-" in f else 1)
    return ", ".join(os.path.join(out, f) for f in found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file"); ap.add_argument("scene")
    ap.add_argument("--res", default="640x360")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--at", type=float, nargs="*")
    ap.add_argument("--still", action="store_true")
    ap.add_argument("--out", default="render")
    ap.add_argument("--timeout", type=int, default=3000)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--no-preflight", action="store_true")
    a = ap.parse_args()

    py = manimgx_python()
    src = os.path.abspath(a.file)
    work = os.path.dirname(src)
    out = os.path.abspath(os.path.join(a.out, a.scene))
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    env = dict(os.environ, MANIM_DRAFT="1" if a.draft else "0")
    gx = [py, "-m", "manimgx"]
    log = []

    if not a.no_preflight and '__name__ == "__main__"' in open(src).read():
        pf = run([py, src], cwd=work, env=env, timeout=600)
        txt = (pf.stdout + pf.stderr).strip()
        print("\n".join(l for l in txt.splitlines() if l.startswith("[")) or txt[-800:])
        if pf.returncode != 0:
            print("[render] preflight FAILED, not rendering"); sys.exit(1)

    def go(cmd, what):
        print(f"[render] {what}:", " ".join(cmd[2:]), "(draft)" if a.draft else "")
        t0 = time.time()
        try:
            r = run(cmd, cwd=work, env=env, timeout=a.timeout)
        except subprocess.TimeoutExpired:
            print("[render] TIMEOUT"); sys.exit(2)
        log.append(f"$ {' '.join(cmd)}\n{r.stdout}{r.stderr}")
        open(os.path.join(out, "render.log"), "w").write("\n".join(log))
        print(f"[render] took {time.time() - t0:.0f}s")
        # inspect exits 1 for layout problems: that is a finding, not a failure
        if r.returncode != 0 and "layout:" not in r.stdout:
            print("[render] FAILED. Last lines of the log:")
            print(tail(r.stdout + r.stderr)); sys.exit(1)
        return r.stdout

    inspect = gx + ["inspect", src, a.scene, "-o", os.path.join(out, "sheet.png"), "-t"]
    if a.still or a.at:
        times = "end" if a.still else ",".join(f"{t:g}" for t in a.at)
        print(report(go(inspect + [times], "frames")))
        print(f"[render] sheet : {sheets(out, 'sheet')}"
              f"  ({'last frame' if a.still else times + ' s'})")
        return

    mp4 = os.path.join(out, f"{a.scene}.mp4")
    said = go(gx + ["render", src, a.scene, "-r", a.res, "--fps", str(a.fps), "-o", mp4],
              "video")
    m = re.search(r"\.mp4\s+([\d.]+) s,", said)
    if not m or not os.path.exists(mp4):
        print("[render] no mp4 produced"); print(tail(said)); sys.exit(1)
    dur = float(m.group(1))
    for f in os.listdir(out):            # <Scene>.storyboard[-2].png -> storyboard[-2].png
        k = re.fullmatch(re.escape(a.scene) + r"\.(storyboard(-\d+)?\.png)", f)
        if k:
            os.replace(os.path.join(out, f), os.path.join(out, k.group(1)))
    times = [round(dur * (i + 0.5) / a.frames, 1) for i in range(a.frames)]
    go(inspect + [",".join(f"{t:g}" for t in times)], "sheet")
    print(report(said))                  # checked at every play's end, not just the sheet's
    print(f"[render] video : {mp4}  ({dur:.1f}s, {a.res} at {a.fps} fps)")
    print(f"[render] sheet : {sheets(out, 'sheet')}"
          f"  (frames at {', '.join(f'{t:.1f}' for t in times)} s)")
    print(f"[render] plays : {sheets(out, 'storyboard')}  (the frame at every play's end)")


if __name__ == "__main__":
    main()
