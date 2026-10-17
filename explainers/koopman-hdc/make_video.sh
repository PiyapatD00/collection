#!/usr/bin/env bash
# Render every chapter, then join them into one MP4.
#
#   bash make_video.sh                 # 1920x1080 at 30 fps (final)
#   bash make_video.sh 640x360 30      # quick draft
#   JOBS=1 bash make_video.sh          # render one scene at a time
#
# Each scene goes to render/<Scene>/ (video, contact sheet, storyboard, log);
# the joined film is video/koopman_hdc_explainer.mp4.
set -euo pipefail
cd "$(dirname "$0")"
RES=${1:-1920x1080}
FPS=${2:-30}
JOBS=${JOBS:-2}
SCENES=(
  "part0_math.py MathWarmup"
  "part1_koopman.py KoopmanExactMap"
  "part1_koopman.py KoopmanLift3D"
  "part1_koopman.py KoopmanEDMD"
  "part1_koopman.py KoopmanPendulum"
  "part2_hdc.py HDCBasics"
)

python3 models.py                     # every number on screen, checked first

printf '%s\n' "${SCENES[@]}" |
  xargs -P "$JOBS" -L 1 sh -c \
    'python3 render_check.py "$0" "$1" --res '"$RES"' --fps '"$FPS"' --timeout 14400 > "render.$1.out" 2>&1 \
     && echo "[make] $1 done" || { echo "[make] $1 FAILED (see render.$1.out)"; exit 255; }'

mkdir -p video
: > video/list.txt
for s in "${SCENES[@]}"; do
  set -- $s
  echo "file '../render/$2/$2.mp4'" >> video/list.txt
done
ffmpeg -loglevel error -y -f concat -safe 0 -i video/list.txt -c copy video/koopman_hdc_explainer.mp4
rm video/list.txt render.*.out
echo "[make] video/koopman_hdc_explainer.mp4 ($(ffprobe -v error -show_entries format=duration -of csv=p=0 video/koopman_hdc_explainer.mp4) s)"
