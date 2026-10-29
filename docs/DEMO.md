# The live demo

## The GIF in the README
`docs/demo/live_demo.gif` is a 3.5-second excerpt of a **phone recording of the live detector
window**, made during the author's live sessions on his own workstation in August 2026. In that
session the C920 webcam was pointed at a **third-party video of a real street fight (from YouTube)
playing on another screen**, and `live_camera.py` classified what the webcam saw. The excerpt shows
the overlay going from *Non-Violence* to *Violence*, the run of violent clips growing, **VIOLENCE
ALERT** appearing, and the prediction returning to *Non-Violence*.

**What was done to the recording, and why:**
- **The third-party video is blurred** (Gaussian, sigma 7) so that the action is visible but no one
  in it can be identified, and its burned-in caption strip is blurred until unreadable (sigma 18).
  Those people did not agree to appear here, and the video is someone else's copyright. In the last
  ten frames the camera is right on top of one person: at sigma 7 a face-like shape remained, so
  those frames, caption strip included, are blurred much harder (sigma 30) and show only colour.
- **The detector's overlay keeps its original meaning; nothing was retyped or redrawn.** Only its
  own white text pixels are retained, so a few letters are missing where the original text sat on a
  white shirt and had no contrast. Every frame was checked against the original: the prediction and
  **VIOLENCE ALERT** lines are faithful throughout. In 7 of the 52 frames the probability value (and
  in the last frame the clip count) could not be recovered cleanly, so it is **left blank** rather
  than shown partially, where it could read as a different number.
- **The handheld shot is stabilised and cropped to the detector window**, so the desk, phone and
  room are not shown. There is no audio, and the file carries no metadata blocks beyond the
  standard GIF loop setting (the export audit checks this).
- **It plays faster than real time**, as the phone recording does: the consecutive-clip count
  rises far faster than the live loop allows (about 4 clips per second at 30 fps). The GIF keeps the
  recording's own speed and its label says so.

**It is not evidence of accuracy**; for accuracy, see the results table. The original recording
is kept privately and is not in this repository.

## Recording a new demo
A new recording should follow these rules:
- **People:** only the author, or people who have agreed to appear.
- **No screens, films or online videos in view**, so no blurring is needed. Dataset videos are
  excluded too, because RWF-2000 does not allow redistribution (see [DATA.md](../DATA.md)).
- **Location:** a plain background with nothing that identifies a place.
- **Audio:** none. The recording below has no audio track.

### 1. Record
Run this from the repository root, with your trained `best_frozen_model.pth` in place:

```bash
python live_camera.py --no-gemma --record outputs/live_demo.mp4
```

- `--record` saves exactly what the window shows: the camera image with the overlay. The
  desktop, other windows and notifications are not captured.
- Demo alerts never touch `results/live/live_alerts.csv`, the default alert log (in the
  author's private archive it holds the 12–15 Aug 2026 dissertation sessions). With `--record`
  they go to `outputs/demo_alerts.csv` by default, and pointing `--alert-log` at the default
  log is refused before the model or camera is opened.
  `tests/test_live_options.py` and `tests/test_live_camera_headless.py` check this.
- `--no-gemma` skips the incident report. That avoids the pause while Gemma writes it, and
  Ollama doesn't need to be running.
- `outputs/` is git-ignored, so the raw recording can't be committed by accident.

Press **Q** to stop. The script prints how many frames it saved and the rate the loop actually
ran at. If that rate is well below the camera's, the saved video plays faster than real time.
Correct it when making the GIF by adding `setpts=<camera fps / loop fps>*PTS,` to the start of
the filter.

### 2. Make the GIF
The GIF is trimmed to the interesting part (`-ss` start and `-t` length, in seconds), 12 fps,
480 px wide, with no audio and no metadata:

```bash
mkdir -p docs/demo
ffmpeg -ss 0 -t 10 -i outputs/live_demo.mp4 -an -map_metadata -1 \
  -vf "fps=12,scale=480:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
  -loop 0 docs/demo/live_demo.gif
```

Aim for under 5 MB. If it's larger, shorten `-t` or reduce `fps`.

### 3. Caption
State the real recording date, and do not imply that the clip measures performance:

> Staged demo by the author, recorded *&lt;date&gt;* with a Logitech C920. The overlay shows each
> clip's prediction and probability, and the run of consecutive violent clips. The alert appears
> after 3 in a row. This is not an accuracy measure; see *Results*.
