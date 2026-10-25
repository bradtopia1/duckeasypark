# carpi

An in-car camera rig (target: Raspberry Pi 3B + camera) that watches for cars
driving past, crops them out of the frame, and decides whether the car belongs to
a **parking inspector**.

The pipeline:

```
video source ──► motion detection ──► save motion clip
                     (OpenCV MOG2)          │
                                            ▼
                              car detection (AI)  ──► crop the car
                                            │
                                            ▼
                        parking-inspector classification (AI)
                                            │
                                            ▼
                                   action  ◄── (not decided yet)
```

Only **test mode** is implemented so far: there is no camera and no live feed —
frames come from a video file named in the config.

## Requirements

- Python 3.9+
- `numpy`, `PyYAML`, `opencv-python-headless` (see `requirements.txt`)
- For the real inspector model: `tflite-runtime` (Pi) or `tensorflow` (dev)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Quick start (test mode)

Test mode reads a video file - **you supply the video**. Point
`modes.test.video_path` in `config.yaml` at your recording, then run:

```bash
.venv/bin/python main.py -c config.yaml -v
```

No video handy yet? Generate a synthetic clip to try things out:

```bash
# writes samples/test.mp4 (the default video_path)
.venv/bin/python scripts/make_sample_video.py
```

You should see a log line for each motion event, and files under `output/`:

```
output/
└── clips/        # the recorded motion clips (*.mp4)
```

By default the app is **motion-only** — the AI stages are disabled
(`detectors.*.backend: none`), so nothing is classified. Enable them (below)
and `cars/` and `inspectors/` appear too.

## Command line

```
python main.py [-c CONFIG] [-m MODE] [--preview | --no-preview] [-v|-vv]
```

| flag | meaning |
|------|---------|
| `-c, --config` | path to the YAML config (default `config.yaml`) |
| `-m, --mode`   | override the mode set in the config |
| `--preview` / `--no-preview` | force the debug preview window on/off |
| `-v`, `-vv`    | info / debug logging |

### Watching a run

With `-v` the pipeline prints a periodic status line while it works, so you can
tell a headless rig is still moving and keeping up (interval:
`logging.status_interval_seconds`, default 30s; `0` disables it):

```
12:00:14 INFO    carpi.pipeline: status: 1800 frames in 30.1s | now 59.8 fps, avg 59.8 fps | source 30.00 fps -> 2.0x, keeping up | 3 motion event(s), 1 car(s), 0 inspector(s)
```

`now` is the rate over the last interval and `avg` the whole run so far; the
multiplier is `now` against the source's own FPS, so `1.0x` means real time. A
file run that is not `realtime` normally shows well above `1.0x` — that is
expected, not a problem. On a live feed it is the number to watch: if it sits
below `1.0x`, the pipeline is falling behind the camera and events will start
being missed.

Together with the per-event lines and the `done: ...` summary at exit, that is
the whole picture of a run; on the Pi it all goes to the journal
(`journalctl -u carpi -f`). The same lines are also written to a log file,
`carpi.log`, in the folder you run from (rotated at ~1 MB, keeping 3 backups, so
an always-on rig won't fill the disk).

## Configuration

Everything lives in `config.yaml` (documented inline). The two settings that
apply to **every** mode are the motion ones:

- **`motion.sensitivity`** (0.0–1.0) — higher means fainter/smaller movement
  triggers detection. Internally it lowers the background-subtraction threshold
  and shrinks the minimum blob size.
- **`motion.area`** — the region of interest, as percentages of the frame.
  Each axis takes either a single number (a centred band of that size) or an
  explicit `[start, end]` span:

  ```yaml
  motion:
    area:
      vertical: [10, 100]   # ignore the top 10% of the frame
      horizontal: 80        # a centred band covering 80% of the width
  ```

Other useful knobs: `motion.min_area_percent`, `motion.trigger_frames`,
`motion.clip.pre_seconds` / `post_seconds`, and `output_dir`.

### Processing resolution

Each stage runs at its own resolution, set by three independent knobs (`null`
means "keep the source resolution"). They are separate because the stages want
different things:

| key | controls | sensible value |
|-----|----------|----------------|
| `processing.motion_width` | resolution of motion detection | small — `640`, or `320`–`480` if CPU-bound |
| `processing.detect_width` | resolution the car detector sees | as large as the model allows — `640`–`1280` |
| `processing.clip_width` | resolution clips are written at | `null` (full res) when clips are training data |

Motion detection only needs to be a coarse trigger, so it runs smallest: lower is
much faster (roughly with the square of the reduction) and uses less RAM, and too
low only matters if small/distant movement gets missed. The car detector is the
opposite — small/distant cars need pixels — so it can run larger. And clips are
worth keeping at full resolution whenever they double as training data, because
you can always downscale later but never upscale.

```yaml
processing:
  motion_width: 640     # cheap trigger
  detect_width: 640     # car-detector input
  clip_width: null      # null = save clips at the source resolution
  full_res_crops: true  # crop cars from the original full-res frame
```

Cost scales roughly with the pixel count, so 1080p → 640 is ~9x fewer pixels. On
a 1080p30 source (measured, motion-only, one core):

| `motion_width` | throughput | vs real-time | peak RAM |
|---------|-----------|--------------|----------|
| `null` (1920) | ~31 fps | ~1x | ~640 MB |
| `640` | ~64 fps | ~2x | ~270 MB |
| `320` | ~92 fps | ~3x | ~240 MB |

`full_res_crops: true` keeps car crops sharp for the inspector model even though
detection runs small. A Pi 3B is roughly an order of magnitude slower than the
desktop these numbers came from, so lower `motion_width` (and a lower camera
capture resolution) matters a lot there.

> **Full-res clips cost RAM.** The pre-roll buffer holds frames at the clip
> resolution, so `clip_width: null` buffers full-resolution frames (~6 MB each at
> 1080p, versus ~0.7 MB at 640) — a real concern on a 1 GB Pi. Lower
> `motion.clip.pre_seconds` if you need to.

### Does it run in real time?

Not by default, and it isn't meant to for files. `modes.test.realtime: false`
processes as fast as the CPU allows — which for a big clip can be *slower* than
the video's own runtime. Set `realtime: true` to pace playback to the source FPS
(useful to simulate a live feed). The main speed levers, in order of impact:

1. `processing.motion_width` (above) — the big one.
2. `modes.test.max_seconds` — only process the first N seconds.
3. `motion.area` — a smaller region of interest means less work per frame.
4. `motion.clip.pre_seconds` / `post_seconds` — shorter clips encode faster.

### Modes

`mode:` selects the runner. Currently only `test` exists:

```yaml
modes:
  test:
    video_path: samples/test.mp4
    loop: false
    realtime: false
    show_preview: false
    max_seconds: null
```

> **Preview note:** the preview window needs a GUI build of OpenCV. The headless
> wheel used on a Pi cannot open a window — the pipeline detects that and just
> carries on without it.

## AI backends

Both AI stages sit behind small interfaces. Each is **disabled by default**
(`backend: none`), so out of the box the app only records motion clips; the
mocks and real models are switched on through config alone.

### Car detection — `detectors.car`

| backend | status |
|---------|--------|
| `none`  | ✅ disabled (default) — no car detection, clips only |
| `mock`  | ✅ fake: promotes the motion region to a car box (no model needed) |
| `opencv_dnn` | ⏳ planned (e.g. a MobileNet-SSD ONNX/TF model) |
| `yolo`  | ⏳ planned (Ultralytics) |

### Inspector classification — `detectors.inspector`

| backend | status |
|---------|--------|
| `none`  | ✅ disabled (default) |
| `mock`  | ✅ fake: seeded coin-flip, `positive_probability` (1.0 flags *every* car) |
| `tflite`| ✅ ready — point it at your trained model |

> The mocks are fake by construction: the mock car detector calls every motion
> blob a car, and the mock inspector is a coin-flip. They exist to exercise the
> pipeline, not to detect anything real.

Train an image classifier in TensorFlow/Keras, export it to `.tflite`, then:

```yaml
detectors:
  inspector:
    backend: tflite
    options:
      model_path: models/inspector.tflite
      labels: [not_inspector, inspector]
      input_size: [224, 224]
      normalize: "0-1"          # "0-1" | "-1-1" | "none"
```

`labels[0]` should be the "not an inspector" class and `labels[1]` the inspector
class; the label named `inspector` is what counts as a positive hit.

## Where to extend

- **The action** when an inspector is found — `default_action_hook()` in
  `carpi/pipeline.py` (notification, recording, alarm, upload, ...).
- **A real car detector** — add a class in `carpi/detectors/` implementing
  `CarDetector.detect()` and register it in `detectors/__init__.py`.
- **A live camera mode** — implement `CameraSource` in `carpi/video.py`
  (with `picamera2` on the Pi) and add a runner in `carpi/modes/`.

## Deploy on a Raspberry Pi (headless, starts at boot)

On Raspberry Pi OS the way to add something to "startup items" without a desktop
login is a **systemd service**. A ready-made unit lives in
[`deploy/carpi.service`](deploy/carpi.service); it runs
`main.py` with the venv interpreter, restarts the program if it ever exits, and
sends all output to the system journal.

> **Not a live watcher yet.** Only `test` mode exists, and it processes a video
> file to completion and exits — so today the service just replays
> `modes.test.video_path` and systemd restarts it every few seconds. It becomes
> a real always-on camera watcher once a live mode is implemented (see the
> Roadmap); the unit itself won't need to change.

### 1. Put the code and a fresh venv on the Pi

```bash
# on the Pi
git clone <your-repo-url> ~/carpi   # or copy the folder across
cd ~/carpi
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Do **not** copy your desktop `.venv/` across: it is built for x86_64 and its
compiled extensions won't import on the Pi's ARM CPU. Build a new one on the Pi,
as above. On Raspberry Pi OS install the venv support first if it's missing:
`sudo apt install -y python3-venv`.

### 2. Edit the config and the unit

- In `config.yaml`: set `mode:` to the mode you want the service to run, and
  point `modes.test.video_path` at a file that exists on the Pi (or generate one
  with `scripts/make_sample_video.py`).
- In `deploy/carpi.service`: change `User=` and the two `/home/pi/...`
  paths if your username or install location differs. `WorkingDirectory` must be
  the project folder, because the relative `-c config.yaml` resolves against it.

### 3. Install, enable, and start it

```bash
cd ~/carpi
sudo cp deploy/carpi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now carpi
```

`enable` is what makes it start on every boot; `--now` also starts it right away.

### 4. Check on it

```bash
systemctl status carpi          # is it running?
journalctl -u carpi -f          # follow the logs (Ctrl-C to stop watching)
sudo systemctl stop carpi       # stop it now
sudo systemctl disable carpi    # stop starting it at boot
```

After editing the unit file, re-run `sudo systemctl daemon-reload` for the
change to take effect.

## Project layout

```
main.py                       CLI entry point
config.yaml                   configuration
deploy/carpi.service          systemd unit for autostart on a Pi
scripts/make_sample_video.py  synthetic test-clip generator
carpi/
  config.py                   typed config loading/validation
  video.py                    FrameSource: FileSource (test) + CameraSource stub
  motion.py                   MOG2 motion detection + ROI + sensitivity
  crop.py                     bounding-box cropping
  pipeline.py                 orchestrates the stages, saves outputs, action hook
  detectors/                  CarDetector interface + mock
  classifiers/                InspectorClassifier interface + mock + tflite
  modes/                      mode registry + test mode
```

## Roadmap

- [ ] live camera mode on the Pi (`picamera2`)
- [ ] real car detection (OpenCV DNN or YOLO)
- [ ] true background removal (segmentation) instead of a bounding-box crop
- [ ] train + wire in the parking-inspector model
- [ ] decide and implement the inspector action
