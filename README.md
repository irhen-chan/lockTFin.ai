# lockTFin.ai
Helps you lock in

Webcam app that catches you looking at your phone and plays a meme video until you stop.

Uses MediaPipe iris tracking + head pose estimation. Detects:
- **phone** — eyes looking down
- **recline** — head tilted back (chair reclined)  
- **away** — face not in frame

---

## setup

```bash
pip install -r requirements.txt
python setup.py        # downloads the face tracking model (~4 MB, once)
```

## run

```bash
# no videos — shows placeholder screen
python main.py

# with your memes (plays in order, loops)
python main.py --videos meme1.mp4 meme2.mp4 meme3.mp4
```

## options

| flag | default | description |
|---|---|---|
| `--videos` | none | mp4 files to play as punishment |
| `--cam` | `0` | webcam index |
| `--threshold` | `0.60` | iris sensitivity (lower = triggers easier) |
| `--cooldown` | `2.0` | seconds to look away before video closes |

## how it works

```
gaze_detector.py   iris ratio + head pitch → triggered / reason
video_player.py    cv2 for frames, ffmpeg → wav → pygame for audio
main.py            state machine + HUD
```

Eye box is **green** when locked in, **red/orange** when caught.

```
