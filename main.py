"""
lock tf in . ai

Watches your webcam. The moment your eyes drop to phone level (or you recline
your chair, or leave the frame), a meme video fires on the left of your screen.
Look back for 2 seconds and it stops.

Usage:
    python main.py
    python main.py --videos meme1.mp4 meme2.mp4 meme3.mp4
    python main.py --cam 1 --threshold 0.58
"""

import argparse
import sys
import time
import os
import cv2
import numpy as np

from gaze_detector import GazeDetector, Reason
from video_player  import VideoPlayer

# window layout
CAM_WIN  = "lock tf in . ai  |  Q to quit"
CAM_W, CAM_H = 800, 540
CAM_X, CAM_Y = 310, 80   # offset right so the meme panel fits on the left
VID_X, VID_Y = 20,  80

# timing
TRIGGER_DELAY = 0.25   # seconds of bad gaze before triggering
COOLDOWN      = 2.0    # seconds of good gaze before dismissing

LABELS = {
    Reason.PHONE:   ("PHONE DETECTED",    (0, 70, 255)),
    Reason.RECLINE: ("RECLINING",         (0, 150, 255)),
    Reason.AWAY:    ("WHERE ARE YOU",     (60, 60, 60)),
    Reason.NONE:    ("locked in",         (0, 220, 100)),
}


def draw_hud(frame, result, sessions, phone_secs, cooldown_left):
    h, w = frame.shape[:2]

    # top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 52), (8, 8, 18), -1)
    cv2.addWeighted(bar, 0.75, frame, 0.25, 0, frame)

    label, color = LABELS[result.reason]
    cv2.putText(frame, label, (12, 34), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1, cv2.LINE_AA)

    # top right: debug values
    cv2.putText(frame, f"iris {result.iris_ratio:.2f}  pitch {result.pitch_deg:+.0f}d",
                (w - 210, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,100), 1, cv2.LINE_AA)

    # bottom: session stats
    m, s = divmod(int(phone_secs), 60)
    cv2.putText(frame, f"caught {sessions}x   phone time {m:02d}:{s:02d}",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100,100,120), 1, cv2.LINE_AA)

    # cooldown bar
    if cooldown_left > 0:
        bw = 140
        cv2.rectangle(frame, (12, h-36), (12+bw, h-30), (30,30,40), -1)
        fill = int(bw * (1 - cooldown_left / COOLDOWN))
        cv2.rectangle(frame, (12, h-36), (12+fill, h-30), (0,180,80), -1)

    return frame


def main():
    ap = argparse.ArgumentParser(description="lock tf in . ai")
    ap.add_argument("--videos",    nargs="+", default=None,  help="meme mp4 files (up to 3)")
    ap.add_argument("--cam",       type=int,  default=0,     help="webcam index")
    ap.add_argument("--threshold", type=float,default=0.60,  help="iris threshold (default 0.60)")
    ap.add_argument("--cooldown",  type=float,default=COOLDOWN, help="dismiss delay in seconds")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open camera {args.cam}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(CAM_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAM_WIN, CAM_W, CAM_H)
    cv2.moveWindow(CAM_WIN, CAM_X, CAM_Y)

    detector = GazeDetector(iris_threshold=args.threshold)
    player   = VideoPlayer(video_paths=args.videos)

    active         = False
    down_since     = None
    up_since       = None
    sessions       = 0
    phone_secs     = 0.0
    session_start  = None
    last_reason    = Reason.NONE

    print(f"\n  lock tf in . ai")
    print(f"  ───────────────")
    print(f"  cam       : {args.cam}")
    print(f"  videos    : {len(args.videos or [])} loaded")
    print(f"  threshold : {args.threshold}")
    print(f"  cooldown  : {args.cooldown}s")
    print(f"  Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)
        annotated, gaze = detector.process(frame)
        now = time.time()

        if gaze.triggered:
            up_since = None
            if down_since is None:
                down_since = now
            last_reason = gaze.reason

            if not active and (now - down_since) >= TRIGGER_DELAY:
                active = True
                sessions += 1
                session_start = now
                player.show(x=VID_X, y=VID_Y)
        else:
            down_since = None
            if up_since is None:
                up_since = now

            if active and (now - up_since) >= args.cooldown:
                active = False
                if session_start:
                    phone_secs += now - session_start
                    session_start = None
                player.hide()
                last_reason = Reason.NONE

        live_secs = phone_secs + (now - session_start if active and session_start else 0)

        cooldown_left = 0.0
        if not gaze.triggered and up_since and active:
            cooldown_left = max(0.0, args.cooldown - (now - up_since))

        # keep showing the trigger reason while active even if gaze briefly clears
        display = gaze
        if active and gaze.reason == Reason.NONE:
            from dataclasses import replace
            display = replace(gaze, reason=last_reason)

        annotated = draw_hud(annotated, display, sessions, live_secs, cooldown_left)
        cv2.imshow(CAM_WIN, cv2.resize(annotated, (CAM_W, CAM_H)))
        player.update()

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    detector.release()
    player.release()
    cv2.destroyAllWindows()

    m, s = divmod(int(phone_secs), 60)
    print(f"\n  caught {sessions}x  |  total phone time {m:02d}:{s:02d}\n")


if __name__ == "__main__":
    main()
