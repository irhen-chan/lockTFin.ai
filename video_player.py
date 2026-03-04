import cv2
import numpy as np
import time
import tempfile
import os
import subprocess
import threading
from pathlib import Path

try:
    import pygame.mixer as mixer
    _PYGAME = True
except ImportError:
    _PYGAME = False

try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    _FFMPEG = None

WIN_NAME  = "lock tf in . ai"
WIN_W, WIN_H = 270, 480     # 9:16 portrait


class VideoPlayer:
    def __init__(self, video_paths=None):
        self._paths = [p for p in (video_paths or []) if Path(p).exists()]
        self._index = 0
        self._cap   = None
        self._wavs  = {}      # path -> temp wav file
        self.visible = False
        self._fps    = 30.0
        self._next   = 0.0
        self._ph_t   = 0.0   # placeholder animation timer

        if _PYGAME:
            try:
                mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            except Exception as e:
                print(f"[warn] audio init failed: {e}")

        if _FFMPEG and _PYGAME and self._paths:
            threading.Thread(target=self._preextract, daemon=True).start()
        elif not _FFMPEG:
            print("[warn] imageio-ffmpeg not found — video audio disabled")
            print("       fix: pip install imageio-ffmpeg")

    def show(self, x=20, y=80):
        if self.visible:
            return
        self.visible = True
        self._open()
        self._play_audio()
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)
        cv2.moveWindow(WIN_NAME, x, y)
        self._next = time.time()

    def hide(self):
        if not self.visible:
            return
        self.visible = False
        if _PYGAME:
            try: mixer.music.stop()
            except: pass
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._paths:
            self._index = (self._index + 1) % len(self._paths)
        try: cv2.destroyWindow(WIN_NAME)
        except: pass

    def update(self):
        if not self.visible:
            return
        now = time.time()
        if now < self._next:
            return
        frame = self._read()
        if frame is not None:
            cv2.imshow(WIN_NAME, frame)
        self._next = now + (1.0 / self._fps)

    def clip_name(self):
        return Path(self._paths[self._index]).name if self._paths else "placeholder"

    def release(self):
        self.hide()
        for w in self._wavs.values():
            try: os.unlink(w)
            except: pass

    # ── internals ────────────────────────────────────────────────────────

    def _preextract(self):
        for p in self._paths:
            self._get_wav(p)

    def _get_wav(self, path):
        if path in self._wavs:
            return self._wavs[path]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            r = subprocess.run(
                [_FFMPEG, "-y", "-i", path,
                 "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                 tmp.name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
            )
            if r.returncode == 0 and os.path.getsize(tmp.name) > 1000:
                self._wavs[path] = tmp.name
                print(f"[ok] audio ready: {Path(path).name}")
                return tmp.name
        except Exception as e:
            print(f"[warn] audio extract failed ({Path(path).name}): {e}")
        try: os.unlink(tmp.name)
        except: pass
        return None

    def _play_audio(self):
        if not _PYGAME or not self._paths:
            return
        wav = self._get_wav(self._paths[self._index])
        if not wav:
            return
        try:
            mixer.music.load(wav)
            mixer.music.set_volume(1.0)
            mixer.music.play()
        except Exception as e:
            print(f"[warn] playback failed: {e}")

    def _open(self):
        if self._cap:
            self._cap.release()
        if not self._paths:
            return
        self._cap = cv2.VideoCapture(self._paths[self._index])
        if self._cap.isOpened():
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._fps = fps if fps > 0 else 30.0

    def _read(self):
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if _PYGAME:
                    try: mixer.music.rewind(); mixer.music.play()
                    except: pass
                ret, frame = self._cap.read()
            if ret:
                return _crop_portrait(frame)
        self._ph_t += 0.04
        return _placeholder(self._ph_t)


def _crop_portrait(frame):
    h, w = frame.shape[:2]
    target = WIN_W / WIN_H
    if (w / h) > target:
        nw = int(h * target)
        frame = frame[:, (w-nw)//2:(w-nw)//2+nw]
    else:
        nh = int(w / target)
        frame = frame[(h-nh)//2:(h-nh)//2+nh, :]
    return cv2.resize(frame, (WIN_W, WIN_H))


def _placeholder(t):
    img = np.full((WIN_H, WIN_W, 3), (15, 15, 25), dtype=np.uint8)
    p   = 0.5 + 0.5 * np.sin(t * 2)
    cx, cy = WIN_W // 2, WIN_H // 2 - 30
    cv2.circle(img, (cx, cy), int(55 + 8*np.sin(t*2)), (0, int(200*p), int(80*p)), 2)
    for text, yf, col in [
        ("LOCK TF IN.",    0.62, (255, 255, 255)),
        ("put the phone",  0.75, (120, 120, 140)),
        ("down.",          0.82, (0, int(200*p), int(100*p))),
    ]:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.putText(img, text, ((WIN_W-tw)//2, int(WIN_H*yf)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, col, 1, cv2.LINE_AA)
    return img
