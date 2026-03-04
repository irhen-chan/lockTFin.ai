"""
Downloads the MediaPipe face landmarker model (~4 MB).
Run once before using the app.
"""

import urllib.request, sys
from pathlib import Path

URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEST = Path(__file__).parent / "face_landmarker.task"

if DEST.exists():
    print(f"already downloaded: {DEST}")
    sys.exit(0)

print("downloading face_landmarker.task (~4 MB)...")

def _progress(n, chunk, total):
    if total > 0:
        pct = min(n * chunk / total * 100, 100)
        print(f"\r  {pct:.1f}%", end="", flush=True)

try:
    urllib.request.urlretrieve(URL, DEST, reporthook=_progress)
    print(f"\ndone → {DEST}")
except Exception as e:
    print(f"\nfailed: {e}")
    print(f"download manually from:\n  {URL}")
    sys.exit(1)
