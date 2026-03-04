import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto

# iris landmark indices (mediapipe 478-point topology)
LEFT_IRIS,  RIGHT_IRIS  = [474,475,476,477], [469,470,471,472]
LEFT_EYE  = dict(top=386, bot=374, lft=362, rgt=263)
RIGHT_EYE = dict(top=159, bot=145, lft=33,  rgt=133)

# head pitch thresholds (degrees)
PITCH_PHONE   =  18.0   # leaning forward / looking down
PITCH_RECLINE = -20.0   # leaning back in chair

NO_FACE_FRAMES = 20     # frames before "face gone" triggers (~0.7s at 30fps)

MODEL_FILE = "face_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


class Reason(Enum):
    NONE    = auto()
    PHONE   = auto()
    RECLINE = auto()
    AWAY    = auto()


@dataclass
class GazeResult:
    triggered:  bool
    reason:     Reason
    iris_ratio: float   # 0 = looking up, 1 = looking down
    pitch_deg:  float


def _find_model():
    for p in [Path(__file__).parent / MODEL_FILE, Path.cwd() / MODEL_FILE]:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"'{MODEL_FILE}' not found. Run: python setup.py")


class GazeDetector:
    def __init__(self, iris_threshold=0.60, smoothing=4):
        self.iris_threshold = iris_threshold
        self._smoothing     = smoothing
        self._iris_hist     = []
        self._pitch_hist    = []
        self._no_face       = 0

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_find_model()),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_facial_transformation_matrixes=True,
        )
        self._detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr):
        """Returns (annotated_frame, GazeResult)"""
        h, w = frame_bgr.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self._detector.detect(mp_img)
        out = frame_bgr.copy()

        # no face
        if not res.face_landmarks:
            self._iris_hist.clear()
            self._pitch_hist.clear()
            self._no_face += 1
            triggered = self._no_face >= NO_FACE_FRAMES
            if triggered:
                cv2.putText(out, "no face", (w//2 - 40, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60,60,60), 1)
            return out, GazeResult(triggered, Reason.AWAY if triggered else Reason.NONE, 0.5, 0.0)

        self._no_face = 0
        lm = res.face_landmarks[0]

        def px(i): return int(lm[i].x * w), int(lm[i].y * h)

        # iris ratio
        ratios = []
        for iris, eye in [(LEFT_IRIS, LEFT_EYE), (RIGHT_IRIS, RIGHT_EYE)]:
            iris_y = np.mean([lm[i].y for i in iris])
            span   = lm[eye["bot"]].y - lm[eye["top"]].y
            if span > 1e-6:
                ratios.append((iris_y - lm[eye["top"]].y) / span)
        iris_raw = float(np.mean(ratios)) if ratios else 0.5

        self._iris_hist.append(iris_raw)
        if len(self._iris_hist) > self._smoothing: self._iris_hist.pop(0)
        iris = float(np.mean(self._iris_hist))

        # head pitch from transformation matrix
        pitch = 0.0
        if res.facial_transformation_matrixes:
            R = np.array(res.facial_transformation_matrixes[0]).reshape(4,4)[:3,:3]
            pitch = float(np.degrees(np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))))

        self._pitch_hist.append(pitch)
        if len(self._pitch_hist) > self._smoothing: self._pitch_hist.pop(0)
        pitch = float(np.mean(self._pitch_hist))

        # classify
        if pitch < PITCH_RECLINE:
            reason, triggered = Reason.RECLINE, True
        elif iris > self.iris_threshold or pitch > PITCH_PHONE:
            reason, triggered = Reason.PHONE, True
        else:
            reason, triggered = Reason.NONE, False

        # draw eye boxes
        color = {Reason.NONE:(0,220,100), Reason.PHONE:(0,70,255), Reason.RECLINE:(0,160,255)}[reason]
        for eye in [LEFT_EYE, RIGHT_EYE]:
            lx, _ = px(eye["lft"]); rx, _ = px(eye["rgt"])
            _, ty = px(eye["top"]); _, by = px(eye["bot"])
            x1 = min(lx,rx) - int(abs(rx-lx)*0.25)
            x2 = max(lx,rx) + int(abs(rx-lx)*0.25)
            y1 = min(ty,by) - int(abs(by-ty)*0.6)
            y2 = max(ty,by) + int(abs(by-ty)*0.6)
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)

        return out, GazeResult(triggered, reason, iris, pitch)

    def release(self):
        self._detector.close()
