# ObjectDetector

HOUSEHOLD OBJECT DETECTOR

A live webcam application built with Python, OpenCV, PyTorch, and pretrained
Ultralytics YOLOE (yoloe-26s-seg.pt). No custom model training is performed.

FEATURES
- 32 configurable object prompts, including people, doors, chairs and laptops.
- Bounding boxes, labels, confidence scores and processing-FPS display.
- CUDA selection when available; CPU otherwise.
- Local frame processing without video recording.

INSTALL
Use your PyCharm or VS Code project's Python environment:
python -m pip install "ultralytics>=8.4.0" opencv-python torch

RUN
Open the detector's Python file in your IDE and click Run.
Internet is needed for initial model and text-encoder downloads.
Allow camera access. Click the video window and press Q or Escape to quit.

SETTINGS
CAMERA_INDEX = 0: camera selection; try 1 for another camera.
IMAGE_SIZE = 640: try 320 for faster processing, with possible accuracy loss.
CONFIDENCE = 0.35: minimum detection score.
OBJECTS: edit the requested object names.

NOTES
Detection depends on lighting, viewpoint and hardware. Requested labels are
not guaranteed detections. Processing FPS measures prediction and annotation,
not the complete capture/display loop. Segmentation masks are hidden.
If the camera fails, check permissions and close other camera applications.
This version stops on errors; it does not automatically retry camera access.

MODEL CREDIT
Ultralytics YOLOE: https://docs.ultralytics.com/models/yoloe/
Model/dependency licensing: https://www.ultralytics.com/license
