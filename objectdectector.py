from pathlib import Path
import time

import cv2
import torch
from ultralytics import YOLOE



OBJECTS = [
    "person",
    "hand",
    "door",
    "window",
    "wardrobe",
    "cupboard",
    "chair",
    "table",
    "sofa",
    "bed",
    "pillow",
    "blanket",
    "curtain",
    "vase",
    "potted plant",
    "shirt",
    "trousers",
    "shoes",
    "bag",
    "bottle",
    "cup",
    "plate",
    "book",
    "mobile phone",
    "laptop",
    "television",
    "remote control",
    "wall clock",
    "mirror",
    "ceiling fan",
    "air cooler",
    "refrigerator",
]

CAMERA_INDEX = 0
IMAGE_SIZE = 640
CONFIDENCE = 0.35
WINDOW = "Home Object Detector | Q = Quit"


def main():
    folder = Path(__file__).resolve().parent


    model_file = folder / "yoloe-26s-seg.pt"

    print("Loading model...")
    print("First setup needs internet and can take several minutes.")

    model = YOLOE(str(model_file))


    print("Preparing household object names...")
    model.set_classes(OBJECTS)


    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")

    camera = cv2.VideoCapture(CAMERA_INDEX)

    try:
        if not camera.isOpened():
            raise RuntimeError(
                "Camera unavailable. Check Windows camera permissions, "
                "close other camera apps, or try CAMERA_INDEX = 1."
            )

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

        print("Click the camera window and press Q to exit.")
        print("Frames are processed locally; this script does not record them.")

        while True:
            ok, frame = camera.read()

            if not ok:
                print("Camera frame unavailable. Stopping.")
                break

            start = time.perf_counter()

            result = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device=device,
                verbose=False,
                save=False,
            )[0]


            display = result.plot(masks=False)

            elapsed = time.perf_counter() - start
            processing_fps = 1.0 / max(elapsed, 0.000001)

            cv2.putText(
                display,
                f"Processing FPS: {processing_fps:.1f} | Q: Quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.imshow(WINDOW, display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if cv2.getWindowProperty(
                WINDOW, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception:
        import traceback
        traceback.print_exc()
