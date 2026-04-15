"""
facefind.py -- Face detection demo (MDview series)
  SPACE  -> capture frame and detect faces (bounding boxes)
  ESC    -> quit

Uses OpenCV's built-in Haar Cascade classifier -- no Ollama required.
"""

import cv2

# ── config ───────────────────────────────────────────────────────────────────
BOX_COLOR      = (0, 255, 80)    # green bounding boxes
BOX_THICKNESS  = 2
LABEL_SCALE    = 0.55
LABEL_COLOR    = (0, 255, 80)
STATUS_SCALE   = 0.6
STATUS_COLOR   = (255, 255, 255)
STATUS_BG      = (30, 30, 30)

# detection tuning
SCALE_FACTOR   = 1.1   # how much the image size is reduced at each scale
MIN_NEIGHBORS  = 5     # higher = fewer false positives
MIN_FACE_PX    = 60    # minimum face size in pixels
# ─────────────────────────────────────────────────────────────────────────────


def load_detector() -> cv2.CascadeClassifier:
    """Load the frontal-face Haar cascade bundled with OpenCV."""
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError(f"Could not load cascade from: {path}")
    return detector


def detect_faces(frame, detector: cv2.CascadeClassifier):
    """Return list of (x, y, w, h) face rectangles."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return detector.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=(MIN_FACE_PX, MIN_FACE_PX),
    )


def draw_status(frame, text: str) -> None:
    """Draw a status bar at the top of the given frame."""
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 32), STATUS_BG, -1)
    cv2.putText(frame, text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, STATUS_SCALE, STATUS_COLOR, 1, cv2.LINE_AA)


def draw_boxes(frame, faces) -> None:
    """Draw bounding boxes and face index labels on the frame."""
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)
        label = f"#{i + 1}"
        cv2.putText(frame, label, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_COLOR, 1, cv2.LINE_AA)


def main():
    detector = load_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    result_frame = None   # last captured frame with boxes drawn
    face_count   = 0
    status       = "SPACE -> detect faces  |  ESC -> quit"

    print("Windows open. SPACE to capture, ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- camera window ---
        live = frame.copy()
        draw_status(live, status)
        cv2.imshow("facefind -- Camera", live)

        # --- result window ---
        if result_frame is not None:
            cv2.imshow("facefind -- Detections", result_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:       # ESC
            break
        elif key == 32:     # SPACE
            captured = frame.copy()
            faces = detect_faces(captured, detector)
            face_count = len(faces)
            draw_boxes(captured, faces)

            noun = "face" if face_count == 1 else "faces"
            label = f"{face_count} {noun} found  |  SPACE -> capture again  |  ESC -> quit"
            draw_status(captured, label)

            result_frame = captured
            status = f"Last capture: {face_count} {noun}  |  SPACE -> capture again  |  ESC -> quit"

            print(f"Detected {face_count} {noun}.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
