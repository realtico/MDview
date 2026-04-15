"""
MDview -- Moondream live camera demo
  SPACE  -> capture frame and request description via Ollama
  ESC    -> quit
"""

import base64
import textwrap
import threading

import cv2
import ollama

# ── config ───────────────────────────────────────────────────────────────────
MODEL        = "moondream"   # change to "moondream2" if needed
PROMPT       = "Describe what you see in this image in detail."
FONT         = cv2.FONT_HERSHEY_SIMPLEX

# camera window
STATUS_SCALE = 0.6
STATUS_COLOR = (255, 255, 255)
STATUS_BG    = (30, 30, 30)

# description window
DESC_WIN     = "MDview -- Description"
DESC_W       = 680
DESC_LINE_H  = 30
DESC_SCALE   = 0.65
DESC_COLOR   = (230, 230, 230)
DESC_BG      = (20, 20, 20)
DESC_PADDING = 20
WRAP_CHARS   = 62            # characters per line in the description window
# ──────────────────────────────────────────────────────────────────────────────


def describe_image(frame_jpg_bytes: bytes) -> str:
    """Send image to Moondream and return the description."""
    b64 = base64.b64encode(frame_jpg_bytes).decode()
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [b64]}],
    )
    return response["message"]["content"].strip()


def draw_status(frame, status: str) -> None:
    """Draw the status bar on the camera frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 32), STATUS_BG, -1)
    cv2.putText(frame, status, (8, 22), FONT, STATUS_SCALE, STATUS_COLOR, 1, cv2.LINE_AA)


def render_description_window(lines: list[str], status: str) -> None:
    """Render the description in a separate large-text window."""
    import numpy as np

    n = max(len(lines), 1)
    h = DESC_PADDING * 3 + DESC_LINE_H * n + 40  # +40 for title row
    canvas = np.full((h, DESC_W, 3), DESC_BG, dtype="uint8")

    # title / status
    title = "Waiting for capture..." if not lines else "Description:"
    cv2.putText(canvas, title, (DESC_PADDING, DESC_PADDING + 20),
                FONT, 0.65, (120, 200, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (DESC_PADDING, DESC_PADDING + 30),
             (DESC_W - DESC_PADDING, DESC_PADDING + 30), (60, 60, 60), 1)

    for i, line in enumerate(lines):
        y = DESC_PADDING + 50 + i * DESC_LINE_H
        cv2.putText(canvas, line, (DESC_PADDING, y),
                    FONT, DESC_SCALE, DESC_COLOR, 1, cv2.LINE_AA)

    cv2.imshow(DESC_WIN, canvas)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    description_lines: list[str] = []
    status = "SPACE -> capture  |  ESC -> quit"
    processing = False

    print("Windows open. SPACE to capture, ESC to quit.")

    # open the description window immediately
    render_description_window([], status)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        draw_status(display, status)
        cv2.imshow("MDview -- Camera", display)
        render_description_window(description_lines, status)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:          # ESC
            break
        elif key == 32 and not processing:   # SPACE
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpg_bytes = buf.tobytes()
            status = "Processing..."
            description_lines = []
            processing = True

            def run():
                nonlocal description_lines, status, processing
                try:
                    text = describe_image(jpg_bytes)
                    description_lines = []
                    for paragraph in text.splitlines():
                        description_lines += textwrap.wrap(paragraph or " ", WRAP_CHARS)
                    status = "SPACE -> capture again  |  ESC -> quit"
                except Exception as exc:
                    description_lines = [f"Error: {exc}"]
                    status = "Request failed"
                finally:
                    processing = False

            threading.Thread(target=run, daemon=True).start()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
