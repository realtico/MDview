"""
MDview -- live camera demo  (with piper TTS)
  SPACE  -> capture frame, request description via Ollama, and read it aloud
  ESC    -> quit
"""

import base64
import textwrap
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import ollama
import sounddevice as sd
from piper import PiperVoice
from piper.download_voices import download_voice

# ── config ───────────────────────────────────────────────────────────────────
# --- qwen2.5vl:3b ---
#MODEL        = "qwen2.5vl:3b"
#PROMPT       = (
#    "Describe only what is clearly visible in this image. "
#    "Focus on people and objects close to them, and ignore small details. "
#)

# --- minicpm-v (alternative) ---
MODEL        = "gemma4:e2b"
PROMPT2       = (
    "Describe only what is clearly visible in this image. "
    "Focus on people and objects close to them, and ignore small details."
)

PROMPT      = (
    "Describe the main person and the nearby objects only."
    "Ignore small details and anything far away. Responda em português brasileiro."
)

# --- moondream (alternative) ---
# MODEL        = "moondream"
# PROMPT       = (
#     "Describe only what is clearly visible in this image. "
#     "Focus on people and objects close to them, and ignore small details. "
# )
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

# piper TTS
VOICE_NAME   = "en_GB-jenny_dioco-medium"
VOICES_DIR   = Path.home() / ".local" / "share" / "piper" / "voices"
# ──────────────────────────────────────────────────────────────────────────────


def load_tts_voice() -> PiperVoice:
    """Auto-download the voice model if needed and return a loaded PiperVoice."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checking voice model '{VOICE_NAME}' in {VOICES_DIR} ...")
    download_voice(VOICE_NAME, VOICES_DIR)
    model_path = VOICES_DIR / f"{VOICE_NAME}.onnx"
    print("Voice model ready.")
    return PiperVoice.load(str(model_path), use_cuda=False)


def speak_text(text: str, voice: PiperVoice) -> None:
    """Synthesise text to PCM audio and play it via sounddevice."""
    chunks = list(voice.synthesize(text))
    if not chunks:
        return
    sample_rate = chunks[0].sample_rate
    audio = np.concatenate([c.audio_float_array for c in chunks])
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    sd.stop()
    sd.play(audio_int16, samplerate=sample_rate)
    sd.wait()


def describe_image(frame_jpg_bytes: bytes) -> str:
    """Send image to Moondream and return the description."""
    b64 = base64.b64encode(frame_jpg_bytes).decode()
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [b64]}],
        options={"temperature": 0.4},
    )
    return response["message"]["content"].strip()


def draw_status(frame, status: str) -> None:
    """Draw the status bar on the camera frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 32), STATUS_BG, -1)
    cv2.putText(frame, status, (8, 22), FONT, STATUS_SCALE, STATUS_COLOR, 1, cv2.LINE_AA)


def render_description_window(lines: list[str], status: str, elapsed: float = 0.0) -> None:
    """Render the description in a separate large-text window."""
    n = max(len(lines), 1)
    h = DESC_PADDING * 3 + DESC_LINE_H * n + 40  # +40 for title row
    canvas = np.full((h, DESC_W, 3), DESC_BG, dtype="uint8")

    # title / status
    if not lines:
        title = "Waiting for capture..."
    else:
        title = f"Description:  ({elapsed:.1f}s)"
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

    tts_voice = load_tts_voice()

    description_lines: list[str] = []
    elapsed_time: float = 0.0
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
        render_description_window(description_lines, status, elapsed_time)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:          # ESC
            sd.stop()
            break
        elif key == 32 and not processing:   # SPACE
            sd.stop()          # interrupt any ongoing speech
            small = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpg_bytes = buf.tobytes()
            status = "Processing..."
            description_lines = []
            processing = True

            def run():
                nonlocal description_lines, status, processing, elapsed_time
                try:
                    t0 = time.monotonic()
                    text = describe_image(jpg_bytes)
                    elapsed_time = time.monotonic() - t0
                    description_lines = []
                    for paragraph in text.splitlines():
                        description_lines += textwrap.wrap(paragraph or " ", WRAP_CHARS)
                    status = "SPACE -> capture again  |  ESC -> quit"
                    threading.Thread(
                        target=speak_text, args=(text, tts_voice), daemon=True
                    ).start()
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
