"""
MDview -- live camera demo  (with piper TTS)
  SPACE  -> capture frame, request description via local Ollama, and read it aloud
  ENTER  -> capture frame, request description via remote Ollama, and read it aloud
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
MODEL        = "gemma3:4b"
PROMPT2       = (
    "Describe only what is clearly visible in this image. "
    "Focus on people and objects close to them, and ignore small details."
)

PROMPT      = (
    "Descreva o que a pessoa está fazendo e os objetos e o ambiente ao redor dela. "
    "Foque nas ações, postura e contexto da cena, não em características físicas. "
    "Ignore detalhes pequenos e elementos ao fundo. "
    "Responda em português brasileiro em parágrafo curto, sem introdução e sem caracteres especiais pois será lido por tts."
)

# --- moondream (alternative) ---
# MODEL        = "moondream"
# PROMPT       = (
#     "Describe only what is clearly visible in this image. "
#     "Focus on people and objects close to them, and ignore small details. "
# )

PROMPT_INTERACT = (
    "Contexto: {desc}\n\n"
    "Faça um comentário curto e descontraído sobre o que a pessoa está fazendo ou o ambiente ao redor, "
    "como se fosse uma fala casual de conversa. No máximo duas frases. "
    "Não mencione características físicas. "
    "Responda em português brasileiro, sem introdução e sem caracteres especiais pois será lido por tts."
)

REMOTE_HOST  = "http://pryde.hardywired.net:11434"

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
VOICE_NAME   = "pt_BR-faber-medium"
# VOICE_NAME   = "en_GB-jenny_dioco-medium"
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
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sd.stop()
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def interact_with_image(frame_jpg_bytes: bytes, last_description: str, host: str | None = None) -> str:
    """Send image + last description to Ollama to generate an interaction with the scene."""
    b64 = base64.b64encode(frame_jpg_bytes).decode()
    client = ollama.Client(host=host) if host else ollama
    prompt = PROMPT_INTERACT.format(desc=last_description)
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt, "images": [b64]}],
        options={"temperature": 0.7},
    )
    return response["message"]["content"].strip()


def describe_image(frame_jpg_bytes: bytes, host: str | None = None) -> str:
    """Send image to Ollama and return the description.
    If host is given, use a remote Ollama client; otherwise use the local default."""
    b64 = base64.b64encode(frame_jpg_bytes).decode()
    client = ollama.Client(host=host) if host else ollama
    response = client.chat(
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


def render_description_window(lines: list[str], status: str, elapsed: float = 0.0, source: str = "") -> None:
    """Render the description in a separate large-text window."""
    n = max(len(lines), 1)
    h = DESC_PADDING * 3 + DESC_LINE_H * n + 40  # +40 for title row
    canvas = np.full((h, DESC_W, 3), DESC_BG, dtype="uint8")

    # title / status
    if not lines:
        title = "Waiting for capture..."
    else:
        src_label = f"  [{source}]" if source else ""
        title = f"Description:  ({elapsed:.1f}s){src_label}"
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
    last_source: str = ""
    last_text: str = ""
    status = "SPACE -> local  |  ENTER -> remote  |  I -> interagir  |  ESC -> quit"
    processing = False

    print("Windows open. SPACE = local, ENTER = remote, I = interagir, ESC to quit.")

    # open the description window immediately
    render_description_window([], status)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        draw_status(display, status)
        cv2.imshow("MDview -- Camera", display)
        render_description_window(description_lines, status, elapsed_time, last_source)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:          # ESC
            sd.stop()
            break
        elif (key == 32 or key == 13) and not processing:   # SPACE or ENTER
            use_remote = (key == 13)
            host = REMOTE_HOST if use_remote else None
            source_label = "remoto" if use_remote else "local"

            try:
                sd.stop()      # interrupt any ongoing speech
            except Exception:
                pass
            small = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpg_bytes = buf.tobytes()
            status = f"Processing... [{source_label}]"
            description_lines = []
            processing = True

            def run(jpg=jpg_bytes, h=host, lbl=source_label):
                nonlocal description_lines, status, processing, elapsed_time, last_source, last_text
                try:
                    t0 = time.monotonic()
                    text = describe_image(jpg, host=h)
                    elapsed_time = time.monotonic() - t0
                    last_source = lbl
                    last_text = text
                    description_lines = []
                    for paragraph in text.splitlines():
                        description_lines += textwrap.wrap(paragraph or " ", WRAP_CHARS)
                    status = "SPACE -> local  |  ENTER -> remote  |  I -> interagir  |  ESC -> quit"
                    threading.Thread(
                        target=speak_text, args=(text, tts_voice), daemon=True
                    ).start()
                except Exception as exc:
                    description_lines = [f"Error: {exc}"]
                    status = "Request failed"
                finally:
                    processing = False

            threading.Thread(target=run, daemon=True).start()

        elif key == ord('i') and not processing and last_text:
            try:
                sd.stop()
            except Exception:
                pass
            small = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpg_bytes_i = buf.tobytes()
            status = "Gerando interação..."
            description_lines = []
            processing = True

            def run_interact(jpg=jpg_bytes_i, desc=last_text):
                nonlocal description_lines, status, processing, elapsed_time, last_source
                try:
                    t0 = time.monotonic()
                    text = interact_with_image(jpg, desc)
                    elapsed_time = time.monotonic() - t0
                    last_source = "interação"
                    description_lines = []
                    for paragraph in text.splitlines():
                        description_lines += textwrap.wrap(paragraph or " ", WRAP_CHARS)
                    status = "SPACE -> local  |  ENTER -> remote  |  I -> interagir  |  ESC -> quit"
                    threading.Thread(
                        target=speak_text, args=(text, tts_voice), daemon=True
                    ).start()
                except Exception as exc:
                    description_lines = [f"Error: {exc}"]
                    status = "Interação falhou"
                finally:
                    processing = False

            threading.Thread(target=run_interact, daemon=True).start()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
