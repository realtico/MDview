"""
MDview -- live camera demo v4  (piper TTS + VOSK voice input)

  ENTER  -> captura frame e descreve (usa fonte configurada na linha de comando)
  I      -> interacao curta com a ultima cena capturada
  R      -> escuta por voz (VOSK) e responde seguindo a conversa
  ESC    -> sair

Uso:
  python demo4.py            # usa Ollama local (padrao)
  python demo4.py --remote   # usa Ollama remoto

VOSK:
  Baixe o modelo portugues em https://alphacephei.com/vosk/models
  Recomendado: vosk-model-small-pt-0.3
  Extraia em: ~/vosk-models/vosk-model-small-pt-0.3/
"""

import argparse
import base64
import json
import queue
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

try:
    from vosk import KaldiRecognizer
    from vosk import Model as VoskModel
    _VOSK_AVAILABLE = True
except ImportError:
    _VOSK_AVAILABLE = False

# ── config ────────────────────────────────────────────────────────────────────
MODEL = "gemma3:4b"

PROMPT = (
    "Descreva o que a pessoa está fazendo e os objetos e o ambiente ao redor dela. "
    "Foque nas ações, postura e contexto da cena, não em características físicas. "
    "Ignore detalhes pequenos e elementos ao fundo. "
    "Responda em português brasileiro em parágrafo curto, "
    "sem introdução e sem caracteres especiais pois será lido por tts."
)

PROMPT_INTERACT = (
    "Contexto: {desc}\n\n"
    "Com base APENAS no que está descrito acima, faça um comentário curto e descontraído "
    "ou uma pergunta curiosa sobre o que a pessoa está fazendo ou os objetos visíveis. "
    "Não invente nada que não esteja no contexto. Máximo duas frases, tom de papo informal. "
    "Não mencione características físicas, não use emojis nem símbolos especiais, "
    "sem introdução, responda direto em português brasileiro pois será lido por tts."
)

SYSTEM_CHAT = (
    "Você é um interlocutor curioso e descontraído. "
    "Baseie-se APENAS no que o usuário disser e no contexto da cena descrito abaixo. "
    "Não invente personagens, histórias ou detalhes que não foram mencionados. "
    "Converse de forma natural: comente o que o usuário disse, faça perguntas sobre o que ele próprio trouxer. "
    "Respostas curtas, no máximo três frases por vez. "
    "Não repita a descrição, não mencione características físicas, "
    "não use emojis nem símbolos especiais, sem introdução, "
    "responda sempre em português brasileiro pois será lido por tts."
)

REMOTE_HOST = "http://pryde.hardywired.net:11434"

# Caminho do modelo VOSK – ajuste se necessário
VOSK_MODEL_PATH = Path.home() / "vosk-models" / "vosk-model-small-pt-0.3"

FONT         = cv2.FONT_HERSHEY_SIMPLEX

# janela câmera
STATUS_SCALE = 0.6
STATUS_COLOR = (255, 255, 255)
STATUS_BG    = (30, 30, 30)

# janela descrição
DESC_WIN     = "MDview -- Description"
DESC_W       = 700
DESC_LINE_H  = 30
DESC_SCALE   = 0.65
DESC_COLOR   = (230, 230, 230)
DESC_BG      = (20, 20, 20)
DESC_PADDING = 20
WRAP_CHARS   = 64

# piper TTS
VOICE_NAME = "pt_BR-faber-medium"
VOICES_DIR = Path.home() / ".local" / "share" / "piper" / "voices"
# ─────────────────────────────────────────────────────────────────────────────


def load_tts_voice() -> PiperVoice:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Verificando modelo de voz '{VOICE_NAME}' em {VOICES_DIR} ...")
    download_voice(VOICE_NAME, VOICES_DIR)
    model_path = VOICES_DIR / f"{VOICE_NAME}.onnx"
    print("Modelo TTS pronto.")
    return PiperVoice.load(str(model_path), use_cuda=False)


def load_vosk_model():
    """Carrega o modelo VOSK. Retorna None se não disponível."""
    if not _VOSK_AVAILABLE:
        print("AVISO: vosk não instalado. Tecla R desabilitada. (pip install vosk)")
        return None
    if not VOSK_MODEL_PATH.exists():
        print(
            f"AVISO: Modelo VOSK não encontrado em {VOSK_MODEL_PATH}\n"
            "       Baixe em https://alphacephei.com/vosk/models e extraia no caminho acima.\n"
            "       Tecla R desabilitada."
        )
        return None
    print(f"Carregando modelo VOSK de {VOSK_MODEL_PATH} ...")
    model = VoskModel(str(VOSK_MODEL_PATH))
    print("Modelo VOSK pronto.")
    return model


def speak_text(text: str, voice: PiperVoice) -> None:
    chunks = list(voice.synthesize(text))
    if not chunks:
        return
    sample_rate = chunks[0].sample_rate
    audio = np.concatenate([c.audio_float_array for c in chunks])
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sd.stop()
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def listen_and_transcribe(vosk_model, samplerate: int = 16000,
                           max_seconds: float = 12.0) -> str:
    """
    Grava do microfone até detectar silêncio ou atingir max_seconds.
    Retorna a transcrição VOSK em texto.

    Lógica de parada:
      - Precisa de MIN_SPEECH blocos com energia acima de SILENCE_RMS antes
        de começar a contar o silêncio.
      - Após MIN_SPEECH, SILENCE_TRAIL blocos consecutivos abaixo do limiar
        encerram a gravação (~1,5 s de silêncio).
    """
    rec    = KaldiRecognizer(vosk_model, samplerate)
    audio_q: queue.Queue = queue.Queue()

    BLOCK_SIZE    = 4000    # amostras por bloco (~0,25 s a 16 kHz)
    SILENCE_RMS   = 0.015   # limiar RMS normalizado para silêncio
    MIN_SPEECH    = 4       # blocos mínimos de fala antes de aceitar silêncio
    SILENCE_TRAIL = 6       # blocos silenciosos consecutivos para parar (~1,5 s)

    speech_blocks  = 0
    silence_blocks = 0

    def _cb(indata, frames, time_info, _status):
        audio_q.put(bytes(indata))

    with sd.RawInputStream(samplerate=samplerate, blocksize=BLOCK_SIZE,
                           dtype="int16", channels=1, callback=_cb):
        deadline = time.monotonic() + max_seconds
        while time.monotonic() < deadline:
            try:
                data = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(arr ** 2)))
            rec.AcceptWaveform(data)

            if rms > SILENCE_RMS:
                speech_blocks += 1
                silence_blocks = 0
            elif speech_blocks >= MIN_SPEECH:
                silence_blocks += 1
                if silence_blocks >= SILENCE_TRAIL:
                    break

    result = json.loads(rec.FinalResult())
    return result.get("text", "").strip()


def describe_image(frame_jpg_bytes: bytes, host: str | None = None) -> str:
    b64    = base64.b64encode(frame_jpg_bytes).decode()
    client = ollama.Client(host=host) if host else ollama
    resp   = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [b64]}],
        options={"temperature": 0.4},
    )
    return resp["message"]["content"].strip()


def interact_with_image(frame_jpg_bytes: bytes, last_description: str,
                        host: str | None = None) -> str:
    b64    = base64.b64encode(frame_jpg_bytes).decode()
    client = ollama.Client(host=host) if host else ollama
    prompt = PROMPT_INTERACT.format(desc=last_description)
    resp   = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt, "images": [b64]}],
        options={"temperature": 0.7},
    )
    return resp["message"]["content"].strip()


def chat_reply(history: list[dict], host: str | None = None) -> str:
    """Envia o histórico da conversa ao Ollama (texto) e retorna a resposta."""
    client = ollama.Client(host=host) if host else ollama
    resp   = client.chat(
        model=MODEL,
        messages=history,
        options={"temperature": 0.7},
    )
    return resp["message"]["content"].strip()


def draw_status(frame, status: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 32), STATUS_BG, -1)
    cv2.putText(frame, status, (8, 22), FONT, STATUS_SCALE, STATUS_COLOR, 1, cv2.LINE_AA)


def render_description_window(lines: list[str], elapsed: float = 0.0,
                               source: str = "") -> None:
    n      = max(len(lines), 1)
    h      = DESC_PADDING * 3 + DESC_LINE_H * n + 40
    canvas = np.full((h, DESC_W, 3), DESC_BG, dtype="uint8")

    if not lines:
        title = "Aguardando captura..."
    else:
        src_label = f"  [{source}]" if source else ""
        title = f"({elapsed:.1f}s){src_label}"

    cv2.putText(canvas, title, (DESC_PADDING, DESC_PADDING + 20),
                FONT, 0.6, (120, 200, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (DESC_PADDING, DESC_PADDING + 30),
             (DESC_W - DESC_PADDING, DESC_PADDING + 30), (60, 60, 60), 1)

    for i, line in enumerate(lines):
        # linhas "Você:" em amarelo, "IA:" em verde, resto em branco
        if line.startswith("Voce:") or line.startswith("Você:"):
            color = (100, 220, 255)
        elif line.startswith("IA:"):
            color = (100, 255, 160)
        else:
            color = DESC_COLOR
        y = DESC_PADDING + 50 + i * DESC_LINE_H
        cv2.putText(canvas, line, (DESC_PADDING, y),
                    FONT, DESC_SCALE, color, 1, cv2.LINE_AA)

    cv2.imshow(DESC_WIN, canvas)


def main():
    parser = argparse.ArgumentParser(description="MDview demo4")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--remote", action="store_true",
                       help="Usa Ollama remoto (" + REMOTE_HOST + ")")
    group.add_argument("--local",  action="store_true",
                       help="Usa Ollama local (padrao)")
    args = parser.parse_args()

    host         = REMOTE_HOST if args.remote else None
    source_label = "remoto" if args.remote else "local"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a câmera.")
        return

    tts_voice  = load_tts_voice()
    vosk_model = load_vosk_model()

    description_lines: list[str] = []
    elapsed_time: float = 0.0
    last_source: str    = ""
    last_text: str      = ""
    chat_history: list[dict] = []
    processing = False

    voice_available = vosk_model is not None
    hint = (
        f"ENTER->descrever [{source_label}] | I->interagir"
        + (" | R->falar" if voice_available else "")
        + " | ESC->sair"
    )
    status = hint

    print(f"Fonte: {source_label.upper()}  "
          f"ENTER=descrever  I=interagir"
          + ("  R=falar" if voice_available else "  [R desabilitado: VOSK indisponivel]")
          + "  ESC=sair")
    render_description_window([], 0.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        draw_status(display, status)
        cv2.imshow("MDview -- Camera", display)
        render_description_window(description_lines, elapsed_time, last_source)

        key = cv2.waitKey(1) & 0xFF

        # ── ESC ──────────────────────────────────────────────────────────────
        if key == 27:
            sd.stop()
            break

        # ── ENTER: descrever imagem ───────────────────────────────────────────────
        elif key == 13 and not processing:

            try:
                sd.stop()
            except Exception:
                pass

            small = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpg_bytes = buf.tobytes()
            status    = f"Descrevendo... [{source_label}]"
            description_lines = []
            processing = True

            def run_describe(jpg=jpg_bytes, h=host, lbl=source_label):
                nonlocal description_lines, status, processing
                nonlocal elapsed_time, last_source, last_text, chat_history
                try:
                    t0   = time.monotonic()
                    text = describe_image(jpg, host=h)
                    elapsed_time = time.monotonic() - t0
                    last_source  = lbl
                    last_text    = text
                    # reinicia o histórico com o contexto visual como system prompt
                    chat_history = [
                        {"role": "system", "content": f"{SYSTEM_CHAT}\n\nContexto da cena: {text}"},
                    ]
                    description_lines = []
                    for para in text.splitlines():
                        description_lines += textwrap.wrap(para or " ", WRAP_CHARS)
                    status = hint
                    threading.Thread(target=speak_text, args=(text, tts_voice),
                                     daemon=True).start()
                except Exception as exc:
                    description_lines = [f"Erro: {exc}"]
                    status = "Descricao falhou"
                finally:
                    processing = False

            threading.Thread(target=run_describe, daemon=True).start()

        # ── I: interação curta com a cena ─────────────────────────────────────
        elif key == ord('i') and not processing and last_text:
            try:
                sd.stop()
            except Exception:
                pass

            small = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpg_i     = buf.tobytes()
            status    = "Gerando interacao..."
            description_lines = []
            processing = True

            def run_interact(jpg=jpg_i, desc=last_text, h=host):
                nonlocal description_lines, status, processing
                nonlocal elapsed_time, last_source, chat_history
                try:
                    t0   = time.monotonic()
                    text = interact_with_image(jpg, desc, host=h)
                    elapsed_time = time.monotonic() - t0
                    last_source  = "interacao"
                    # adiciona ao histórico para que R possa continuar
                    chat_history.append({"role": "assistant", "content": text})
                    description_lines = []
                    for para in text.splitlines():
                        description_lines += textwrap.wrap(para or " ", WRAP_CHARS)
                    status = hint
                    threading.Thread(target=speak_text, args=(text, tts_voice),
                                     daemon=True).start()
                except Exception as exc:
                    description_lines = [f"Erro: {exc}"]
                    status = "Interacao falhou"
                finally:
                    processing = False

            threading.Thread(target=run_interact, daemon=True).start()

        # ── R: entrada por voz → chat ─────────────────────────────────────────
        elif (key == ord('r') and not processing
              and voice_available and chat_history):
            try:
                sd.stop()
            except Exception:
                pass

            status     = "Ouvindo...  (fale agora)"
            processing = True

            def run_voice(h=host):
                nonlocal description_lines, status, processing
                nonlocal elapsed_time, last_source, chat_history
                try:
                    # 1. transcreve fala
                    spoken = listen_and_transcribe(vosk_model)
                    if not spoken:
                        status = "Nao entendi. Tente de novo."
                        processing = False
                        return

                    status = f"Voce disse: {spoken[:60]}"

                    # 2. envia ao Ollama com histórico completo
                    chat_history.append({"role": "user", "content": spoken})
                    t0    = time.monotonic()
                    reply = chat_reply(chat_history, host=h)
                    elapsed_time = time.monotonic() - t0
                    last_source  = "voz"
                    chat_history.append({"role": "assistant", "content": reply})

                    # 3. exibe no painel (você / IA em cores diferentes)
                    description_lines = (
                        textwrap.wrap(f"Voce: {spoken}", WRAP_CHARS)
                        + [""]
                        + textwrap.wrap(f"IA: {reply}", WRAP_CHARS)
                    )
                    status = hint
                    threading.Thread(target=speak_text, args=(reply, tts_voice),
                                     daemon=True).start()
                except Exception as exc:
                    description_lines = [f"Erro: {exc}"]
                    status = "Voz falhou"
                finally:
                    processing = False

            threading.Thread(target=run_voice, daemon=True).start()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
