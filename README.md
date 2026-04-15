# MDview — Moondream Vision Demos

A collection of local computer vision demos built around [Moondream](https://github.com/vikhyat/moondream) (via [Ollama](https://ollama.com)) and OpenCV. Everything runs offline — no API key, no cloud.

This repository is also the prototyping ground for a larger **mmWave-triggered interaction agent** (see Roadmap below).

---

## Scripts

### `demo.py` — Scene description

Captures a webcam frame and asks Moondream to describe what it sees.

**Controls**

| Key | Action |
|-----|--------|
| `SPACE` | Capture the current frame and request a description |
| `ESC` | Quit |

**Requirements**

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Moondream model pulled: `ollama pull moondream`

> If your local model is listed as `moondream2` in `ollama list`, open `demo.py` and change `MODEL = "moondream"` to `MODEL = "moondream2"`.

```bash
pip install -r requirements.txt
python demo.py
```

Two windows open side by side:

- **MDview -- Camera**: live camera feed with a status bar at the top
- **MDview -- Description**: dedicated text window with the model's response in a larger font

---

### `facefind.py` — Face detection

Detects faces in a captured frame using OpenCV's built-in Haar Cascade classifier. No Ollama or extra models required.

**Controls**

| Key | Action |
|-----|--------|
| `SPACE` | Capture frame and detect faces |
| `ESC` | Quit |

```bash
python facefind.py   # same requirements.txt, no extra dependencies
```

Detected faces are shown with green bounding boxes and index labels (`#1`, `#2`, …) in a second **Detections** window. Detection sensitivity can be tuned via `SCALE_FACTOR`, `MIN_NEIGHBORS`, and `MIN_FACE_PX` at the top of the file.

---

## Roadmap — mmWave Interaction Agent

The end goal is a privacy-first, fully local interaction agent triggered by a millimeter-wave radar sensor:

```
mmWave radar (UART)
       |
  agent_pi.py  (Raspberry Pi Zero 2W)
  |- capture frame (picamera2)
  |- face detection (Haar / lightweight model)
  `- POST frame to server
                              agent_server.py  (local network host)
                              |- Moondream  -> scene description
                              `- Gemma      -> contextual greeting
       <--- { scene, greeting } ---
  |- Piper TTS  -> speaks greeting
  `- Vosk STT   -> listens, loops back to /chat
```

The architecture uses independent microservices communicating over IP sockets so each component can run on any machine on the local network and be promoted or demoted freely (e.g. move TTS/STT to the Pi, keep vision models on the host). No cloud services involved at any stage.

**Status:** hardware preparation in progress (Pi Zero 2W + mmWave sensor). Implementation will start once the physical layer is ready.

---

## Compatibility

| Platform | Status | Notes |
|----------|--------|-------|
| Linux (native) | ✅ | Works out of the box |
| macOS | ✅ | Works out of the box |
| Windows (native Python) | ✅ | Works out of the box |
| WSL2 | ⚠️ | Camera passthrough required — see below |

---

## WSL2 — USB Camera Passthrough with usbipd-win

WSL2 does not expose USB devices (including webcams) to the Linux kernel by default. The tool **usbipd-win** bridges the gap by forwarding USB devices from Windows into WSL2.

### Prerequisites

- Windows 10 (21H2+) or Windows 11
- WSL2 with a Linux distro installed
- A kernel that supports USBIP — most modern WSL2 kernels do

### 1. Install usbipd-win (Windows side)

Using **winget** (recommended):

```powershell
winget install --interactive --exact dorssel.usbipd-win
```

Or download the installer from the [releases page](https://github.com/dorssel/usbipd-win/releases).

Restart your terminal (or the machine) after installing.

### 2. Install USBIP tools (WSL2 / Linux side)

```bash
# Debian / Ubuntu
sudo apt install linux-tools-generic hwdata
sudo update-alternatives --install /usr/local/bin/usbip usbip \
  /usr/lib/linux-tools/*-generic/usbip 20
```

### 3. Attach the camera

Open **PowerShell as Administrator** on Windows:

```powershell
# List all USB devices and find your camera
usbipd list

# Bind the device (replace X-Y with the correct BUS ID, e.g. 2-3)
usbipd bind --busid X-Y

# Attach it to WSL
usbipd attach --wsl --busid X-Y
```

The camera should now appear inside WSL as `/dev/video0` — verify with:

```bash
ls /dev/video*
```

### 4. Run the demo normally

```bash
python demo.py
```

### Detaching

When you are done, detach the device from PowerShell:

```powershell
usbipd detach --busid X-Y
```

### Troubleshooting

- **Device not found in `/dev/video*`** — make sure `v4l2` is available:
  ```bash
  sudo apt install v4l-utils && v4l2-ctl --list-devices
  ```
- **Permission denied on `/dev/video0`** — add your user to the `video` group:
  ```bash
  sudo usermod -aG video $USER  # re-login after
  ```
- **Ollama not reachable from WSL** — by default Ollama binds to `localhost` on Windows, which WSL2 can reach via the host IP. If it fails, start Ollama with:
  ```powershell
  $env:OLLAMA_HOST="0.0.0.0"; ollama serve
  ```
  and check Windows Firewall rules for port `11434`.

## License

MIT
