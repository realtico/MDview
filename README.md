# MDview — Moondream Live Camera Demo

A minimal Python demo that uses your webcam and a local [Moondream](https://github.com/vikhyat/moondream) vision model (via [Ollama](https://ollama.com)) to describe what the camera sees — entirely offline, no API key required.

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Capture the current frame and request a description |
| `ESC` | Quit |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Moondream model pulled in Ollama

## Setup

### 1. Pull the model

```bash
ollama pull moondream
```

> If your local model is listed as `moondream2` in `ollama list`, open `demo.py` and change `MODEL = "moondream"` to `MODEL = "moondream2"`.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python demo.py
```

Two windows open side by side:

- **MDview -- Camera**: live camera feed with a status bar at the top
- **MDview -- Description**: dedicated text window that shows the model's description in a larger, readable font

Press `SPACE` to capture the current frame. The description appears in the second window once the model responds. Press `SPACE` again to capture a new frame.

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
