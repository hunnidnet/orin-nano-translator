Jetson Live Translator

English ↔ Spanish real-time speech-to-speech translator
Built on NVIDIA Jetson Orin, Faster-Whisper (ASR), CTranslate2 Marian (MT), and NVIDIA Riva TTS.

⸻

✨ Features
	•	Two modes
	1.	Local Mode – Two headsets for in-person bilingual conversation (translation entirely on the Jetson).
	2.	Jitsi Call Mode – Jetson joins a self-hosted Jitsi Meet call and provides real-time translation for a remote caller.
	•	Switchable: physical button (GPIO/USB) or web toggle.
	•	Low-latency pipeline:
	•	ASR: Faster-Whisper (CUDA, FP16)
	•	MT: CTranslate2 Marian EN↔ES (CUDA, FP16)
	•	TTS: Riva FastPitch + HiFiGAN (en-US / es-US)

⸻

🛠️ Hardware

Item	Purpose
NVIDIA Jetson Orin Nano 8 GB	Core compute
2× USB audio headsets or 2× low-latency 2.4 GHz USB dongle headsets	Local Mode
Conference mic/speaker (e.g., Polycom)	Shared device for calls
Physical toggle button (optional)	Mode switch
USB audio interface	For TRRS to Jetson


⸻

📦 Software Stack
	•	JetPack 6.x (CUDA 12.x)
	•	Docker + NVIDIA runtime
	•	ASR: Faster-Whisper microservice (Docker)
	•	MT: CTranslate2 Marian (host systemd service)
	•	TTS: NVIDIA Riva TTS (Docker or Quickstart)
	•	Router: Python service (Docker) – VAD, chunking, routing

⸻

🚀 Quick Start

1) Jetson Prep

sudo apt update && sudo apt install -y docker.io
sudo usermod -aG docker $USER
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure
sudo systemctl restart docker

2) Clone and Build

git clone https://github.com/<yourname>/jetson-live-translator.git
cd jetson-live-translator
docker compose build   # builds ASR + Router

3) Riva TTS models

Follow the Riva Quickstart to deploy the TTS voices you want:
	•	English (en-US)
	•	Spanish (es-US)
Ensure the Riva server exposes gRPC 50051.

4) MT (CTranslate2) models

Download Marian models and convert with CTranslate2 on the Jetson:

~/mt/
├── opus-mt-en-es/ (Helsinki-NLP)
├── opus-mt-es-en/ (Helsinki-NLP)
├── ct2-en-es/     (converted)
└── ct2-es-en/     (converted)

5) Install MT systemd service (autostarts on reboot)

Use the provided unit and env file:

# create /etc/jetson-live-translator/mt-ct2.env and /etc/systemd/system/mt-ct2.service
sudo systemctl daemon-reload
sudo systemctl enable mt-ct2
sudo systemctl start mt-ct2
curl -s http://127.0.0.1:7010/health

6) Start Docker services

docker compose up -d asr router   # riva-tts only if you don't already run quickstart

Endpoints used by the Router
	•	ASR: http://127.0.0.1:7001/asr/chunk
	•	MT:  http://127.0.0.1:7010/mt
	•	TTS: 127.0.0.1:50051 (gRPC)

⸻

🎛 Mode Switching
	•	Physical switch: GPIO input toggles router mode (edit pin in config).
	•	Web toggle: Router exposes a simple dashboard on port 8080.

⸻

🗂 Config (config.yaml)

mode: local
audio:
  sample_rate: 16000
  frame_ms: 20
  burst_ms_min: 300
  burst_ms_max: 600
local_devices:
  in_A: "plughw:0,0"
  out_B: "plughw:1,0"
  in_B: "plughw:1,0"
  out_A: "plughw:0,0"
languages:
  A_src: "es"
  A_tgt: "en"
  B_src: "en"
  B_tgt: "es"
tts:
  voice_en: "English-US.Female-1"
  voice_es: "Spanish-US.Female-1"


⸻

📊 Data Flow (Local Mode)

Headset A mic → Faster-Whisper → (es text)
               → CTranslate2 (es→en)
               → Riva TTS (en)
               → Headset B spk

Headset B mic → Faster-Whisper → (en text)
               → CTranslate2 (en→es)
               → Riva TTS (es)
               → Headset A spk


⸻

🧪 Health Checks

curl -s http://127.0.0.1:7001/health  # ASR
curl -s http://127.0.0.1:7010/health  # MT
/opt/riva/examples/talk.py --server 127.0.0.1:50051 --list-voices  # Riva voices


⸻

🛠 Troubleshooting
	•	MT won’t start: journalctl -u mt-ct2 -f and confirm paths in /etc/jetson-live-translator/mt-ct2.env.
	•	No CUDA in CT2: Ensure you compiled CT2 with -DWITH_CUDA=ON -DCUDA_ARCH_LIST=8.7 and that /usr/local/lib is in LD_LIBRARY_PATH.
	•	Router no audio: docker exec -it translator-router aplay -L to verify ALSA device names, and confirm they match env vars.

⸻

If you want, I can also drop a minimal asr_server.py and router/app.py that already call these endpoints—you’ve got most of it, but I can line it up with your current env vars exactly.
