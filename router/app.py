import os
import time
import threading
import queue
import requests
import numpy as np
import webrtcvad
import alsaaudio
import grpc
import riva.client

# Configuration
CANARY_URL = os.getenv("CANARY_URL", "http://127.0.0.1:7000/ast")
RIVA_ADDR = os.getenv("RIVA_ADDR", "127.0.0.1:50051")

# Audio settings
SAMPLE_RATE = 16000
FRAME_MS = 20
CHANNELS = 1
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
VAD_MODE = 2
BURST_MIN_MS = 500
BURST_MAX_MS = 2000

# Devices
A_IN = os.getenv("A_IN", "hw:0,0")
A_OUT = os.getenv("A_OUT", "hw:0,0")
B_IN = os.getenv("B_IN", "hw:1,0")
B_OUT = os.getenv("B_OUT", "hw:1,0")

class AudioDevice:
    """Wrapper for audio devices that handles mono/stereo conversion"""
    
    def __init__(self, device_name, mode='capture', channels=1):
        self.device_name = device_name
        self.mode = mode
        self.target_channels = channels
        
        # Try different device formats
        self.device = None
        self.actual_channels = None
        
        # Try opening with requested channels first
        for ch in [channels, 2, 1]:  # Try mono, then stereo, then mono again
            try:
                if mode == 'capture':
                    self.device = alsaaudio.PCM(
                        type=alsaaudio.PCM_CAPTURE,
                        mode=alsaaudio.PCM_NORMAL,
                        device=device_name,
                        channels=ch,
                        rate=SAMPLE_RATE,
                        format=alsaaudio.PCM_FORMAT_S16_LE,
                        periodsize=FRAME_SIZE
                    )
                else:
                    self.device = alsaaudio.PCM(
                        type=alsaaudio.PCM_PLAYBACK,
                        mode=alsaaudio.PCM_NORMAL,
                        device=device_name,
                        channels=ch,
                        rate=SAMPLE_RATE,
                        format=alsaaudio.PCM_FORMAT_S16_LE,
                        periodsize=FRAME_SIZE
                    )
                self.actual_channels = ch
                print(f"Opened {device_name} with {ch} channels")
                break
            except alsaaudio.ALSAAudioError as e:
                continue
        
        if not self.device:
            raise Exception(f"Could not open device {device_name}")
    
    def read(self):
        """Read audio and convert to mono if needed"""
        length, data = self.device.read()
        if length > 0 and self.actual_channels == 2 and self.target_channels == 1:
            # Convert stereo to mono
            stereo = np.frombuffer(data, dtype=np.int16)
            # Take every other sample (left channel) or average
            mono = stereo[0::2]  # Just left channel
            # Or average: mono = ((stereo[0::2] + stereo[1::2]) // 2).astype(np.int16)
            return mono.tobytes()
        return data
    
    def write(self, data):
        """Write audio, converting from mono to stereo if needed"""
        if self.actual_channels == 2 and self.target_channels == 1:
            # Convert mono to stereo by duplicating
            mono = np.frombuffer(data, dtype=np.int16)
            stereo = np.zeros(len(mono) * 2, dtype=np.int16)
            stereo[0::2] = mono  # Left channel
            stereo[1::2] = mono  # Right channel
            self.device.write(stereo.tobytes())
        else:
            self.device.write(data)
    
    def close(self):
        if self.device:
            self.device.close()




class TranslationPipeline:
    def __init__(self):
        print("Initializing Translation Pipeline...")
        
        # Initialize Riva TTS
        self.riva_auth = riva.client.Auth(uri=RIVA_ADDR)
        self.tts = riva.client.SpeechSynthesisService(self.riva_auth)
        print(f"✓ Connected to Riva TTS at {RIVA_ADDR}")
        
        # Test Canary connection
        self.test_canary()
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(VAD_MODE)
        
        self.running = True
    
    def test_canary(self):
        """Test Canary AST service"""
        try:
            r = requests.get(f"http://127.0.0.1:7000/health", timeout=2)
            if r.json()["model_loaded"]:
                print("✓ Canary AST model loaded")
            else:
                print("⚠ Canary AST not ready")
        except:
            print("✗ Cannot reach Canary AST service")
    
    def translate_audio(self, audio_bytes, src_lang, tgt_lang):
        """Send audio to Canary for translation"""
        try:
            payload = {
                "pcm16_hex": audio_bytes.hex(),
                "sr": SAMPLE_RATE,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang
            }
            
            r = requests.post(CANARY_URL, json=payload, timeout=10)
            r.raise_for_status()
            
            result = r.json()
            return result.get("translation", "") or result.get("transcript", "")
            
        except Exception as e:
            print(f"Translation error: {e}")
            return ""
    
    def synthesize_speech(self, text, language):
        """Use Riva TTS to synthesize speech"""
        try:
            if language == "en":
                voice = "English-US-Female-1"
                lang_code = "en-US"
            else:
                voice = "Spanish-US-Female-1"
                lang_code = "es-US"
            
            response = self.tts.synthesize(
                text=text,
                voice_name=voice,
                language_code=lang_code,
                sample_rate_hz=SAMPLE_RATE
            )
            
            return response.audio
            
        except Exception as e:
            print(f"TTS error: {e}")
            return b""
    
    def capture_with_vad(self, device):
        """Capture audio until silence detected"""
        cap = alsaaudio.PCM(
            type=alsaaudio.PCM_CAPTURE,
            mode=alsaaudio.PCM_NORMAL,
            device=device,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=FRAME_SIZE
        )
        
        frames = []
        speech_frames = 0
        silence_frames = 0
        max_silence = int(1000 / FRAME_MS)  # 1 second of silence
        min_speech = int(BURST_MIN_MS / FRAME_MS)
        max_speech = int(BURST_MAX_MS / FRAME_MS)
        
        recording = False
        
        while True:
            length, data = cap.read()
            if length > 0:
                is_speech = self.vad.is_speech(data, SAMPLE_RATE)
                
                if is_speech:
                    if not recording:
                        print("Speech detected...")
                        recording = True
                    
                    frames.append(data)
                    speech_frames += 1
                    silence_frames = 0
                    
                    if speech_frames >= max_speech:
                        # Max recording length reached
                        break
                        
                elif recording:
                    frames.append(data)
                    silence_frames += 1
                    
                    if silence_frames >= max_silence:
                        if speech_frames >= min_speech:
                            # Enough speech captured
                            break
                        else:
                            # Not enough speech, reset
                            frames = []
                            speech_frames = 0
                            silence_frames = 0
                            recording = False
        
        cap.close()
        return b''.join(frames) if frames else b""
    
    def play_audio(self, audio_bytes, device):
        """Play audio through specified device"""
        if not audio_bytes:
            return
        
        play = alsaaudio.PCM(
            type=alsaaudio.PCM_PLAYBACK,
            mode=alsaaudio.PCM_NORMAL,
            device=device,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=FRAME_SIZE
        )
        
        play.write(audio_bytes)
        play.close()
    
    def translation_loop(self, name, in_dev, out_dev, src_lang, tgt_lang):
        """Main translation loop for one direction"""
        print(f"[{name}] Starting: {in_dev} ({src_lang}) -> {tgt_lang} -> {out_dev}")
        
        while self.running:
            try:
                # Capture speech
                audio = self.capture_with_vad(in_dev)
                if not audio:
                    continue
                
                print(f"[{name}] Processing {len(audio)} bytes...")
                
                # Translate
                translation = self.translate_audio(audio, src_lang, tgt_lang)
                if not translation:
                    print(f"[{name}] No translation")
                    continue
                
                print(f"[{name}] {src_lang}->{tgt_lang}: {translation[:50]}...")
                
                # Synthesize
                tts_audio = self.synthesize_speech(translation, tgt_lang)
                
                # Play
                if tts_audio:
                    self.play_audio(tts_audio, out_dev)
                    print(f"[{name}] Played {len(tts_audio)} bytes")
                
            except Exception as e:
                print(f"[{name}] Error: {e}")
                time.sleep(1)
    
    def run(self):
        """Run bidirectional translation"""
        # Direction A: Spanish -> English
        thread_a = threading.Thread(
            target=self.translation_loop,
            args=("A", A_IN, B_OUT, "es", "en"),
            daemon=True
        )
        
        # Direction B: English -> Spanish  
        thread_b = threading.Thread(
            target=self.translation_loop,
            args=("B", B_IN, A_OUT, "en", "es"),
            daemon=True
        )
        
        thread_a.start()
        thread_b.start()
        
        print("\n=== TRANSLATION ACTIVE ===")
        print("Speak in either English or Spanish")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            self.running = False

if __name__ == "__main__":
    pipeline = TranslationPipeline()
    pipeline.run()
