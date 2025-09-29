import os, time, threading, queue, requests
import numpy as np
import webrtcvad
import alsaaudio
import traceback

# ... (keep all the ENV/CONFIG section as before)

def test_audio_devices():
    """Test which audio devices are accessible"""
    print("\n=== Testing Audio Devices ===")
    test_devices = [
        "hw:0,0", "hw:1,0", "hw:Audio,0", "hw:Audio_1,0",
        "headset0_in", "headset0_out", "headset1_in", "headset1_out"
    ]
    
    for dev in test_devices:
        try:
            # Try capture
            pcm = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL, device=dev)
            pcm.close()
            print(f"✓ {dev} - capture OK")
        except Exception as e:
            print(f"✗ {dev} - capture failed: {e}")
        
        try:
            # Try playback
            pcm = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NORMAL, device=dev)
            pcm.close()
            print(f"✓ {dev} - playback OK")
        except Exception as e:
            print(f"✗ {dev} - playback failed: {e}")
    print("=============================\n")

def open_capture(device_name):
    try:
        pcm = alsaaudio.PCM(
            type=alsaaudio.PCM_CAPTURE, 
            mode=alsaaudio.PCM_NORMAL, 
            device=device_name,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=int(SAMPLE_RATE * FRAME_MS / 1000)
        )
        frame_size = int(SAMPLE_RATE * FRAME_MS / 1000)
        print(f"[SUCCESS] Opened capture device: {device_name}")
        return pcm, frame_size
    except Exception as e:
        print(f"[ERROR] Failed to open capture device {device_name}: {e}")
        raise

def open_playback(device_name):
    try:
        pcm = alsaaudio.PCM(
            type=alsaaudio.PCM_PLAYBACK, 
            mode=alsaaudio.PCM_NORMAL, 
            device=device_name,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=int(SAMPLE_RATE * FRAME_MS / 1000)
        )
        print(f"[SUCCESS] Opened playback device: {device_name}")
        return pcm
    except Exception as e:
        print(f"[ERROR] Failed to open playback device {device_name}: {e}")
        raise

# Add the rest of your functions here (capture_loop, vad_burster, etc.)
# but with added error handling and debug prints

def main():
    print("[router] starting…")
    
    # Test devices first
    test_audio_devices()
    
    print(f" A: {A_IN} -> {A_TGT} via {A_SRC}  out:{A_OUT}")
    print(f" B: {B_IN} -> {B_TGT} via {B_SRC}  out:{B_OUT}")
    
    # Continue with the rest of main() as before
    # ...

if __name__ == "__main__":
    main()
