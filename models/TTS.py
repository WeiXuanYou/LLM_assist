import os
import sys
import piper

class TTSModel():
    def __init__(self, piper_voice_onnx_path,
                 piper_voice_json_path,
                 tts_device):
        self.piper_voice_onnx_path = piper_voice_onnx_path
        self.piper_voice_json_path = piper_voice_json_path
        self.tts_device = tts_device

    def load_model(self):
        try:
            print(f"Loading Piper TTS voice model onto {self.tts_device}...")
            if not os.path.exists(self.piper_voice_onnx_path) or not os.path.exists(self.piper_voice_json_path):
                print(f"Error: Cannot find Piper voice files. Please ensure the following files exist:")
                print(f"  - {self.piper_voice_onnx_path}")
                print(f"  - {self.piper_voice_json_path}")
                print(f"Please download from https://huggingface.co/rhasspy/piper-voices/tree/main and place them in the correct path.")
                sys.exit(1)
            tts_model = piper.PiperVoice.load(self.piper_voice_onnx_path, self.piper_voice_json_path)
            print("Piper TTS voice model loaded successfully.")
        except Exception as e:
            print(f"Error: Failed to load Piper TTS voice: {e}")
            sys.exit(1)
        
        return tts_model