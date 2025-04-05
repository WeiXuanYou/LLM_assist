import sys
class STTModel():
    def __init__(self, stt_model_size = "base",
                 stt_device = "cpu"):
        self.stt_model_size = stt_model_size
        self.stt_device = stt_device

    def load_model(self):
        
        try:
            print(f"Loading Whisper STT model ({self.stt_model_size}) onto {self.stt_device}...")
            import whisper # Import here to ensure availability check passes first
            stt_model = whisper.load_model(self.stt_model_size, device = self.stt_device)
            print("Whisper STT model loaded successfully.")
        except Exception as e:
            print(f"Error: Failed to load Whisper STT model: {e}")
            sys.exit(1)

        return stt_model