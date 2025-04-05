import piper # Assuming piper-tts library is installed
import sys
import torch

# Check if piper needs different import based on installation
def check_module():
    try:
        if not hasattr(piper, 'PiperVoice'):
            # Try alternative import if standard doesn't work (might vary)
            from piper.voice import PiperVoice
            piper.PiperVoice = PiperVoice
    except ImportError:
        print("Error: Cannot import PiperVoice from 'piper' or 'piper.voice'. Please check your piper-tts installation.")
        sys.exit(1)
    except AttributeError:
        # If piper exists but PiperVoice is not directly under it,
        # assume the first import worked or handle other structures if known.
        if not hasattr(piper, 'PiperVoice'):
            print("Error: Cannot find PiperVoice class. Please check your piper-tts installation and version.")

def check_version():
    # Check PyTorch and CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

def main():
    check_module()
    check_version()
    
if __name__ == "__main__":
    main()