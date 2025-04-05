from utils import *
from models import *
import torch
import queue
import sounddevice as sd
from time import sleep
import os

# --- Configuration ---
# LLM
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2" # Or other suitable 7B model
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# STT (Whisper)
STT_MODEL_SIZE = "base" # Choose from 'tiny', 'base', 'small', 'medium', 'large' (.en for English only)
STT_DEVICE = "cpu" # Prioritize CPU to save VRAM for LLM

# TTS (Piper)
PIPER_VOICE_ONNX_PATH = './zh_CN-huayan-medium.onnx' # MODIFY THIS PATH
PIPER_VOICE_JSON_PATH = './zh_CN-huayan-medium.onnx.json' # MODIFY THIS PATH
TTS_DEVICE = "cpu" # Piper runs on CPU

# Audio Recording
SAMPLE_RATE = 16000 # Whisper requires 16kHz
CHANNELS = 1
AUDIO_DTYPE = 'int16'
RECORDING_DURATION_SECONDS = 5 # Duration for each recording

def main():
    audio_queue = queue.Queue()
    try:
        stt_model, llm_model, llm_tokenizer, tts_model = load_models(PIPER_VOICE_ONNX_PATH,
                                                                    PIPER_VOICE_JSON_PATH,
                                                                    LLM_MODEL_ID,
                                                                    LLM_DEVICE,
                                                                    STT_MODEL_SIZE,
                                                                    STT_DEVICE,
                                                                    TTS_DEVICE)
        print("\n--- Voice Assistant Ready ---")
        print("Hint: Press Enter to start recording for approx. 5 seconds. Press Ctrl+C to exit.")
        while True:
            try:
                # Simple text input loop instead of audio recording
                input("Press Enter to start recording...") # Original trigger
                audio_file = record_audio(audio_queue,
                                          duration_seconds=RECORDING_DURATION_SECONDS,
                                          sample_rate = SAMPLE_RATE,
                                          channels = CHANNELS,
                                          audio_dtype = AUDIO_DTYPE) # Original recording call
                # # Original STT call - replaced by text input
                if audio_file:
                    # 1. Speech to Text
                    user_text = speech_to_text(stt_model = stt_model,
                                               audio_path = audio_file,
                                               device = STT_DEVICE)
                else:
                    print("Recording failed, falling back to text input")
                    user_text = input("Please enter your question: ") # Text input trigger
                
                # Check if text input is valid
                if user_text and len(user_text.strip()) > 1:
                    # 2. LLM Processing
                    response = get_llm_response(llm_model = llm_model,
                                                llm_tokenizer = llm_tokenizer,
                                                device = LLM_DEVICE,
                                                user_prompt = user_text)

                    # 3. Text to Speech
                    text_to_speech_and_play(tts_model = tts_model,
                                            text = response)
                elif user_text != None: # Handle empty input vs failed recording
                    print("Input text is too short.")
                # else: # This case is now unlikely with direct text input
                #      print("Could not recognize valid speech command or result too short.")

                # Cleanup recorded audio file (no longer needed with text input)
                try:
                    if audio_file and os.path.exists(audio_file):
                         os.remove(audio_file)
                except OSError as e:
                     print(f"Failed to delete recording temporary file: {e}")

                print("\nWaiting for the next command...")


            except EOFError: # Handle case where input stream is closed
                print("\nDetected end of input, shutting down...")
                break

            except Exception as e: # Catch unexpected errors in the loop
                print(f"\nError in main loop: {e}")
                print("Attempting to continue...")
                import traceback
                traceback.print_exc() # Print traceback for loop errors
                sleep(2) # Pause briefly before potentially retrying


    except KeyboardInterrupt:
        print("\nDetected Ctrl+C, shutting down voice assistant...")
    except Exception as e:
        print(f"\nCritical error during startup or model loading: {e}")
        import traceback
        traceback.print_exc() # Print traceback for critical errors
    finally:
        print("Closing audio stream (if active)...")
        try:
            sd.stop() # Stop any active audio streams
        except Exception as e:
            print(f"Ignoring error while stopping sounddevice: {e}")
        print("Program finished.")

        #Delete models and clear cache if needed
        del stt_model
        del llm_model
        del llm_tokenizer
        del tts_model
        del audio_queue
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
             torch.cuda.empty_cache()
    

if __name__ == "__main__":
    main()

