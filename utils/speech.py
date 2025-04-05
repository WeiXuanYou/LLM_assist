import tempfile
import sounddevice as sd
import os
import soundfile as sf
import sys # Added for potential stderr output, useful for debugging
import wave # Added import based on previous Piper fixes, though not used in this specific version of TTS func

# Assuming 'piper' and 'whisper' libraries are imported elsewhere if needed by the models passed

def speech_to_text(stt_model,
                   audio_path,
                   stt_device = 'cpu'):
    """Transcribes audio file to text using Whisper."""
    if stt_model is None or audio_path is None:
        return ""
    print("Performing speech recognition...") 
    try:
        # Use fp16=False if running on CPU
        result = stt_model.transcribe(audio_path, fp16=(stt_device != "cpu"))
        transcribed_text = result['text']
        print(f"Recognition result: {transcribed_text}") 
        return transcribed_text
    except Exception as e:
        print(f"Speech recognition failed: {e}") 
        # Consider logging the full exception traceback for debugging
        # import traceback
        # traceback.print_exc()
        return ""

def get_piper_model_config(tts_model):
    """Attempt to get audio parameters from the loaded Piper voice model config."""
    sample_rate = 16000 # Default if not found
    sample_width = 2   # 16-bit
    channels = 1       # Mono

    if hasattr(tts_model, 'config'):
        voice_config = getattr(tts_model, 'config', None)
        if voice_config:
            sample_rate = getattr(voice_config, 'sample_rate', sample_rate)
            # Usually Piper is 16-bit mono, but check if config provides info
            # sample_width = getattr(voice_config, 'sample_width', sample_width)
            # channels = getattr(voice_config, 'num_channels', channels)
            print(f"Read Sample Rate from Piper config: {sample_rate}")
        else:
            print("Warning: Piper voice object has config attribute but it's empty.")
    else:
        print("Warning: Piper voice object does not have config attribute, using default audio parameters.")

    print(f"Using Piper TTS parameters: Rate={sample_rate}, Width={sample_width}, Channels={channels}")
    return sample_rate, sample_width, channels

def text_to_speech_and_play(tts_model, text):
    """Synthesizes text to speech (e.g., using Piper) and plays it."""
    if tts_model == None:
        print("Error: TTS voice not loaded.") 
        return
    if not text:
        print("No text to synthesize.") 
        return

    print("Generating speech...") 
    output_wav_path = tempfile.mktemp(prefix='response_audio_', suffix='.wav', dir='.')
    
  
    try:
        # Get audio parameters from the Piper model config
        sample_rate, sample_width, channels = get_piper_model_config(tts_model)

        # Write the returned audio data to a file
        with wave.open(output_wav_path, "wb") as wav_file:
            # Set WAV parameters before synthesis
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            
            # Synthesize using the provided TTS model
            tts_model.synthesize(text, wav_file = wav_file)

        print(f"Speech generation complete: {output_wav_path}") 

        # Play the generated audio
        print("Playing response...") 
        # Check if file has content before playing
        if os.path.getsize(output_wav_path) > 0:
            data, fs = sf.read(output_wav_path, dtype='float32')
            try:
                sd.play(data, fs)
                sd.wait() # Wait for playback to finish
                print("Playback finished.")
            except Exception as e:
                print(f"Playback failed: {e}")  
        else:
            print("Skipping playback: Generated audio file is empty.")


    except AttributeError as e:
         # Catch errors if synthesize doesn't return expected data type
         print(f"TTS failed: Potential issue with synthesize method's return type or usage: {e}")
         import traceback
         traceback.print_exc()
    except Exception as e:
        print(f"TTS or playback failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up the temporary audio file
        try:
            if os.path.exists(output_wav_path):
                os.remove(output_wav_path)
        except OSError as e:
            print(f"Failed to delete TTS temporary file: {e}") 