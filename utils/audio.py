import sys
import queue
import tempfile
import sounddevice as sd
import soundfile as sf
import os

def audio_callback(audio_queue, 
                   indata, 
                   frames, 
                   time, 
                   status):
    """Sounddevice callback to put audio data into the queue."""
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def record_audio(audio_queue,
                 duration_seconds = 5,
                 sample_rate = 16000,
                 channels = 1,
                 audio_dtype = 'int16'):
    """Records audio for a specified duration and saves to a temporary file."""
    q = audio_queue # Use the provided audio queue
    # Clear the queue before starting recording
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break

    temp_filename = tempfile.mktemp(prefix='temp_audio_', suffix='.wav', dir='.')
    print(f"Starting recording ({duration_seconds} seconds)... Please speak...") 

    try:
        # Use soundfile.SoundFile for direct writing in the context manager
        with sf.SoundFile(temp_filename, mode='x', samplerate=sample_rate, channels=channels, subtype='PCM_16') as file:
            # Assume audio_callback is globally defined or passed correctly to InputStream
            with sd.InputStream(samplerate=sample_rate, channels=channels, dtype=audio_dtype, callback=audio_callback):
                # Keep the stream active for the specified duration
                print("Recording...")
                sd.sleep(int(duration_seconds * 1000)) # Use sd.sleep for duration

            # After the sleep duration, process data that accumulated in the queue
            print("Processing recorded data...") 
            while not q.empty():
                try:
                    # Write data from queue to the file
                    file.write(q.get_nowait())
                except queue.Empty:
                    break # Exit loop if queue is empty
                
        print(f"Recording finished, file saved as {temp_filename}") 
        return temp_filename
    
    except Exception as e:
        print(f"Recording failed: {e}") 
        # Cleanup if recording failed
        if os.path.exists(temp_filename):
            os.remove(temp_filename) 
        # Consider logging the full exception traceback for debugging
        # import traceback
        # traceback.print_exc()
        return