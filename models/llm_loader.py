import sys, os
import torch
import piper #TTS
from .STT import STTModel
from .LLM import LLMModel
from .TTS import TTSModel

def load_models(piper_voice_onnx_path,
                piper_voice_json_path,
                llm_model_id = "mistralai/Mistral-7B-Instruct-v0.2",
                llm_device = "cuda",
                stt_model_size = "base",
                stt_device = "cpu",
                tts_device = "cpu",
                ):
    """Loads STT, LLM, and TTS models."""
    print("\n--- Starting model loading ---")

    # 1. Load STT (Whisper)
    stt_model = STTModel(stt_model_size,
                         stt_device).load_model()

    # 2. Load LLM (Quantized)
    llm_model, llm_tokenizer = LLMModel(llm_model_id, 
                                         llm_device).load_model()

    # 3. Load TTS (Piper)
    tts_model = TTSModel(piper_voice_onnx_path,
                         piper_voice_json_path,
                         tts_device).load_model()

    print("--- All models loaded successfully ---\n")
    return stt_model, llm_model, llm_tokenizer, tts_model

def main():
    PIPER_VOICE_ONNX_PATH = '../zh_CN-huayan-medium.onnx' # MODIFY THIS PATH
    PIPER_VOICE_JSON_PATH = '../zh_CN-huayan-medium.onnx.json' # MODIFY THIS PATH

    stt_model, llm_model, llm_tokenizer, tts_model = load_models(PIPER_VOICE_ONNX_PATH,
                                                                 PIPER_VOICE_JSON_PATH)

if __name__ == "__main__":
    main()