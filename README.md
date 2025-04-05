# Introduction
This LLM voice assistant operates as follows: First, it utilizes the Whisper model for Speech-to-Text (STT). The transcribed text is then passed to a Large Language Model (LLM) to generate a response. Considering the hardware limitation of an NVIDIA RTX 3070 Ti with 8GB VRAM, Mistral-7B-Instruct-v0.2 was selected as the core LLM due to its effective balance between performance and computational resource requirements. Finally, Piper handles Text-to-Speech (TTS) synthesis to output the LLM's response as audio.

# Demo
![image](https://github.com/user-attachments/assets/c43cd712-2528-479b-8a97-950bb7bcc2b7)

