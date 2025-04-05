import torch # Assuming torch is imported elsewhere if needed
# Assuming llm_model, llm_tokenizer, and LLM device are defined globally or passed as arguments

def get_llm_response(llm_model,
                     llm_tokenizer,
                     device,
                     user_prompt):
    """Generates a response from the LLM based on the user prompt."""
    # Ensure models are accessible (replace with actual check if they are passed as args)
    if llm_model == None or llm_tokenizer == None:
        return "Sorry, the language model is not loaded." 
    if not user_prompt:
        return "Please provide input first." 

    print("Generating LLM response...") 
    # Prepare prompt for instruct model
    messages = [{"role": "user", "content": user_prompt}]
    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Determine input device based on LLM device config and CUDA availability
    input_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    # Ensure model parts are on the correct device if using manual placement,
    # device_map="auto" usually handles this.
    # llm_model.to(input_device) # Typically not needed with device_map="auto"

    # Ensure inputs are sent to the device where the model expects them
    # For device_map="auto", model.device might point to the first device (often GPU 0 or CPU)
    try:
        effective_input_device = llm_model.device # Use the device reported by the model object
    except AttributeError:
        effective_input_device = input_device # Fallback if model has no .device attribute
        print(f"Warning: Could not determine model device from model object, using configured LLM_DEVICE: {effective_input_device}")

    inputs = llm_tokenizer(prompt, return_tensors="pt").to(effective_input_device)


    try:
        # Generate response
        # Use torch.no_grad() for inference to save memory
        with torch.inference_mode():
             outputs = llm_model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                pad_token_id=llm_tokenizer.eos_token_id
             )
        # Decode the generated tokens, skipping the prompt part
        response_text = llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"LLM response: {response_text}") 
        return response_text
    
    except Exception as e:
        print(f"LLM generation failed: {e}") 
        # Potential OOM error might happen here
        if "out of memory" in str(e).lower() or "CUDA error: out of memory" in str(e):
            print("Error: GPU out of memory! Try closing other applications or using smaller models/settings.") 
        # Consider logging the full traceback for debugging
        # import traceback
        # traceback.print_exc()
        return "Sorry, I encountered a problem processing your request." 
    
    finally:
         # Clean up GPU memory proactively if using CUDA
        if effective_input_device == "cuda":
            # Delete tensors that were explicitly moved to GPU
            try:
                del inputs
                del outputs
            except NameError: # In case generation failed before outputs was assigned
                 pass
            # Force garbage collection and empty cache ONLY if memory is extremely tight,
            # as it can slow down subsequent inferences.
            # import gc
            # gc.collect()
            # torch.cuda.empty_cache()
