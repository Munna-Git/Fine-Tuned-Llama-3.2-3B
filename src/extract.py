# ══════════════════════════════════════════════════════════════
# CELL 3: Create Extraction Function (FIXED VERSION)
# ══════════════════════════════════════════════════════════════

import time
import torch

def extract_clause(contract_text, clause_type, max_length=200, temperature=0.1):
    """
    Extract a specific clause from contract text.
    
    Args:
        contract_text: The full contract text
        clause_type: Type of clause to extract (e.g., 'Indemnification')
        max_length: Maximum length of generated text
        temperature: Sampling temperature (0.1 = more focused)
    
    Returns:
        tuple: (extracted_clause, confidence, latency_ms)
    """
    
    start_time = time.time()
    
    # Validate inputs
    if not contract_text or not contract_text.strip():
        return "❌ Error: Contract text cannot be empty", 0.0, 0
    
    if not clause_type or not clause_type.strip():
        return "❌ Error: Clause type cannot be empty", 0.0, 0
    
    # Truncate long contracts
    if len(contract_text) > 2000:
        contract_text = contract_text[:2000]
        truncated = True
    else:
        truncated = False
    
    # Format prompt
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Copy the exact '{clause_type}' clause from this contract:

{contract_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    try:
        # Tokenize
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_length),
                temperature=float(temperature) if temperature > 0 else 1.0,
                do_sample=False,  # Always deterministic for extraction
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode - FIX: outputs[0] instead of outputs
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "assistant" in full_output:
            extracted = full_output.split("assistant")[-1].strip()
        else:
            extracted = full_output.strip()
        
        # Clean up - FIX: [0] before .strip()
        if "<|eot_id|>" in extracted:
            extracted = extracted.split("<|eot_id|>")[0].strip()
        
        # Validate response is not empty
        if not extracted or len(extracted) < 5:
            extracted = "⚠️ Model generated empty or very short response. Try with different contract text."
            confidence = 0.1
            latency_ms = int((time.time() - start_time) * 1000)
            return extracted, confidence, latency_ms
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Estimate confidence (based on response characteristics)
        if len(extracted) > 40 and not extracted.startswith("I cannot") and not extracted.startswith("There is no"):
            confidence = 0.85
        elif len(extracted) > 20:
            confidence = 0.65
        elif len(extracted) > 10:
            confidence = 0.45
        else:
            confidence = 0.25
        
        # Add truncation warning if needed
        if truncated:
            extracted = f"⚠️ Contract text truncated to 2000 characters.\n\n{extracted}"
        
        return extracted, confidence, latency_ms
    
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        error_msg = f"❌ Error during extraction: {str(e)}\n\nDebug info:\n- Contract length: {len(contract_text)}\n- Clause type: {clause_type}"
        return error_msg, 0.0, latency_ms
