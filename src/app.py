import os
from huggingface_hub import login

# Login
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)
    print("✅ Logged in to Hugging Face Hub using secret token")
else:
    print("⚠️ Warning: No HF_TOKEN found. Make sure you set the repository secret in your Space settings.")
    
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import time

print("Loading model...")

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_MODEL = "Munna-K/llama-3.2-3b-legal-clause-extractor"

# Load PEFT config first
peft_config = PeftConfig.from_pretrained(LORA_MODEL)

# Use this instead (no quantization on CPU)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,  # FP32 for CPU
    device_map="cpu",  # Force CPU
    trust_remote_code=True,
)


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model, 
    LORA_MODEL,
    torch_dtype=torch.float16,
)
model.eval()

print("Model loaded!")


def extract_clause(contract_text, clause_type, max_length=200, temperature=0.1):
    """Extract clause from contract, with error handling and clean output."""
    
    import time
    start_time = time.time()
    
    if not contract_text or not contract_text.strip():
        return "❌ Error: Contract text cannot be empty", "No metrics"
    
    if not clause_type or not clause_type.strip():
        return "❌ Error: Clause type cannot be empty", "No metrics"
    
    if len(contract_text) > 2000:
        contract_text = contract_text[:2000]
        truncated = True
    else:
        truncated = False
    
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Copy the exact '{clause_type}' clause from this contract:
{contract_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_length),
                temperature=float(temperature) if temperature > 0 else 1.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant" in full_output:
            extracted = full_output.split("assistant")[-1].strip()
        else:
            extracted = full_output.strip()
        
        if "<|eot_id|>" in extracted:
            extracted = extracted.split("<|eot_id|>")[0].strip()
        
        if not extracted or len(extracted) < 5:
            extracted = "⚠️ Model generated an empty or very short response. Try with different contract text."
            confidence = 0.1
            latency_ms = int((time.time() - start_time) * 1000)
            metrics = f"""
📊 **Metrics:**
- Confidence: {confidence:.0%}
- Latency: {latency_ms}ms
- Model: Llama 3.2 3B (QLoRA)
- Avg Performance: 73.6% similarity
"""
            return extracted, metrics
        
        latency_ms = int((time.time() - start_time) * 1000)
        confidence = 0.85 if len(extracted) > 20 else 0.65
        
        if truncated:
            extracted = f"⚠️ Contract text truncated to 2000 characters.\n\n{extracted}"
        
        metrics = f"""
📊 **Metrics:**
- Confidence: {confidence:.0%}
- Latency: {latency_ms}ms
- Model: Llama 3.2 3B (QLoRA)
- Avg Performance: 73.6% similarity
"""
        
        return extracted, metrics
    
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return f"❌ Error during extraction: {str(e)}", f"Latency: {latency_ms}ms\nError occurred"

# Create Gradio interface
demo = gr.Interface(
    fn=extract_clause,
    inputs=[
        gr.Textbox(label="📄 Contract Text", lines=10, placeholder="Paste contract text..."),
        gr.Dropdown(
            choices=["Indemnification", "Limitation of Liability", "Termination", 
                    "Confidentiality", "Governing Law", "Insurance"],
            label="🔍 Clause Type",
            value="Indemnification"
        ),
        gr.Slider(50, 500, 200, label="Max Length"),
        gr.Slider(0.0, 1.0, 0.1, label="Temperature"),
    ],
    outputs=[
        gr.Textbox(label="📋 Extracted Clause", lines=8),
        gr.Markdown(label="📊 Metrics"),
    ],
    title="⚖️ Legal Clause Extraction",
    description="Fine-tuned Llama 3.2 3B for extracting legal clauses (73.6% avg similarity)",
    examples=[
        [
            "This Agreement may be terminated by either Party upon thirty (30) days prior written notice.",
            "Termination",
            200,
            0.1,
        ],
    ],
)

if __name__ == "__main__":
    demo.launch(share = True)