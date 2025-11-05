# **Legal Clause Extraction with Fine-Tuned Llama 3.2 3B**

[![Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/Munna-K/legal-clause-extractor-demo2)
[![Model](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/Munna-K/llama-3.2-3b-legal-clause-extractor)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

> **"Fine-tuned Llama 3.2 3B model for legal clause extraction achieving 73.6% accuracy and 39% excellent matches; improved baseline performance by 202% through dataset optimization; deployed production-ready application on HuggingFace with <2s latency; overcame 30+ technical challenges."**

---

## **Overview**

This project fine-tunes Meta's Llama 3.2 3B model for legal clause extraction using QLoRA on Kaggle T4 x2 GPUs. The model can extract specific clause types (indemnification, termination, liability, etc.) from legal contracts with high accuracy.

**Key Results:**
- **73.6% average similarity** between extracted and actual clauses
- **39% excellent matches** (>90% similarity)
- **<2 second inference latency** on single GPU
- **Production-ready deployment** on HuggingFace Spaces

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Training Loss | 0.407 |
| Average Similarity | 73.6% |
| Excellent (>90%) | 39% |
| Good (>70%) | 60% |
| Partial (>50%) | 78% |
| Inference Latency | ~1.2 seconds |
| Training Time | 174 minutes |

### Performance by Clause Type

| Clause Type | Avg Similarity | Excellent % |
|-------------|----------------|-------------|
| Indemnification | 76.2% | 42% |
| Termination | 75.1% | 41% |
| Limitation of Liability | 74.8% | 39% |
| Confidentiality | 72.3% | 37% |
| Governing Law | 71.9% | 35% |


---

## **Tech Stack**

### **Languages & Frameworks**

* **Python 3.10+** — Core programming language
* **PyTorch** — Deep learning framework for model training and inference
* **Transformers** — For model loading, tokenization, and text generation
* **PEFT (LoRA / QLoRA)** — Efficient fine-tuning with low memory usage
* **BitsAndBytes** — 4-bit quantization for large model training
* **Accelerate** — Manages multi-GPU and mixed-precision training

---

### **Model & Dataset**

* **Meta Llama 3.2 3B Instruct** — Base model fine-tuned for legal clause extraction
* **ACORD Dataset** — Expert-annotated legal clauses used for fine-tuning and evaluation

---

### **Deployment & Interface**

* **Gradio** — Interactive demo interface hosted on Hugging Face Spaces
* **Hugging Face Hub** — For model hosting and inference API integration

---

### **Tools & Infrastructure**

* **Kaggle Notebooks** — Training environment using dual T4 GPUs
* **difflib (SequenceMatcher)** — Evaluation metric for clause similarity
* **Git & GitHub** — Version control and project management

---


## **Quick Start**

### **Try the Demo**

Visit the [live demo](https://huggingface.co/spaces/Munna-K/legal-clause-extractor-demo2) to test the model instantly.

### **Use the Model**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = PeftModel.from_pretrained(base_model, "your-username/llama-3.2-3b-legal-clause-extractor")

# Extract clause
contract = """
This Agreement may be terminated by either Party upon thirty (30) days 
prior written notice to the other Party. Upon termination, all rights 
and obligations shall cease immediately.
"""

prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Copy the exact 'Termination' clause from this contract:

{contract}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## **Architecture**

### **Model Details**

- **Base Model:** Meta Llama 3.2 3B Instruct
- **Fine-Tuning Method:** QLoRA (4-bit quantization)
- **Trainable Parameters:** 48,627,712 (1.5% of total)
- **LoRA Configuration:**
  - Rank: 32
  - Alpha: 64
  - Dropout: 0.05
  - Target modules: All linear layers

### **Training Configuration**

- **Dataset:** ACORD (expert-annotated by lawyers)
- **Training samples:** ~6200 (after augmentation)
- **Hardware:** Kaggle T4 x2 (32GB total VRAM)
- **Training time:** 174.6 minutes (~2.9 hours)
- **Learning rate:** 1e-4
- **Epochs:** 3
- **Batch size:** 2 per GPU with gradient accumulation

---

## **Project Structure**

```
legal-llm-fine-tuning/
├── notebooks/
│   ├── week1_data_prep.ipynb          # Data conversion (BEIR → instruction format)
│   ├── week2_fine_tuning.ipynb        # Model training with QLoRA
│   └── week3_deployment.ipynb         # HuggingFace Hub upload
├── src/
│   ├── app.py                         # Gradio interface
│   ├── extraction.py                  # Extraction logic
│   └── utils.py                       # Helper functions
├── data/
│   ├── acord_train.json               # Training data
│   ├── acord_val.json                 # Validation data
│   └── acord_test.json                # Test data
├── models/
│   └── lora_adapters/                 # Fine-tuned LoRA adapters
├── README.md                          # This file
├── CHALLENGES.md                      # Challenges overcome
├── requirements.txt                   # Python dependencies
```

---

## **Installation**

### **Prerequisites**

- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM recommended)
- 32GB+ system RAM

### **Setup**

```bash
# Clone repository
git clone https://github.com/your-username/legal-llm-fine-tuning.git
cd legal-llm-fine-tuning

# Install dependencies
pip install -r requirements.txt

# Download model (automatic on first run)
python src/app.py
```

### **Requirements**

```
torch==2.0.0
transformers==4.46.3
peft==0.13.2
accelerate==1.11.0
bitsandbytes==0.45.5
gradio==5.0.0
trl==0.11.4
datasets==2.14.0

```

---

## **Usage Examples**

### Example 1: Extract Termination Clause

```python
contract = """
This Agreement may be terminated by either Party upon thirty (30) days 
prior written notice. Upon termination, all obligations cease immediately.
"""

result = extract_clause(contract, "Termination")
print(result)
# Output: "This Agreement may be terminated by either Party upon thirty (30) 
# days prior written notice. Upon termination, all obligations cease immediately."
```

### Example 2: Extract Indemnification Clause

```python
contract = """
Company A shall indemnify and hold harmless Company B from any and all 
claims, damages, losses, and expenses arising from this Agreement.
"""

result = extract_clause(contract, "Indemnification")
print(result)
# Output: "Company A shall indemnify and hold harmless Company B from any 
# and all claims, damages, losses, and expenses arising from this Agreement."
```

### Example 3: Clause Not Present

```python
contract = """
This is a simple contract with no specific clauses.
"""

result = extract_clause(contract, "Insurance")
print(result)
# Output: "There is no insurance clause present in this contract."
```

---

## **Methodology**

### **Data Preparation**

1. **Dataset Selection:** Switched from CUAD to ACORD for better label alignment
2. **Format Conversion:** Converted BEIR format to instruction-tuning format
3. **Data Augmentation:** 5 instruction templates × 3 = 15 variations per sample
4. **Train/Val/Test Split:** 70/15/15 with stratification

### **Training Process**

1. **Initial Attempt (CUAD):** 24.4% similarity - diagnosed data quality issue
2. **Dataset Switch:** Researched and switched to ACORD (expert-annotated)
3. **Hyperparameter Tuning:** Optimized learning rate, warmup, grad norm
4. **Final Training:** 73.6% similarity (+202% improvement)

### **Evaluation**

- **Similarity Metric:** SequenceMatcher ratio (Python difflib)
- **Tiers:** Excellent (>90%), Good (>70%), Partial (>50%)
- **Per-Clause Analysis:** Breakdown by clause type
- **Edge Cases:** Tested on missing clauses, ambiguous text

---

## **Challenges Overcome**

**Total:** 30+ technical challenges across 5 domains

**Critical Challenges:**
1. **CUAD Dataset Misalignment:** Labels didn't match inputs (24% → 73% with ACORD)
2. **Llama 3.2 Config Bug:** 404 error on model loading (fixed with error suppression)
3. **Hardware Constraints:** 3B model on 32GB GPU (solved with QLoRA + optimization)

**Full challenge list:** See [Challenges.md](Challenges.md)

**Skills Demonstrated:**
- Root cause analysis and problem diagnosis
- Research and alternative solution finding
- Hardware optimization and memory management
- Library dependency resolution
- Prompt engineering and instruction-tuning
- Deployment and MLOps

---

## **Results Comparison**

### Before vs After Dataset Optimization

| Metric | CUAD (Before) | ACORD (After) | Change |
|--------|---------------|---------------|--------|
| Training Loss | 1.478 | 0.407 | -72% |
| Avg Similarity | 24.4% | 73.6% | +202% |
| Excellent (>90%) | 2% | 39% | +1,850% |
| Good (>70%) | 4% | 60% | +1,400% |
| Status | ❌ Unusable | ✅ Production-ready | 🎉 |

---

## **Limitations**

1. **Language:** English only (no multi-lingual support)
2. **Context Length:** Truncates contracts >2000 characters
3. **Clause Types:** Limited to common commercial contract clauses
4. **Jurisdiction:** Trained primarily on US contracts
5. **Confidence:** Heuristic-based (not learned confidence model)

**Note:** This is a research/demo project, not a replacement for legal advice.

---

## **Future Work**

### Short-Term
- Multi-clause extraction (extract all clauses at once)
- Learned confidence scoring
- Edge case handling improvements

### Medium-Term
- Clause type auto-detection
- Multi-contract batch processing
- Fine-grained clause parsing

### Long-Term
- Cross-lingual support (Spanish, French, German)
- Domain expansion (employment, real estate, IP)
- REST API for enterprise integration

---

## **Acknowledgments**

- **Meta AI** for Llama 3.2 3B Instruct
- **The Atticus Project** for ACORD dataset
- **HuggingFace** for Transformers, PEFT, and hosting
- **Kaggle** for free GPU compute

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

**Note:** Llama 3.2 model follows Meta's license agreement.

---

## Contact

**Author:** Your Name  
**Email:** munna88mn@gmail.com  
**LinkedIn:** [LinkedIn](linkedin.com/in/munna-a4ab07253)  
**GitHub:** [@Munna-Git](https://github.com/Munna-Git)

---

## Citation

If you use this project, please cite:

```bibtex
@misc{legal-llm-2025,
  author = {Munna},
  title = {Legal Clause Extraction using Fine-Tuned Llama 3.2 3B},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Munna-Git/Fine-Tuned-Llama-3.2-3B.git}
}
```

---

## **⭐ Star History**

If you find this project useful, please consider giving it a star!

---
