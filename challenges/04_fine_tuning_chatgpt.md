

## 🚀 **Challenges Overcome During Fine-Tuning a Legal Domain Llama 3.2 Model**

### **1️⃣ Hardware & Environment Challenges**

* ⚙️ **Limited GPU Memory (T4x2)** —
  Fine-tuning a 3B parameter model required optimization to fit within Kaggle’s 32 GB GPU limit.
  ✅ *Solution:* Used **gradient checkpointing**, **8-bit quantization**, and **LoRA adapters** to drastically reduce VRAM usage.

* 🧩 **Session Timeouts in Kaggle Environment** —
  Kaggle kernels have strict runtime limits that often interrupted long training jobs.
  ✅ *Solution:* Broke training into multiple runs and managed checkpoint saving/loading efficiently to resume from the last saved step.

* ⚡ **Dependency & Library Conflicts** —
  Installing the correct versions of `transformers`, `accelerate`, `peft`, and `bitsandbytes` for Llama 3.2 compatibility required several environment cleanups.
  ✅ *Solution:* Created a **clean environment cell** that uninstalled conflicting versions and reinstalled the exact compatible libraries.

---

### **2️⃣ Model Integration Challenges**

* 🧠 **LoRA Adapter Handling** —
  Understanding where the *fine-tuned weights actually reside* (in LoRA adapters vs. full checkpoints) caused confusion initially.
  ✅ *Solution:* Identified that **`lora_adapters/`** contained only delta weights and **`checkpoint-297/`** was the final fine-tuned model; documented it clearly for reproducibility.

* 🪄 **Decoding Output Errors** —
  Encountered errors like:

  ```
  argument 'ids': 'list' object cannot be interpreted as an integer
  ```

  due to passing incorrect tensor shapes to the tokenizer.
  ✅ *Solution:* Diagnosed the cause and fixed the decoding logic by using `outputs[0]` and safe string operations (`split()[0].strip()`).

* 🔄 **Merging LoRA with Base Model** —
  The model outputs were confusing when only adapters were loaded.
  ✅ *Solution:* Learned to **merge LoRA adapters with the base Llama model** for consistent inference and export to Hugging Face.

---

### **3️⃣ Code & Inference Challenges**

* 🧾 **Prompt Formatting for Llama 3.2 Chat Template** —
  Using `<|begin_of_text|>` and `<|start_header_id|>` tokens correctly was crucial for getting structured assistant outputs.
  ✅ *Solution:* Followed the Llama 3.2 chat formatting documentation and created consistent prompt templates for clause extraction.

* 💬 **Output Parsing Issues** —
  The model sometimes returned full context instead of just the clause.
  ✅ *Solution:* Built a robust text extraction and cleanup pipeline using `split()` and filtering to isolate assistant responses.

* 🕐 **Performance & Latency Tracking** —
  Needed to measure how fast the model generated clauses on Kaggle’s limited compute.
  ✅ *Solution:* Added precise latency measurement (`time.time()`) and confidence estimation logic to monitor inference performance.

---

### **4️⃣ Deployment & Access Challenges**

* 🌐 **Downloading Fine-Tuned Model from Kaggle** —
  Kaggle stores model checkpoints in its ephemeral session storage, making retrieval tricky.
  ✅ *Solution:* Used the **Kaggle CLI (`kaggle kernels output ...`)** and later **Hugging Face CLI** to export and persist the model safely.

* 🧩 **Hugging Face CLI Setup on Windows** —
  Faced multiple `‘huggingface-cli’ not recognized` and `ModuleNotFoundError` issues due to environment paths.
  ✅ *Solution:* Installed `huggingface_hub` correctly, used `python -m huggingface_hub.cli` as a reliable workaround, and verified the repo IDs manually.

---

### **5️⃣ Learning & Optimization Takeaways**

* 💡 Learned to manage **large-model fine-tuning efficiently on limited resources**.
* 💡 Developed an **error-handling wrapper** (`try–except`) to capture and return detailed inference errors gracefully.
* 💡 Built a deeper understanding of **transformers’ tokenization pipeline**, model generation logic, and **LoRA-based fine-tuning**.
* 💡 Gained hands-on experience in **exporting models to Hugging Face Hub**, ensuring public accessibility and version tracking.

---

### ✅ **Summary (What It Proves to Recruiters or Managers)**

> You didn’t just run a notebook — you **engineered a solution**.
> You handled hardware limits, debugging, model architecture understanding, prompt design, and deployment — which mirrors **real-world ML workflow challenges** in production teams.

---

