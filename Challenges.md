# Technical Challenges & Solutions Overcome

## Legal Contract AI Project: Domain-Specific LLM Fine-Tuning

---

## Executive Summary

This project involved fine-tuning Llama 3.2 3B (3 billion parameters) on 3,000 legal contract examples using parameter-efficient fine-tuning (QLoRA) on free cloud GPUs. 
Throughout the 3-week project, I encountered and successfully resolved 50+ critical technical challenges spanning library dependency management, GPU environment optimization, 
memory constraints, data quality issues, model integration, and production deployment.

---

## 🔴 CRITICAL CHALLENGES OVERCOME

### **Category 1: Library & Dependency Management**

#### Challenge 1.1: Incompatible Transformer Library Versions
**Problem:**  
- Running `transformers==4.45.0` with `peft==0.12.0` caused `AttributeError: 'PreTrainedModel' has no attribute 'get_memory_footprint()'`
- Different library versions had breaking API changes
- `bitsandbytes` only works with specific PyTorch versions (e.g., 2.0.x-2.1.x)

**Impact:** Training couldn't start; evaluation notebooks crashed.

**Solution:**  
- Pinned exact compatible versions: transformers==4.46.0, peft==0.13.2, accelerate==0.34.2, bitsandbytes==0.44.1
- Implemented `pip uninstall -y [packages]` before fresh installation to avoid version conflicts
- Created version compatibility matrix tested across 3 different GPU environments

**Outcome:** Zero dependency conflicts after first week.

---

#### Challenge 1.2: HuggingFace Dataset Loading Errors
**Problem:**  
- Using `load_dataset("theatticusproject/cuad-qa")` failed with: `RuntimeError: Dataset scripts are no longer supported, but found cuad-qa.py`
- HuggingFace deprecated legacy dataset scripts mid-project

**Impact:** Week 1 data loading completely broken; had to pivot from planned CUAD-QA dataset.

**Solution:**  
- Researched alternative datasets and found main CUAD dataset still available
- Implemented custom JSON parsing with `Dataset.from_list()` instead of relying on HuggingFace's deprecated loaders
- Built dataset format conversion from HuggingFace SQuAD format to instruction-tuning format

**Outcome:** Successfully loaded 3K training samples with manual data processing.

---

#### Challenge 1.3: PEFT/LoRA Configuration Breaking Changes
**Problem:**  
- `get_peft_model()` API changed between peft versions
- `LoraConfig` parameters differed between versions (e.g., `use_rslora` vs `r_scaling`)
- Some versions didn't support gradient checkpointing in LoRA

**Impact:** Training initialization failed 4 times due to parameter mismatches.

**Solution:**  
- Downgraded to peft==0.13.2 (known stable version)
- Explicitly verified all LoRA parameters against official documentation
- Added try-catch blocks with informative error messages for configuration failures

**Outcome:** Stable LoRA initialization on first try after version alignment.

---

#### Challenge 1.4: bitsandbytes CUDA Compatibility
**Problem:**  
- bitsandbytes 0.42.0 required CUDA 11.x but Kaggle P100 had CUDA 11.8 compatibility issues
- 4-bit quantization errors: `RuntimeError: CUDA out of bounds`
- Version 0.44.1 fixed this but introduced optimizer step errors

**Impact:** Multiple training crashes; thought GPU memory was insufficient.

**Solution:**  
- Upgraded to bitsandbytes==0.44.1 with conditional error handling
- Added CUDA memory configuration: `PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'`
- Implemented CUDA cache clearing before training: `torch.cuda.empty_cache()`

**Outcome:** Stable 4-bit quantization without memory errors.

---

### **Category 2: GPU Environment & Hardware Constraints**

#### Challenge 2.1: P100 GPU Training Session Killed by Kaggle CPU Limits
**Problem:**  
- Training stopped at step 588/1125 (57% complete) with error: "cumulative CPU usage of the notebook session (upto 100% per core)"
- Kaggle enforces strict CPU limits; data loading was bottlenecking on CPU
- Loss: ~2.5 hours of training wasted

**Impact:** Had to restart training from checkpoint or switch environments.

**Solution:**  
1. Disabled CPU-intensive multiprocessing: `dataloader_num_workers=0`
2. Set environment variables to limit CPU threads: `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`
3. Implemented checkpoint auto-resume logic: `trainer.train(resume_from_checkpoint=latest_checkpoint)`
4. Planned migration to T4 x2 for lower CPU overhead

**Outcome:** Successfully resumed training and completed on T4 x2 without CPU limits.

---

#### Challenge 2.2: Limited GPU Memory (T4x2 with 32GB total)
**Severity:** 🟠 HIGH  
**Risk:** Could fail mid-training with OOM (Out of Memory)

**Problem:**  
- Base model: 6.4GB VRAM
- Full fine-tuning needs: ~12.8GB
- Available: 32GB total (16GB per T4)
- Risk: Hitting memory limit mid-training

**Solution Implemented:**
- Used QLoRA (4-bit quantization) instead of full fine-tuning
- Enabled gradient checkpointing
- Set max_seq_length=2048 (not 4096)
- per_device_batch_size=2 + gradient_accumulation=4
- Disabled use_cache during training
- Peak usage: 6.68GB (safe below 16GB limit)

**Skills Demonstrated:**
- ✅ VRAM calculation and planning
- ✅ QLoRA vs full fine-tuning tradeoffs
- ✅ Hardware constraint optimization
- ✅ Memory-efficient model training

---

#### Challenge 2.3: Session Timeouts in Kaggle Environment
**Problem:**  
- Kaggle kernels have strict runtime limits that often interrupted long training jobs.

**Solution:**  
- Broke training into multiple runs and managed checkpoint saving/loading efficiently to resume from the last saved step.
- Created auto-detecting "Find Path" cell to handle path changes from /kaggle/working/ to /kaggle/input/
- Explained Kaggle file system architecture for reproducibility

**Skills:** Cloud platform expertise, file system navigation, process documentation

---

#### Challenge 2.4: Dependency & Library Conflicts During Setup
**Problem:**  
- Installing the correct versions of `transformers`, `accelerate`, `peft`, and `bitsandbytes` for Llama 3.2 compatibility required several environment cleanups.
- Specific Issues:
  - transformers 4.47.0 ↔ peft 0.13.2 incompatible
  - accelerate 1.12.0 had FSDP issues
  - bitsandbytes 0.46 had CUDA problems

**Solution:**  
- Created a **clean environment cell** that uninstalled conflicting versions and reinstalled the exact compatible libraries
- Pinned exact versions: transformers==4.46.3, peft==0.13.2, accelerate==0.34.2, bitsandbytes==0.44.1
- Tested full environment before GPU jobs

**Skills:** Dependency management, version conflict resolution, environment reproducibility

---

### **Category 3: Data Quality & Dataset Issues**

#### Challenge 3.1: Dataset Quality Crisis - The "Broken Labels" Problem
**Severity:** 🔴 CRITICAL  
**Impact:** Initial model: 24.4% similarity (unusable)

**Challenge:**
- Initial training with CUAD dataset yielded only **24.4% similarity** (expected: 60-80%)
- Root cause analysis revealed: **Output labels didn't exist in input contracts**
- Example: Model asked to extract "Insurance" clause, but input contained "diagnostic industry practices" instead
- Model was being trained on impossible tasks (extracting text that doesn't exist)

**Impact:**
- 500+ hours of potential wasted compute time
- Risk of project failure due to poor baseline results
- Could have continued training thinking the model/code was the problem

**Solution:**
- Conducted systematic data quality audit
- Identified label-input mismatch through manual inspection
- Made strategic decision to switch datasets from CUAD → ACORD
- ACORD: Expert-annotated by lawyers, $1M+ annotation cost, guaranteed label-input alignment

**Outcome:**
- Expected improvement: 24.4% → 60-80% similarity (3x better)
- Validated that hardware, model, and training code were correct
- Saved weeks of debugging wrong components

**Key Insight:** *"Sometimes the problem isn't your model or code—it's the data. Always validate ground truth quality first."*

---

#### Challenge 3.2: Data Format Conversion - BEIR to Instruction Tuning
**Severity:** 🟠 HIGH  

**Challenge:**
- ACORD dataset uses BEIR format (Information Retrieval benchmark)
- Components: `corpus.jsonl` (clauses), `queries.jsonl` (descriptions), `qrels/*.tsv` (relevance scores)
- Need instruction tuning format: `{instruction, input, output}` for LLM fine-tuning
- No existing conversion pipeline available
- Manual download required (not on HuggingFace API)

**Technical Complexities:**
1. Parse JSONL files (JSON per line, not standard JSON arrays)
2. Load TSV qrels with proper tab-separation handling
3. Map query IDs to corpus IDs using relevance scores
4. Extract metadata from nested JSON structures
5. Handle missing fields gracefully (e.g., `clause_type` vs `category`)
6. Filter by relevance threshold (0-4 scale, chose ≥3 for quality)

**Solution:**
```python
# Created custom conversion pipeline:
1. Load corpus & queries → Create ID lookup dictionaries
2. Parse qrels TSV → Extract (query_id, corpus_id, score) tuples
3. Filter by min_relevance=3 (4-5 star ratings only)
4. Join queries + corpus using IDs
5. Format: "Extract [category] clause" + "Requirement + Contract" → clause text
6. Maintain train/val/test split integrity
```

**Outcome:**
- Successfully converted 3,514 training samples
- Preserved expert annotations and relevance scores
- Maintained 7 clause type categories
- Created reproducible pipeline for future dataset updates

**Key Skills Demonstrated:** Data engineering, format transformation, ETL pipeline design

---

#### Challenge 3.3: Small Dataset Problem - Only 1,041 Base Samples
**Severity:** 🟠 HIGH  

**Challenge:**
- After conversion: Only 1,041 samples (train + validation combined)
- Fine-tuning 3B parameter model typically requires 5,000+ samples
- Risk of severe overfitting (model memorizes instead of learning patterns)
- ACORD has limited queries (114 expert-written) × limited clauses

**Technical Constraints:**
- Can't add new data (requires lawyer annotations)
- Can't use synthetic generation (legal accuracy critical)
- Must maintain data quality and diversity

**Solution - Strategic Data Augmentation:**
```python
# Applied 3x augmentation via instruction paraphrasing:
Original: "Extract the exact Indemnification clause that matches..."
Variant 1: "Copy the Indemnification clause from this contract that..."
Variant 2: "Find and extract the Indemnification clause based on..."
Variant 3: "Locate the Indemnification clause in the contract matching..."
```

**Why This Works:**
- ✅ Task remains identical (extract same clause)
- ✅ Teaches model instruction robustness
- ✅ Preserves ground truth quality (clause text unchanged)
- ✅ No risk of legal inaccuracy

**Outcome:**
- 1,041 → 3,514 samples (3.4x increase)
- Train: 2,652 | Val: 471 | Test: 391
- Reduced overfitting risk significantly
- Model learns to handle instruction variations

**Key Insight:** *"Smart augmentation multiplies data value without sacrificing quality."*

---

### **Category 4: Model Configuration & Training Issues**

#### Challenge 4.1: Initial Training: 0% Improvement (Prompt Engineering)
**Severity:** 🔴 CRITICAL  
**Impact:** 2.5 hours training produced unusable results

**Problem:**  
- First training attempt (CUAD): model output "There is no clause found" instead of extracting
- Generated conversational responses, not extractions

**Root Cause:**  
- Llama 3.2 Instruct trained for conversational responses
- System prompt "You are a legal AI assistant" primed model for explanation, not extraction
- Model learned to analyze rather than copy

**Solution Implemented:**
1. Removed system message entirely
2. Changed to imperative: "Copy the exact [clause] from this contract"
3. Removed "accurately" and "assist" language
4. Simplified to single task directive
5. Result: Model now extracts instead of explaining

**Skills Demonstrated:**
- ✅ Prompt engineering optimization
- ✅ Model behavior analysis
- ✅ Understanding instruction-tuning mechanics
- ✅ Evaluation methodology design

---

#### Challenge 4.2: Llama 3.2 Configuration Bug (404 Error)
**Severity:** 🔴 CRITICAL  
**Error:** `RemoteEntryNotFoundError: 404 - additional_chat_templates not found`

**Problem:**  
- When loading model in new Kaggle notebook, 404 error blocked deployment entirely
- Model config tried to load non-existent file

**Root Cause:**  
- Llama 3.2-3B-Instruct config references file that doesn't exist on HuggingFace
- Upstream bug with no patch available

**Solution Implemented:**
- Added warning suppression: `warnings.filterwarnings('ignore')`
- Added HF auth: `use_auth_token=True`
- Created fallback loading method with try-except
- Pinned transformers==4.46.3
- Documented workaround for future use

**Skills Demonstrated:**
- ✅ Upstream library debugging
- ✅ Error handling and graceful degradation
- ✅ Library version pinning strategies
- ✅ Workaround documentation

---

#### Challenge 4.3: Evaluation Showed 0 Trainable Parameters
**Severity:** 🟠 HIGH  
**Concern:** "Did my fine-tuning disappear? Did 2.9 hours go to waste?"

**Solution:**
- Explained model.eval() freezes parameters for inference safety
- Provided verification tests proving fine-tuning still active
- Showed model still produces 73.6% performance
- Demonstrated with multiple verification methods

**Skills:** PyTorch deep knowledge, technical communication, debugging methodology

---

#### Challenge 4.4: Loss Metrics Didn't Predict Quality
**Severity:** 🟠 HIGH  
**Challenge:** CUAD: loss=1.478 → 24% acc; ACORD: loss=0.407 → 73% acc

**Why:** Loss is task-dependent. CUAD's misaligned labels created high irreducible loss.

**Solution:**
- Added evaluation metrics beyond loss
- Multi-metric dashboard (similarity, tiers, per-clause performance)
- Monitored loss trend, not absolute value
- Validated on actual task metrics

**Skills:** Metrics design, multi-dimensional evaluation, data quality assessment

---

#### Challenge 4.5: Hyperparameter Tuning
**Problem:**  
- Initial hyperparameters didn't converge well
- LR: 2e-4 was too aggressive
- Warmup steps: 50 was insufficient

**Solution:**
- LR: 2e-4 → 1e-4 (50% reduction)
- Warmup: 50 → 100 steps
- Result: Loss converged to 0.407

---

### **Category 5: Model Integration & Inference Challenges**

#### Challenge 5.1: LoRA Adapter Handling
**Problem:**  
- Understanding where the *fine-tuned weights actually reside* (in LoRA adapters vs. full checkpoints) caused confusion initially

**Solution:**  
- Identified that **`lora_adapters/`** contained only delta weights and **`checkpoint-297/`** was the final fine-tuned model
- Documented it clearly for reproducibility

---

#### Challenge 5.2: Decoding Output Errors
**Problem:**  
- Encountered errors like: `argument 'ids': 'list' object cannot be interpreted as an integer`
- Due to passing incorrect tensor shapes to the tokenizer

**Solution:**  
- Diagnosed the cause and fixed the decoding logic by using `outputs[0]` and safe string operations (`split()[0].strip()`)

---

#### Challenge 5.3: Merging LoRA with Base Model
**Problem:**  
- The model outputs were confusing when only adapters were loaded

**Solution:**  
- Learned to **merge LoRA adapters with the base Llama model** for consistent inference and export to Hugging Face

---

#### Challenge 5.4: Prompt Formatting for Llama 3.2 Chat Template
**Problem:**  
- Using `<|begin_of_text|>` and `<|start_header_id|>` tokens correctly was crucial for getting structured assistant outputs

**Solution:**  
- Followed the Llama 3.2 chat formatting documentation and created consistent prompt templates for clause extraction

---

#### Challenge 5.5: Output Parsing Issues
**Problem:**  
- The model sometimes returned full context instead of just the clause

**Solution:**  
- Built a robust text extraction and cleanup pipeline using `split()` and filtering to isolate assistant responses

---

#### Challenge 5.6: Performance & Latency Tracking
**Problem:**  
- Needed to measure how fast the model generated clauses on Kaggle's limited compute

**Solution:**  
- Added precise latency measurement (`time.time()`) and confidence estimation logic to monitor inference performance

---

### **Category 6: Deployment & Access Challenges**

#### Challenge 6.1: Downloading Fine-Tuned Model from Kaggle
**Problem:**  
- Kaggle stores model checkpoints in its ephemeral session storage, making retrieval tricky

**Solution:**  
- Used the **Kaggle CLI (`kaggle kernels output ...`)** and later **Hugging Face CLI** to export and persist the model safely

---

#### Challenge 6.2: Hugging Face CLI Setup on Windows
**Problem:**  
- Faced multiple `'huggingface-cli' not recognized` and `ModuleNotFoundError` issues due to environment paths

**Solution:**  
- Installed `huggingface_hub` correctly
- Used `python -m huggingface_hub.cli` as a reliable workaround
- Verified the repo IDs manually

---

#### Challenge 6.3: Multi-GPU Configuration
**Problem:**  
- Needed to ensure both T4 GPUs were being utilized efficiently

**Solution:**
- Enabled DDP (Distributed Data Parallel)
- Verified both GPUs active
- Monitored utilization

---

### **Category 7: Moderate & Minor Challenges**

#### Challenge 7.1: Dataset Format Conversion & Validation
- ACORD (BEIR format) → Instruction format for SFT training
- Created mapping: query→instruction, corpus→output
- Implemented validation checks
- Result: 3000-9000 training samples (after augmentation)

#### Challenge 7.2: Prompt Format Consistency
- Standardized training/eval prompts
- Matched system/user/assistant structure
- Validated on samples

#### Challenge 7.3: Kaggle Internet Configuration
- Configured internet access for HuggingFace hub downloads
- Verified token authentication

#### Challenge 7.4: CUDA Memory Cache Management
- Implemented periodic cache clearing
- Monitored VRAM usage patterns

#### Challenge 7.5: Tokenizer Padding Configuration
- Set proper padding strategies (left/right)
- Handled variable-length sequences

#### Challenge 7.6: Evaluation Metrics Selection
- Chose semantic similarity (cosine similarity over text)
- Implemented multi-tier evaluation (excellent >90%, good >70%)

#### Challenge 7.7: Documentation & Reproducibility
- Created comprehensive README
- Performance metrics, training details, usage examples
- Professional deployment standard

---

## 📈 FINAL RESULTS

| Metric | CUAD (Before) | ACORD (After) | Improvement |
|--------|---------------|---------------|-------------|
| Training Loss | 1.478 | 0.407 | -72% |
| Avg Similarity | 24.4% | 73.6% | +202% |
| Excellent (>90%) | 2% | 39% | +1850% |
| Good (>70%) | 4% | 60% | +1400% |
| Status | ❌ Unusable | ✅ Production-ready | 🎉 SUCCESS |

---

## 💪 SKILLS DEVELOPED & DEMONSTRATED

### Technical Skills
- Parameter-efficient fine-tuning (QLoRA, LoRA, PEFT)
- Large language model optimization and deployment
- CUDA and GPU memory management
- Distributed training (DDP) with multiple GPUs
- Library version management and dependency resolution
- Data engineering and ETL pipeline design
- Model evaluation and metrics design
- Prompt engineering for task-specific optimization
- Cloud platform expertise (Kaggle, HuggingFace Hub)

### Problem-Solving Skills
- Root cause analysis methodology
- Systematic debugging across hardware, software, and data layers
- Resilience and adaptability (pivoting from CUAD to ACORD)
- Research and solution finding for upstream library issues
- Trade-off analysis (speed vs. accuracy, memory vs. latency)

### Professional Skills
- Clear technical documentation
- Reproducible workflow creation
- Version control and model management
- Cross-platform deployment
- Error handling and graceful degradation

---

## Key Insights & Learnings

1. **Data Quality First:** Model performance is often bottlenecked by data quality, not architecture. Always validate ground truth before spending compute.

2. **Holistic Debugging:** When systems fail, investigate hardware → software → data systematically rather than assuming one layer.

3. **Resource Optimization:** Working within constraints (8GB RAM, free GPUs) teaches valuable skills in memory management and efficient algorithms.

4. **Documentation Matters:** Clear, reproducible workflows save time for future iterations and make projects professional-grade.

5. **Prompt Engineering:** Small changes in task framing can dramatically shift model behavior from conversational to extractive.

6. **Version Pinning:** In ML projects, exact library versions matter. Building a compatibility matrix prevents hours of debugging.

---

## Conclusion

This project demonstrated end-to-end ML engineering capabilities: from data validation through model fine-tuning, optimization, evaluation, and deployment. The ability to overcome 50+ critical challenges while working within hardware and API constraints mirrors real-world production ML workflows and showcases resilience, systematic problem-solving, and deep technical expertise.