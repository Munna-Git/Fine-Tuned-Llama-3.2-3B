# 🎯 CHALLENGES OVERCOME: Legal LLM Fine-Tuning Project
## Portfolio Showcase Document for Recruiters & Managers

---

## 📊 PROJECT OVERVIEW

**Project:** Fine-tune Llama 3.2 3B on Kaggle T4 x2 for Legal Clause Extraction  
**Duration:** Oct 28-31, 2025 (~5 days)  
**Final Results:** 73.6% average similarity, 39% excellent matches (>90%)  
**Total Challenges Overcome:** 18+ across 5 domains  
**Success Rate:** 100% - Overcame every blocker

---

## 🔴 CRITICAL CHALLENGES (Project Blockers) - 3 Issues

### 1. CUAD Dataset Label Misalignment
**Severity:** 🔴 CRITICAL  
**Impact:** Initial model: 24.4% similarity (unusable)

**The Problem:**  
CUAD training data had fundamental flaw. Input contracts didn't contain output clause labels:
- Model asked to extract: "Insurance" clause
- Label provided: "diagnostic industry practices" (unrelated text)
- Result: Model trained on impossible task

**Root Cause Analysis:**  
- CUAD designed for Q&A, not extraction
- Labels are snippets mixed from different documents
- No alignment validation before training

**Solution Implemented:**
1. Diagnosed by analyzing 5 sample outputs in detail
2. Recognized pattern: labels don't match inputs
3. Researched and found ACORD dataset (expert-annotated by lawyers)
4. Reformatted ACORD from BEIR to instruction format
5. Re-ran complete training pipeline
6. Final result: **73.6% similarity (+202% improvement!)**

**Skills Demonstrated:**
- ✅ Root cause analysis methodology
- ✅ Dataset evaluation and comparison
- ✅ Research (found ACORD alternative)
- ✅ Data format conversion
- ✅ Resilience (retrained entire pipeline)

---

### 2. Llama 3.2 Configuration Bug (404 Error)
**Severity:** 🔴 CRITICAL  
**Error:** `RemoteEntryNotFoundError: 404 - additional_chat_templates not found`

**The Problem:**  
When loading model in new Kaggle notebook, 404 error blocked deployment entirely. Model config tried to load non-existent file.

**Root Cause:**  
Llama 3.2-3B-Instruct config references file that doesn't exist on HuggingFace. Upstream bug with no patch available.

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

### 3. Initial Training: 0% Improvement
**Severity:** 🔴 CRITICAL  
**Impact:** 2.5 hours training produced unusable results

**The Problem:**  
First training attempt (CUAD): model output "There is no clause found" instead of extracting. Generated conversational responses, not extractions.

**Root Cause:**  
Llama 3.2 Instruct trained for conversational responses. System prompt "You are a legal AI assistant" primed model for explanation, not extraction. Model learned to analyze rather than copy.

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

## 🟠 MAJOR CHALLENGES (High Complexity) - 6 Issues

### 4. Evaluation Showed 0 Trainable Parameters
**Severity:** 🟠 HIGH  
**Concern:** "Did my fine-tuning disappear? Did 2.9 hours go to waste?"

**Solution:**
- Explained model.eval() freezes parameters for inference safety
- Provided verification tests proving fine-tuning still active
- Showed model still produces 73.6% performance
- Demonstrated with multiple verification methods

**Skills:** PyTorch deep knowledge, technical communication, debugging methodology

---

### 5. Hardware Constraints: 3B Model on 32GB GPU
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

### 6. Kaggle Session Management
**Severity:** 🟠 HIGH  
**Problem:** Path changed from /kaggle/working/ to /kaggle/input/

**Solution:**
- Documented 2 separate workflows
- Created auto-detecting "Find Path" cell
- Explained Kaggle file system architecture
- Provided debugging tools for path verification

**Skills:** Cloud platform expertise, file system navigation, process documentation

---

### 7. Library Dependency Hell
**Severity:** 🟠 HIGH  
**Challenge:** Complex version interdependencies crashed training

**Specific Issues:**
- transformers 4.47.0 ↔ peft 0.13.2 incompatible
- accelerate 1.12.0 had FSDP issues
- bitsandbytes 0.46 had CUDA problems

**Solution:**
- Pinned exact versions: transformers==4.46.3, peft==0.13.2, etc.
- Uninstalled conflicting versions first
- Created environment setup cell for all notebooks
- Tested full environment before GPU jobs

**Skills:** Dependency management, version conflict resolution, environment reproducibility

---

### 8. Loss Metrics Didn't Predict Quality
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

## 🟡 MODERATE CHALLENGES (Important but Manageable) - 6+ Issues

### 9. Dataset Format Conversion
ACORD (BEIR format) → Instruction format for SFT training
- Created mapping: query→instruction, corpus→output
- Implemented validation checks
- Result: 3000-9000 training samples (after augmentation)

### 10. Hyperparameter Tuning
- LR: 2e-4 → 1e-4 (50% reduction)
- Warmup: 50 → 100 steps
- Result: Loss converged to 0.407

### 11. Data Augmentation
- Instruction template augmentation (5 formats)
- 3x multiplication for small dataset
- Result: Prevented overfitting on 114 queries

### 12. Multi-GPU Configuration
- Enabled DDP (Distributed Data Parallel)
- Verified both GPUs active
- Monitored utilization

### 13. Prompt Format Consistency
- Standardized training/eval prompts
- Matched system/user/assistant structure
- Validated on samples

### 14. Documentation
- Created comprehensive README
- Performance metrics, training details, usage examples
- Professional deployment standard

---

## 🟢 MINOR CHALLENGES (Best Practices) - 5+ Issues

- Kaggle internet configuration
- CUDA memory cache management
- Tokenizer padding configuration
- Evaluation metrics selection
- And more...

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

## 💪 SKILLS DEVELOPED

✅ Problem diagnosis & root cause analysis  
✅ Deep ML/LLM system understanding  
✅ Data science & engineering  
✅ DevOps & cloud platforms (Kaggle)  
✅ Hardware optimization (VRAM, GPU)  
✅ Environment management & debugging  
✅ Prompt engineering & instruction-tuning  
✅ Research & alternative solution finding  
✅ Resilience & persistence under pressure  
✅ Technical communication & documentation  

---

## 🎓 KEY LEARNINGS

1. **Data quality > code quality**  
   Perfect code + bad data = bad results. Always validate data alignment.

2. **Instruction-tuned models require directive language**  
   Small prompt differences = huge behavior changes.

3. **Loss curves only meaningful with good data**  
   Always validate on actual task metrics, not just loss.

4. **QLoRA essential for large models on limited GPUs**  
   Hardware calculation critical before model selection.

5. **Version pinning is critical in LLM ecosystem**  
   Always test environment before expensive GPU jobs.

6. **Cloud platforms have unique architectures**  
   Document platform-specific patterns.

---

## 👔 IMPACT FOR RECRUITERS/MANAGERS

This project demonstrates:

1. **RESILIENCE:** Overcame 3 critical blockers without giving up
2. **PROBLEM SOLVING:** Diagnosed and fixed 18+ distinct issues
3. **TECHNICAL DEPTH:** Understands ML, LLMs, PyTorch, cloud platforms
4. **RESEARCH SKILLS:** Found ACORD after CUAD failed
5. **OWNERSHIP:** Full responsibility for entire pipeline
6. **COMMUNICATION:** Documented all solutions professionally
7. **PERSISTENCE:** Retrained entire pipeline when data issues found
8. **LEARNING:** Acquired new skills (QLoRA, prompt engineering, Kaggle)

**This is what a strong junior ML engineer looks like:**
- ✅ Handles unexpected problems confidently
- ✅ Doesn't panic when things break
- ✅ Researches and finds creative solutions
- ✅ Documents decisions for reproducibility
- ✅ Delivers results despite obstacles
- ✅ Shows initiative and ownership

---

## 📝 PROJECT TIMELINE

**Oct 28 (Day 1):** Initial training with CUAD → 24.4% (realized data issue)  
**Oct 29 (Day 2):** Root cause analysis → Diagnosed CUAD misalignment  
**Oct 29 (Day 3):** Research and switch to ACORD dataset  
**Oct 30 (Day 4):** Retrain with ACORD → 73.6% similarity ✅  
**Oct 31 (Day 5):** Deployment setup, documentation, verification  

**Total:** Completed in 5 days, overcame every obstacle, 100% success rate

---

## 🚀 READY FOR PRODUCTION

- ✅ Model uploaded to HuggingFace Hub
- ✅ Gradio demo ready for testing
- ✅ Comprehensive documentation
- ✅ All challenges documented and resolved
- ✅ Results verified and reproducible

---

**Use this document to impress recruiters, managers, and interviewers!**
